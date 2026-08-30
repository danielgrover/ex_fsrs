if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Model.Batched do
    @moduledoc """
    Vectorized form of `ExFsrs.Optimizer.Model`.

    Computes the same arithmetic as the scalar model but with every card in a
    minibatch advanced together: one tensor operation per timestep on a batch of
    cards, rather than one scalar operation per review. That cuts the size of the
    traced expression graph by roughly the batch width, which is what dominates
    the runtime under `Nx`.

    Semantics are unchanged — same per-review loss, same minibatch boundaries,
    same three-way stability branch. Only the *order* in which the losses and
    gradients are summed differs, which perturbs results by around 1e-15. The
    optimizer propagates that noise linearly with a gain below 1, so final
    parameters shift by about the same amount.

    ## Padding and branch safety

    Card histories are ragged, so sequences are padded to the longest in the
    batch. Two rules keep padding from corrupting the result:

    * Every element carries its state forward unchanged on an invalid timestep,
      so padding cannot advance a finished card.
    * Both stability branches are evaluated for *every* element and then
      selected between, because per-element branching is not available. Unused
      lanes must therefore still produce finite numbers: a NaN in a discarded
      branch would survive `Nx.select/3` into the gradient, since `0 * NaN` is
      NaN. Padding and not-yet-started lanes are seeded with stability 1.0 and
      difficulty 1.0 rather than zero for exactly this reason.
    """

    alias ExFsrs.Optimizer.Loss

    @stability_min 0.001
    @min_difficulty 1.0
    @max_difficulty 10.0

    # See ExFsrs.Optimizer.Model: bare float literals become f32 tensors.
    defp c(value), do: Nx.tensor(value, type: :f64)

    # Sequences are padded to the longest in their batch, so mixing a 1-review
    # card with a 54-review one wastes most of the work. Sorting by length and
    # splitting into buckets of this size keeps each batch's rows close in
    # length; on the reference collection it cuts padded steps by ~2.5x.
    @bucket_size 32

    # EXLA compiles one executable per distinct input shape, so a bucket's
    # dimensions are rounded up onto these ladders and the slack is filled with
    # masked-out rows and timesteps. That caps the whole run at a handful of
    # compilations instead of one per minibatch. Under the default backend the
    # rounding only adds a little masked work.
    @row_sizes [4, 8, 16, 32]
    @length_sizes [2, 4, 8, 16, 32, 64]

    defp round_up(value, ladder) do
      Enum.find(ladder, fn candidate -> candidate >= value end) || value
    end

    defmodule Bucket do
      @moduledoc false
      defstruct [
        :ratings,
        :elapsed,
        :labels,
        :scored,
        :weights,
        :valid,
        :stability0,
        :difficulty0,
        :has_state0,
        :indices,
        :size,
        :length
      ]
    end

    defstruct [:buckets, :last_index]

    @doc """
    Builds the padded tensors for one minibatch.

    Independent of the parameters, so a chunk is prepared once and reused for
    both the gradient pass and the carry-state forward pass.

    `carry` is the `{stability, difficulty}` a card carries in from the previous
    minibatch, or `nil`. Only a chunk's first segment can continue a card.

    Segments are reordered into length buckets. Order does not affect the summed
    loss beyond float association, but two positions must be tracked through the
    reordering: the segment that receives the incoming carry (the chunk's first,
    when it continues a card) and the one that supplies the outgoing carry (the
    chunk's last).
    """
    def prepare(chunk, carry) do
      continues? = match?([%{continues?: true} | _rest], chunk)

      buckets =
        chunk
        |> Enum.with_index()
        |> Enum.sort_by(fn {segment, _index} -> length(segment.reviews) end)
        |> Enum.chunk_every(@bucket_size)
        |> Enum.map(&build_bucket(&1, carry, continues?))

      %__MODULE__{buckets: buckets, last_index: length(chunk) - 1}
    end

    defp build_bucket(indexed_segments, carry, continues?) do
      indices = Enum.map(indexed_segments, fn {_segment, index} -> index end)
      segments = Enum.map(indexed_segments, fn {segment, _index} -> segment end)

      max_length =
        segments |> Enum.map(&length(&1.reviews)) |> Enum.max() |> round_up(@length_sizes)

      size = round_up(length(segments), @row_sizes)

      # Slack rows are all-padding: no valid timesteps, so they contribute no
      # loss and never advance. They exist only to reach a canonical shape.
      rows =
        Enum.map(segments, &pad_row(&1.reviews, max_length)) ++
          List.duplicate(pad_row([], max_length), size - length(segments))

      carry_row = if continues?, do: Enum.find_index(indices, &(&1 == 0))
      {stability0, difficulty0, has_state0} = initial_columns(size, carry_row, carry)

      %Bucket{
        ratings: rows |> Enum.map(& &1.ratings) |> Nx.tensor(type: :s64),
        elapsed: rows |> Enum.map(& &1.elapsed) |> Nx.tensor(type: :s64),
        labels: rows |> Enum.map(& &1.labels) |> Nx.tensor(type: :f64),
        scored: rows |> Enum.map(& &1.scored) |> Nx.tensor(type: :f64),
        weights: rows |> Enum.map(& &1.weights) |> Nx.tensor(type: :f64),
        valid: rows |> Enum.map(& &1.valid) |> Nx.tensor(type: :f64),
        stability0: stability0,
        difficulty0: difficulty0,
        has_state0: has_state0,
        indices: indices,
        size: size,
        length: max_length
      }
    end

    # Padding values are chosen to be numerically harmless rather than zero:
    # rating 3 and elapsed 1 keep every branch finite, and the masks discard the
    # result anyway.
    defp pad_row(reviews, max_length) do
      padding = max_length - length(reviews)

      %{
        ratings: Enum.map(reviews, & &1.rating) ++ List.duplicate(3, padding),
        elapsed: Enum.map(reviews, &max(&1.elapsed_days, -1)) ++ List.duplicate(1, padding),
        labels: Enum.map(reviews, & &1.label) ++ List.duplicate(1.0, padding),
        scored:
          Enum.map(reviews, &if(&1.counts_for_loss?, do: 1.0, else: 0.0)) ++
            List.duplicate(0.0, padding),
        weights: Enum.map(reviews, & &1.weight) ++ List.duplicate(1.0, padding),
        valid: List.duplicate(1.0, length(reviews)) ++ List.duplicate(0.0, padding)
      }
    end

    # A card with no prior state starts at stability 1.0 / difficulty 1.0 rather
    # than zero, so that the branches it does not take stay finite.
    defp initial_columns(size, nil, _carry) do
      {Nx.broadcast(c(1.0), {size}), Nx.broadcast(c(1.0), {size}), Nx.broadcast(c(0.0), {size})}
    end

    defp initial_columns(size, carry_row, {stability, difficulty}) do
      one_hot =
        0..(size - 1)
        |> Enum.map(&if(&1 == carry_row, do: 1.0, else: 0.0))
        |> Nx.tensor(type: :f64)

      carried? = Nx.equal(one_hot, 1)
      fresh = Nx.broadcast(c(1.0), {size})

      {Nx.select(carried?, Nx.broadcast(stability, {size}), fresh),
       Nx.select(carried?, Nx.broadcast(difficulty, {size}), fresh), one_hot}
    end

    @doc """
    Runs the batch, returning `{summed_loss, stability, difficulty}`.

    The state tensors are the carry for the next minibatch, taken from the
    chunk's last segment wherever bucketing placed it.
    """
    def run(params, %__MODULE__{} = prepared) do
      Enum.reduce(prepared.buckets, {c(0.0), nil, nil}, fn bucket,
                                                           {total, stability, difficulty} ->
        {loss, final_stability, final_difficulty} = run_bucket(params, bucket)

        case Enum.find_index(bucket.indices, &(&1 == prepared.last_index)) do
          nil -> {Nx.add(total, loss), stability, difficulty}
          row -> {Nx.add(total, loss), final_stability[row], final_difficulty[row]}
        end
      end)
    end

    @doc """
    Loss and gradient for a whole minibatch, summed over its buckets.

    Buckets within a minibatch are independent, so the gradient of the summed
    loss is the sum of the buckets' gradients.

    This model unrolls the timestep loop into the graph, so it is not usable with
    a compiler — the graph grows with sequence length and XLA's compile time
    grows superlinearly with it. `ExFsrs.Optimizer.Model.Loop` is the compilable
    form.
    """
    def value_and_grad(params, %__MODULE__{} = prepared) do
      zero_gradient = Nx.broadcast(c(0.0), Nx.shape(params))

      Enum.reduce(prepared.buckets, {c(0.0), zero_gradient}, fn bucket, {total, gradient} ->
        {loss, bucket_gradient} =
          Nx.Defn.value_and_grad(params, fn p ->
            {loss, _stability, _difficulty} = run_bucket(p, bucket)
            loss
          end)

        {Nx.add(total, loss), Nx.add(gradient, bucket_gradient)}
      end)
    end

    # The bucket already carries its columns and initial state, so it is passed
    # whole rather than unpacked into ten arguments.
    defp run_bucket(params, %Bucket{} = bucket) do
      initial = {c(0.0), bucket.stability0, bucket.difficulty0, bucket.has_state0}

      Enum.reduce(0..(bucket.length - 1), initial, fn t,
                                                      {total, stability, difficulty, has_state} ->
        step(params, bucket, t, total, stability, difficulty, has_state)
      end)
      |> then(fn {total, stability, difficulty, _has_state} -> {total, stability, difficulty} end)
    end

    defp step(params, bucket, t, total, stability, difficulty, has_state) do
      rating = bucket.ratings[[.., t]]
      elapsed = bucket.elapsed[[.., t]]
      valid = bucket.valid[[.., t]]
      scored = bucket.scored[[.., t]]
      weight = bucket.weights[[.., t]]
      label = bucket.labels[[.., t]]

      retrievability = retrievability(params, stability, elapsed)

      # Cross-entropy is evaluated on every lane and masked afterwards, but a
      # same-day or padding lane has elapsed == 0 and so a prediction of exactly
      # 1.0, where log(1 - 1) is -inf. binary_cross_entropy clamps that to a
      # finite *value*, yet the gradient of the log at that point is still
      # infinite, and multiplying it by a zero mask yields NaN rather than zero.
      # Substituting a benign prediction *before* the log keeps the gradient
      # finite in the lanes the mask discards. The unmasked retrievability is
      # still what feeds the stability branch below, which has no singularity
      # at elapsed == 0.
      scored? = Nx.equal(scored, 1)
      safe_retrievability = Nx.select(scored?, retrievability, c(0.5))

      total =
        Nx.add(
          total,
          Nx.sum(
            Loss.binary_cross_entropy(safe_retrievability, label)
            |> Nx.multiply(scored)
            |> Nx.multiply(weight)
          )
        )

      first? = Nx.equal(has_state, 0)
      same_day? = Nx.less(elapsed, 1)

      next_stability =
        Nx.select(
          first?,
          initial_stability(params, rating),
          Nx.select(
            same_day?,
            short_term_stability(params, stability, rating),
            long_term_stability(params, difficulty, stability, retrievability, rating)
          )
        )

      next_difficulty =
        Nx.select(
          first?,
          initial_difficulty(params, rating),
          next_difficulty(params, difficulty, rating)
        )

      valid? = Nx.equal(valid, 1)

      {total, Nx.select(valid?, next_stability, stability),
       Nx.select(valid?, next_difficulty, difficulty), Nx.max(has_state, valid)}
    end

    defp retrievability(params, stability, elapsed) do
      decay = Nx.negate(params[20])
      factor = Nx.subtract(Nx.pow(c(0.9), Nx.divide(1, decay)), 1)
      days = Nx.as_type(Nx.max(elapsed, 0), :f64)

      Nx.pow(Nx.add(1, Nx.divide(Nx.multiply(factor, days), stability)), decay)
    end

    defp initial_stability(params, rating) do
      Nx.max(Nx.take(params, Nx.subtract(rating, 1)), c(@stability_min))
    end

    defp initial_difficulty(params, rating) do
      params
      |> initial_difficulty_unclamped(Nx.as_type(rating, :f64))
      |> Nx.clip(c(@min_difficulty), c(@max_difficulty))
    end

    defp initial_difficulty_unclamped(params, rating) do
      Nx.add(Nx.subtract(params[4], Nx.exp(Nx.multiply(params[5], Nx.subtract(rating, 1)))), 1)
    end

    defp next_difficulty(params, difficulty, rating) do
      rating = Nx.as_type(rating, :f64)
      delta = Nx.negate(Nx.multiply(params[6], Nx.subtract(rating, 3)))
      damped = Nx.divide(Nx.multiply(Nx.subtract(c(10.0), difficulty), delta), c(9.0))

      arg1 = initial_difficulty_unclamped(params, c(4.0))
      arg2 = Nx.add(difficulty, damped)

      params[7]
      |> Nx.multiply(arg1)
      |> Nx.add(Nx.multiply(Nx.subtract(1, params[7]), arg2))
      |> Nx.clip(c(@min_difficulty), c(@max_difficulty))
    end

    defp short_term_stability(params, stability, rating) do
      rating_f = Nx.as_type(rating, :f64)

      increase =
        Nx.multiply(
          Nx.exp(Nx.multiply(params[17], Nx.add(Nx.subtract(rating_f, 3), params[18]))),
          Nx.pow(stability, Nx.negate(params[19]))
        )

      # A successful same-day review must never shrink stability.
      increase =
        Nx.select(Nx.not_equal(rating, 1), Nx.max(increase, c(1.0)), increase)

      Nx.max(Nx.multiply(stability, increase), c(@stability_min))
    end

    defp long_term_stability(params, difficulty, stability, retrievability, rating) do
      forget = forget_stability(params, difficulty, stability, retrievability)
      recall = recall_stability(params, difficulty, stability, retrievability, rating)

      Nx.max(Nx.select(Nx.equal(rating, 1), forget, recall), c(@stability_min))
    end

    defp forget_stability(params, difficulty, stability, retrievability) do
      long_term =
        params[11]
        |> Nx.multiply(Nx.pow(difficulty, Nx.negate(params[12])))
        |> Nx.multiply(Nx.subtract(Nx.pow(Nx.add(stability, 1), params[13]), 1))
        |> Nx.multiply(Nx.exp(Nx.multiply(Nx.subtract(1, retrievability), params[14])))

      short_term = Nx.divide(stability, Nx.exp(Nx.multiply(params[17], params[18])))

      Nx.min(long_term, short_term)
    end

    defp recall_stability(params, difficulty, stability, retrievability, rating) do
      hard_penalty = Nx.select(Nx.equal(rating, 2), params[15], c(1.0))
      easy_bonus = Nx.select(Nx.equal(rating, 4), params[16], c(1.0))

      increase =
        params[8]
        |> Nx.exp()
        |> Nx.multiply(Nx.subtract(11, difficulty))
        |> Nx.multiply(Nx.pow(stability, Nx.negate(params[9])))
        |> Nx.multiply(
          Nx.subtract(Nx.exp(Nx.multiply(Nx.subtract(1, retrievability), params[10])), 1)
        )
        |> Nx.multiply(hard_penalty)
        |> Nx.multiply(easy_bonus)

      Nx.multiply(stability, Nx.add(1, increase))
    end
  end
end
