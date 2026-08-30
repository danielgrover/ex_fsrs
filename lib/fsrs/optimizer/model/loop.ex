if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Model.Loop do
    @moduledoc """
    The batched model expressed as a real loop inside `defn`.

    `ExFsrs.Optimizer.Model.Batched` steps timesteps with `Enum.reduce`, which
    runs at trace time and therefore *inlines* one copy of the body per
    timestep. At 16 timesteps that is already 9,373 operations in a single
    straight-line graph, and XLA's compile time grows superlinearly in graph
    size — enough to make it unusable past a handful of timesteps.

    Here the timestep loop is a `while` generator, so the graph is a fixed size
    regardless of sequence length and a compiler has something small to work on.

    Two rules make this work, both learned the hard way:

    * Everything the loop touches must be a loop argument. `defn` cannot close
      over outside variables inside `while`.
    * The gradient must be taken *inside* `defn`. Calling
      `Nx.Defn.value_and_grad/2` from ordinary Elixir on a function containing
      `while` silently returns zeros rather than raising.

    Columns are stored transposed as `{timesteps, cards}` so that indexing by
    the loop variable yields one `{cards}` slice per step.
    """

    import Nx.Defn

    alias ExFsrs.Optimizer.Model.Batched

    @doc """
    Prepares a minibatch, reusing the batched model's padding and bucketing.

    Column tensors are transposed to `{timesteps, cards}` for loop indexing.
    """
    def prepare(chunk, carry) do
      prepared = Batched.prepare(chunk, carry)

      %{prepared | buckets: Enum.map(prepared.buckets, &transpose_bucket/1)}
    end

    defp transpose_bucket(bucket) do
      %{
        bucket
        | ratings: Nx.transpose(bucket.ratings),
          elapsed: Nx.transpose(bucket.elapsed),
          labels: Nx.transpose(bucket.labels),
          scored: Nx.transpose(bucket.scored),
          valid: Nx.transpose(bucket.valid)
      }
    end

    @doc """
    Loss and gradient for a minibatch, summed over its buckets.

    Buckets are independent, so the gradient of the summed loss is the sum of
    their gradients, and each can be compiled for its own shape.
    """
    def loss_and_grad(params, prepared, compiler \\ nil) do
      zero_gradient = Nx.broadcast(Nx.tensor(0.0, type: :f64), Nx.shape(params))

      Enum.reduce(prepared.buckets, {Nx.tensor(0.0, type: :f64), zero_gradient}, fn bucket,
                                                                                    {total,
                                                                                     gradient} ->
        {loss, bucket_gradient} = apply_bucket(&bucket_value_and_grad/9, params, bucket, compiler)

        {Nx.add(total, loss), Nx.add(gradient, bucket_gradient)}
      end)
    end

    @doc """
    Runs a minibatch forward, returning `{loss, stability, difficulty}` where the
    state tensors are the carry for the next minibatch.
    """
    def run(params, prepared, compiler \\ nil) do
      Enum.reduce(prepared.buckets, {Nx.tensor(0.0, type: :f64), nil, nil}, fn bucket,
                                                                               {total, stability,
                                                                                difficulty} ->
        {loss, final_stability, final_difficulty} =
          apply_bucket(&bucket_forward/9, params, bucket, compiler)

        case Enum.find_index(bucket.indices, &(&1 == prepared.last_index)) do
          nil ->
            {Nx.add(total, loss), stability, difficulty}

          row ->
            # Bring the carry back to the default backend. A compiler returns
            # device-backed tensors, and letting those seed the next minibatch's
            # `prepare/2` makes every tensor in it device-backed too, which turns
            # a 33ms compiled minibatch into a 2s one. It is two scalars.
            {Nx.add(total, loss), to_default(final_stability[row]),
             to_default(final_difficulty[row])}
        end
      end)
    end

    defp to_default(tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

    defp apply_bucket(fun, params, bucket, compiler) do
      args = [
        params,
        bucket.ratings,
        bucket.elapsed,
        bucket.labels,
        bucket.scored,
        bucket.valid,
        bucket.stability0,
        bucket.difficulty0,
        bucket.has_state0
      ]

      if compiler do
        Nx.Defn.jit_apply(fun, args, compiler: compiler)
      else
        apply(fun, args)
      end
    end

    # --- everything below runs inside defn ---

    defnp bucket_value_and_grad(params, ratings, elapsed, labels, scored, valid, s0, d0, h0) do
      value_and_grad(params, fn p ->
        {loss, _stability, _difficulty} =
          sequence(p, ratings, elapsed, labels, scored, valid, s0, d0, h0)

        loss
      end)
    end

    defnp bucket_forward(params, ratings, elapsed, labels, scored, valid, s0, d0, h0) do
      sequence(params, ratings, elapsed, labels, scored, valid, s0, d0, h0)
    end

    defnp sequence(params, ratings, elapsed, labels, scored, valid, s0, d0, h0) do
      {_params, _ratings, _elapsed, _labels, _scored, _valid, total, stability, difficulty,
       _has_state} =
        while {params, ratings, elapsed, labels, scored, valid,
               total = Nx.tensor(0.0, type: :f64), stability = s0, difficulty = d0,
               has_state = h0},
              i <- 0..(Nx.axis_size(ratings, 0) - 1) do
          {total, stability, difficulty, has_state} =
            step(
              params,
              ratings[i],
              elapsed[i],
              labels[i],
              scored[i],
              valid[i],
              total,
              stability,
              difficulty,
              has_state
            )

          {params, ratings, elapsed, labels, scored, valid, total, stability, difficulty,
           has_state}
        end

      {total, stability, difficulty}
    end

    defnp step(
            params,
            rating,
            elapsed,
            label,
            scored,
            valid,
            total,
            stability,
            difficulty,
            has_state
          ) do
      retrievability = retrievability(params, stability, elapsed)

      # A same-day or padding lane predicts exactly 1.0, and log(1 - 1) is -inf.
      # The value is clamped, but the gradient of the log there is not, and a
      # zero mask turns infinity into NaN. Substituting a benign prediction
      # before the log keeps the discarded lanes finite.
      scored? = Nx.equal(scored, 1)
      safe = Nx.select(scored?, retrievability, f64(0.5))
      total = total + Nx.sum(binary_cross_entropy(safe, label) * scored)

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

    # A bare float literal would become an f32 constant and lose precision before
    # being promoted, which shifts retrievability in the 8th decimal place.
    deftransformp(f64(value), do: Nx.tensor(value, type: :f64))

    defnp binary_cross_entropy(prediction, label) do
      log_p = Nx.max(Nx.log(prediction), f64(-100.0))
      log_1_p = Nx.max(Nx.log(1 - prediction), f64(-100.0))

      -(label * log_p + (1 - label) * log_1_p)
    end

    defnp retrievability(params, stability, elapsed) do
      decay = -params[20]
      factor = Nx.pow(f64(0.9), 1 / decay) - 1
      days = Nx.as_type(Nx.max(elapsed, 0), :f64)

      Nx.pow(1 + factor * days / stability, decay)
    end

    defnp initial_stability(params, rating) do
      Nx.max(Nx.take(params, rating - 1), f64(0.001))
    end

    defnp initial_difficulty(params, rating) do
      Nx.clip(initial_difficulty_unclamped(params, Nx.as_type(rating, :f64)), f64(1.0), f64(10.0))
    end

    defnp initial_difficulty_unclamped(params, rating) do
      params[4] - Nx.exp(params[5] * (rating - 1)) + 1
    end

    defnp next_difficulty(params, difficulty, rating) do
      rating = Nx.as_type(rating, :f64)
      delta = -(params[6] * (rating - 3))
      damped = (f64(10.0) - difficulty) * delta / f64(9.0)

      arg1 = initial_difficulty_unclamped(params, f64(4.0))
      arg2 = difficulty + damped

      Nx.clip(params[7] * arg1 + (1 - params[7]) * arg2, f64(1.0), f64(10.0))
    end

    defnp short_term_stability(params, stability, rating) do
      rating_f = Nx.as_type(rating, :f64)

      increase =
        Nx.exp(params[17] * (rating_f - 3 + params[18])) * Nx.pow(stability, -params[19])

      # A successful same-day review must never shrink stability.
      increase = Nx.select(Nx.not_equal(rating, 1), Nx.max(increase, f64(1.0)), increase)

      Nx.max(stability * increase, f64(0.001))
    end

    defnp long_term_stability(params, difficulty, stability, retrievability, rating) do
      forget = forget_stability(params, difficulty, stability, retrievability)
      recall = recall_stability(params, difficulty, stability, retrievability, rating)

      Nx.max(Nx.select(Nx.equal(rating, 1), forget, recall), f64(0.001))
    end

    defnp forget_stability(params, difficulty, stability, retrievability) do
      long_term =
        params[11] * Nx.pow(difficulty, -params[12]) *
          (Nx.pow(stability + 1, params[13]) - 1) *
          Nx.exp((1 - retrievability) * params[14])

      short_term = stability / Nx.exp(params[17] * params[18])

      Nx.min(long_term, short_term)
    end

    defnp recall_stability(params, difficulty, stability, retrievability, rating) do
      hard_penalty = Nx.select(Nx.equal(rating, 2), params[15], f64(1.0))
      easy_bonus = Nx.select(Nx.equal(rating, 4), params[16], f64(1.0))

      increase =
        Nx.exp(params[8]) * (11 - difficulty) * Nx.pow(stability, -params[9]) *
          (Nx.exp((1 - retrievability) * params[10]) - 1) *
          hard_penalty * easy_bonus

      stability * (1 + increase)
    end
  end
end
