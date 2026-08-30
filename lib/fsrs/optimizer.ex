if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer do
    @moduledoc """
    Trains FSRS-6 parameters on a collection of review logs.

    A port of py-fsrs's `Optimizer`. FSRS is treated as a many-to-many sequence
    model: each card's review history is a sequence, the input at each step is
    the time of a review, the output is the predicted retrievability, and the
    target is whether the card was actually recalled. The 21 model weights are
    what gets optimized.

        parameters = ExFsrs.Optimizer.compute_optimal_parameters(review_logs)
        scheduler = ExFsrs.Scheduler.new(parameters: parameters)

    Requires the optional `:nx` dependency; without it this module is not
    compiled and scheduling stays dependency-free.

    ## Scope

    This reproduces py-fsrs, which starts from the default parameters and runs
    plain Adam on binary cross-entropy. The `fsrs-optimizer` Python package and
    `fsrs-rs` (what Anki ships) do more: they fit `w[0..3]` from data before
    training, add an L2 penalty toward the starting weights, weight samples by
    recency, and drop outlier intervals. Those are not implemented here.
    `ExFsrs.Optimizer.Loss.l2_penalty/3` exists as the seam for the second.
    """

    alias ExFsrs.Optimizer.Adam
    alias ExFsrs.Optimizer.Data
    alias ExFsrs.Optimizer.Loss
    alias ExFsrs.Optimizer.Model
    alias ExFsrs.Optimizer.Model.Batched
    alias ExFsrs.Optimizer.Model.Loop

    @num_epochs 5
    @mini_batch_size 512
    @learning_rate 4.0e-2

    # From py-fsrs's scheduler.py. `ExFsrs.Scheduler` does not enforce these;
    # the optimizer clamps to them after every gradient step so that training
    # cannot wander into parameters that produce nonsense schedules.
    @lower_bounds [
      0.001,
      0.001,
      0.001,
      0.001,
      1.0,
      0.001,
      0.001,
      0.001,
      0.0,
      0.0,
      0.001,
      0.001,
      0.001,
      0.001,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      0.1
    ]

    @upper_bounds [
      100.0,
      100.0,
      100.0,
      100.0,
      10.0,
      4.0,
      4.0,
      0.75,
      4.5,
      0.8,
      3.5,
      5.0,
      0.25,
      0.9,
      4.0,
      1.0,
      6.0,
      2.0,
      2.0,
      0.8,
      0.8
    ]

    @doc "Lower bounds for each of the 21 parameters."
    def lower_bounds, do: @lower_bounds

    @doc "Upper bounds for each of the 21 parameters."
    def upper_bounds, do: @upper_bounds

    @doc """
    Computes optimized parameters from review logs.

    Accepts `%ExFsrs.ReviewLog{}` structs, `{card_id, rating, review_datetime}`
    tuples, any enumerable of either, or sequences already built by
    `ExFsrs.Optimizer.Data.build_sequences/1` or
    `ExFsrs.Optimizer.Data.build_sequences_from_elapsed/1`. Returns a list of 21
    floats suitable for `ExFsrs.Scheduler.new/1`.

    Returns the default parameters unchanged when there is too little data —
    fewer than #{@mini_batch_size} reviews that follow an earlier review of the
    same card by at least a day.

    ## Options

      * `:card_orders` — a list of card-id orderings, one per epoch, replacing
        the random shuffle. Used by the test suite to replay py-fsrs's exact
        ordering; the ordering fixes the minibatch boundaries and therefore the
        whole optimization trajectory.

      * `:on_step` — a one-argument function called after each gradient step
        with `%{step:, loss:, gradient:, params:}`. Useful for progress
        reporting and for comparing a run against a reference trace.

      * `:model` — `:batched` (default) advances every card in a minibatch
        together; `:scalar` walks reviews one at a time. Same arithmetic, and
        the two are diffed against each other in the test suite, but `:scalar`
        is markedly slower. It is kept as the reference implementation.

      * `:compiler` — an `Nx.Defn` compiler such as `EXLA` for the batched
        model. Minibatch shapes are rounded onto a small ladder so one compiled
        executable is reused across many minibatches.
    """
    def compute_optimal_parameters(logs, opts \\ []) do
      sequences = to_sequences(logs)
      num_reviews = Data.num_reviews(sequences)

      if num_reviews < @mini_batch_size do
        ExFsrs.Scheduler.new().parameters
      else
        train(sequences, num_reviews, opts)
      end
    end

    @doc """
    Mean binary cross-entropy over every scored review at the given parameters.

    Takes review logs in either accepted form, or sequences already built by
    `ExFsrs.Optimizer.Data.build_sequences/1`.

    Deterministic: no shuffling and no gradient steps, so this is the cheapest
    way to compare two parameter sets on the same collection.
    """
    def batch_loss(logs_or_sequences, parameters)

    def batch_loss(logs_or_sequences, parameters) when is_list(parameters) do
      batch_loss(logs_or_sequences, Nx.tensor(parameters, type: :f64))
    end

    def batch_loss(logs_or_sequences, %Nx.Tensor{} = params) do
      {total, count} =
        logs_or_sequences
        |> to_sequences()
        |> Enum.reduce({Nx.tensor(0.0, type: :f64), 0}, fn {_card_id, reviews}, {total, count} ->
          {loss, _state, scored} = replay(params, reviews, nil)
          {Nx.add(total, loss), count + scored}
        end)

      if count == 0 do
        raise ArgumentError, "no reviews to score: every review is a first or same-day review"
      end

      Nx.to_number(total) / count
    end

    # Already-built sequences pass straight through, so callers scoring several
    # parameter sets against one collection do not rebuild them each time.
    defp to_sequences([{_card_id, [%Data.Review{} | _]} | _] = sequences), do: sequences

    defp to_sequences(logs), do: Data.build_sequences(logs)

    defp train(sequences, num_reviews, opts) do
      t_max = ceil(num_reviews / @mini_batch_size) * @num_epochs
      params = Nx.tensor(ExFsrs.Scheduler.new().parameters, type: :f64)
      adam = Adam.new(21, learning_rate: @learning_rate, t_max: t_max)

      card_orders = Keyword.get(opts, :card_orders)
      lower = Nx.tensor(@lower_bounds, type: :f64)
      upper = Nx.tensor(@upper_bounds, type: :f64)

      on_step = Keyword.get(opts, :on_step, fn _ -> :ok end)
      model = Keyword.get(opts, :model, :batched)

      compiler = Keyword.get(opts, :compiler)

      if model == :loop and not Loop.supported?() do
        raise ArgumentError, """
        `model: :loop` needs an Nx that computes f64 gradients through `while`
        correctly, and the installed one does not.

        Nx returns silently wrong gradients — often zeros — when an f64 adjoint
        is scaled inside a `while`. Nothing raises, so training would otherwise
        proceed on garbage. See bench/NX_WHILE_GRAD_F64.md.

        Use `model: :batched` (the default), which does not rely on `while`.
        """
      end

      by_id = Map.new(sequences)
      card_ids = Enum.map(sequences, fn {card_id, _reviews} -> card_id end)

      initial = {params, adam, card_ids, nil, :infinity, 0}

      {_params, _adam, _ids, best_params, _best_loss, _step} =
        Enum.reduce(0..(@num_epochs - 1), initial, fn epoch,
                                                      {params, adam, card_ids, best, best_loss,
                                                       step} ->
          card_ids = epoch_order(card_orders, epoch, card_ids)

          ordered = Enum.map(card_ids, fn card_id -> {card_id, Map.fetch!(by_id, card_id)} end)

          {params, adam, step} =
            run_epoch(params, adam, ordered, lower, upper, step, on_step, {model, compiler})

          loss = batch_loss(sequences, params)

          if loss < best_loss do
            {params, adam, card_ids, Nx.to_flat_list(params), loss, step}
          else
            {params, adam, card_ids, best, best_loss, step}
          end
        end)

      best_params
    end

    defp epoch_order(nil, _epoch, card_ids), do: Enum.shuffle(card_ids)

    defp epoch_order(card_orders, epoch, _card_ids), do: Enum.at(card_orders, epoch)

    # One epoch: cut the ordered reviews into minibatches of @mini_batch_size
    # scored reviews and take a gradient step after each.
    defp run_epoch(params, adam, ordered, lower, upper, step, on_step, {model, compiler}) do
      ordered
      |> chunk()
      |> Enum.reduce({params, adam, nil, step}, fn chunk, {params, adam, carry, step} ->
        prepared = prepare_chunk(model, chunk, carry)

        {loss, gradient} = chunk_value_and_grad(model, params, prepared, compiler)

        # The state a card carries across a minibatch boundary is computed with
        # the parameters in force during that minibatch and then detached, as
        # torch does. Recomputing it outside the traced function is what
        # detaches it; a forward pass costs ~2% of the gradient pass.
        {_loss, carry_out} = chunk_forward(model, params, prepared, compiler)

        {adam, params} = Adam.step(adam, params, gradient)
        # Nx.clip/3 takes scalar bounds; these are per-parameter.
        params = Nx.min(Nx.max(params, lower), upper)

        # Keep the parameters on the default backend. A compiler hands back
        # device-backed tensors, and Adam — plain Nx outside defn — passes that
        # backend on to the parameters. Everything downstream that is not
        # compiled then dispatches op by op to the device: batch_loss walks
        # 12,580 reviews as individual scalar operations and goes from 0.5s to
        # 53s. These are 21 floats.
        params = Nx.backend_copy(params, Nx.BinaryBackend)

        on_step.(%{
          step: step,
          loss: Nx.to_number(loss),
          gradient: Nx.to_flat_list(gradient),
          params: Nx.to_flat_list(params)
        })

        {params, adam, carry_out, step + 1}
      end)
      |> then(fn {params, adam, _carry, step} -> {params, adam, step} end)
    end

    # The scalar model consumes the chunk directly; the batched one needs its
    # padded tensors built first, which is parameter-independent and so is done
    # once and reused by both the gradient and carry passes.
    defp prepare_chunk(:scalar, chunk, carry), do: {chunk, carry}
    defp prepare_chunk(:batched, chunk, carry), do: Batched.prepare(chunk, carry)
    defp prepare_chunk(:loop, chunk, carry), do: Loop.prepare(chunk, carry)

    defp chunk_value_and_grad(:scalar, params, prepared, _compiler) do
      Nx.Defn.value_and_grad(params, fn p ->
        {loss, _carry_out} = chunk_forward(:scalar, p, prepared, nil)
        loss
      end)
    end

    defp chunk_value_and_grad(:batched, params, prepared, _compiler) do
      Batched.value_and_grad(params, prepared)
    end

    defp chunk_value_and_grad(:loop, params, prepared, compiler) do
      Loop.loss_and_grad(params, prepared, compiler)
    end

    defp chunk_forward(:scalar, params, {chunk, carry}, _compiler),
      do: chunk_loss(params, chunk, carry)

    defp chunk_forward(:batched, params, batch, _compiler) do
      {loss, stability, difficulty} = Batched.run(params, batch)

      {loss, {stability, difficulty}}
    end

    defp chunk_forward(:loop, params, batch, compiler) do
      {loss, stability, difficulty} = Loop.run(params, batch, compiler)

      {loss, {stability, difficulty}}
    end

    defp chunk_loss(params, chunk, carry) do
      Enum.reduce(chunk, {Nx.tensor(0.0, type: :f64), carry}, fn segment, {total, carry} ->
        state = if segment.continues?, do: carry, else: nil
        {loss, state, _scored} = replay(params, segment.reviews, state)
        {Nx.add(total, loss), state}
      end)
    end

    # Replays a card's reviews, returning {summed_loss, end_state, scored_count}.
    defp replay(params, reviews, state) do
      Enum.reduce(reviews, {Nx.tensor(0.0, type: :f64), state, 0}, fn review,
                                                                      {total, state, scored} ->
        {total, scored} =
          if review.counts_for_loss? do
            {stability, _difficulty} = state
            prediction = Model.retrievability(params, stability, review.elapsed_days)
            {Nx.add(total, Loss.binary_cross_entropy(prediction, review.label)), scored + 1}
          else
            {total, scored}
          end

        {total, Model.step(params, state, review.rating, review.elapsed_days), scored}
      end)
    end

    @doc """
    The minibatches of one epoch, as lists of segments.

    Exposed so the batched model can be diffed against the scalar one on exactly
    the chunks the optimizer would produce.
    """
    def minibatches(sequences, card_order) do
      by_id = Map.new(sequences)

      card_order
      |> Enum.map(fn card_id -> {card_id, Map.fetch!(by_id, card_id)} end)
      |> chunk()
    end

    @doc """
    Sums a chunk's loss with the scalar model, returning `{loss, carry_state}`.

    The reference implementation the batched model is checked against.
    """
    def scalar_chunk_loss(params, chunk, carry), do: chunk_loss(params, chunk, carry)

    @doc """
    The number of scored reviews in each minibatch of one epoch.

    Exposed for tests: the minibatch partition depends only on the data and the
    card ordering, never on the parameters, so it can be checked against a
    reference trace without running any gradients.
    """
    def minibatch_sizes(sequences, card_order) do
      by_id = Map.new(sequences)

      card_order
      |> Enum.map(fn card_id -> {card_id, Map.fetch!(by_id, card_id)} end)
      |> chunk()
      |> Enum.map(fn chunk ->
        Enum.reduce(chunk, 0, fn segment, total ->
          total + Enum.count(segment.reviews, & &1.counts_for_loss?)
        end)
      end)
    end

    # Splits the epoch's cards into minibatches of exactly @mini_batch_size
    # scored reviews. A cut can land mid-card, in which case that card's
    # remaining reviews open the next minibatch as a continuing segment.
    defp chunk(ordered) do
      {chunks, current, _count} =
        Enum.reduce(ordered, {[], [], 0}, fn {_card_id, reviews}, acc ->
          split_card(reviews, false, acc)
        end)

      chunks = if current == [], do: chunks, else: [Enum.reverse(current) | chunks]

      Enum.reverse(chunks)
    end

    defp split_card(reviews, continues?, {chunks, current, count}) do
      {taken, count, rest} = take_until_full(reviews, count, [])

      current = [%{reviews: taken, continues?: continues?} | current]

      cond do
        # The minibatch filled part-way through this card: close it, and let
        # the card's remaining reviews open the next one.
        count == @mini_batch_size and rest != [] ->
          split_card(rest, true, {[Enum.reverse(current) | chunks], [], 0})

        # The minibatch filled exactly on this card's last review. Close it
        # here too — otherwise the next card would keep extending a minibatch
        # that is already full.
        count == @mini_batch_size ->
          {[Enum.reverse(current) | chunks], [], 0}

        true ->
          {chunks, current, count}
      end
    end

    defp take_until_full([], count, taken), do: {Enum.reverse(taken), count, []}

    defp take_until_full([review | rest], count, taken) do
      count = if review.counts_for_loss?, do: count + 1, else: count
      taken = [review | taken]

      if count == @mini_batch_size do
        {Enum.reverse(taken), count, rest}
      else
        take_until_full(rest, count, taken)
      end
    end
  end
end
