if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Model do
    @moduledoc """
    Differentiable FSRS-6 forward model.

    Mirrors the stability, difficulty and retrievability math of
    `ExFsrs.Scheduler`, expressed with `Nx` so that gradients flow back to the
    21 model weights. This is the single copy of that math on the optimizer
    side: `ExFsrs.Optimizer.Model.Batched` and `ExFsrs.Optimizer.Model.Loop`
    call these functions on whole columns of cards at once.

    Only a subset of the scheduler is reproduced here. In FSRS-6 the card state
    machine (`:learning`/`:review`/`:relearning`), the step index, due dates,
    intervals and fuzzing never influence stability, difficulty or
    retrievability — see `ExFsrs.Scheduler.review_card/5`, whose memory-state
    update reads only stability, difficulty and days-since-last-review. That
    makes the trainable model a pure function of
    `{stability, difficulty, elapsed_days, rating}`.

    `rating` and `elapsed_days` may be plain integers (one card) or integer
    tensors (a column of cards); every function broadcasts. Where a scalar
    rating lets a branch be skipped outright, a clause does so; the tensor
    clause evaluates both branches and selects, since per-element branching is
    not available.

    ## Kernel operators

    These functions run inside `Nx.Defn.value_and_grad/2` outside of `defn`,
    where `Kernel` operators are unavailable. Every arithmetic operation on a
    tensor must therefore go through an `Nx` function (`Nx.multiply/2`,
    `Nx.negate/1`, ...) rather than `*` or `-`. That also makes them callable
    from inside `defn` during tracing.
    """

    @stability_min 0.001
    @min_difficulty 1.0
    @max_difficulty 10.0

    # A bare float literal becomes an f32 tensor, so `Nx.pow(0.9, w)` rounds 0.9
    # through f32 before promoting the result to f64 — enough to shift
    # retrievability in the 8th decimal place. Every float constant goes through
    # here to stay in f64. Integer literals convert exactly and are used as-is.
    defp c(value), do: Nx.tensor(value, type: :f64)

    @doc "Forgetting-curve decay, `-w[20]`."
    def decay(params), do: Nx.negate(params[20])

    @doc "Forgetting-curve factor, `0.9^(1/decay) - 1`."
    def factor(params) do
      Nx.subtract(Nx.pow(c(0.9), Nx.divide(1, decay(params))), 1)
    end

    @doc """
    Predicted probability of recall after `elapsed_days` at the given stability.

    Both `decay` and `factor` derive from `w[20]`, which is itself trained, so
    gradients flow through them too. Negative elapsed days (the `-1` that marks
    a first review) are floored at 0.
    """
    def retrievability(params, stability, elapsed_days) do
      elapsed = Nx.as_type(Nx.max(elapsed_days, 0), :f64)

      Nx.pow(
        Nx.add(1, Nx.divide(Nx.multiply(factor(params), elapsed), stability)),
        decay(params)
      )
    end

    @doc "Initial stability for a card's first review, `w[rating - 1]`."
    def initial_stability(params, rating) do
      Nx.max(Nx.take(params, Nx.subtract(rating, 1)), c(@stability_min))
    end

    @doc "Initial difficulty for a card's first review, clamped to 1.0..10.0."
    def initial_difficulty(params, rating) do
      params
      |> initial_difficulty_unclamped(rating)
      |> Nx.clip(c(@min_difficulty), c(@max_difficulty))
    end

    defp initial_difficulty_unclamped(params, rating) do
      Nx.add(Nx.subtract(params[4], Nx.exp(Nx.multiply(params[5], Nx.subtract(rating, 1)))), 1)
    end

    @doc "Difficulty after a review, with linear damping and mean reversion."
    def next_difficulty(params, difficulty, rating) do
      delta = Nx.negate(Nx.multiply(params[6], Nx.subtract(rating, 3)))
      damped = Nx.divide(Nx.multiply(Nx.subtract(c(10.0), difficulty), delta), c(9.0))

      baseline = initial_difficulty_unclamped(params, 4)
      reverted = Nx.add(difficulty, damped)

      params[7]
      |> Nx.multiply(baseline)
      |> Nx.add(Nx.multiply(Nx.subtract(1, params[7]), reverted))
      |> Nx.clip(c(@min_difficulty), c(@max_difficulty))
    end

    @doc """
    Stability after a same-day (< 1 day elapsed) review.

    A successful same-day review never shrinks stability; only Again can.
    """
    def short_term_stability(params, stability, rating) do
      increase =
        Nx.multiply(
          Nx.exp(Nx.multiply(params[17], Nx.add(Nx.subtract(rating, 3), params[18]))),
          Nx.pow(stability, Nx.negate(params[19]))
        )

      increase = Nx.select(Nx.not_equal(rating, 1), Nx.max(increase, c(1.0)), increase)

      Nx.max(Nx.multiply(stability, increase), c(@stability_min))
    end

    @doc """
    Stability after a review at least a day later.

    With a scalar rating only the branch that applies is evaluated. With a
    tensor rating both are evaluated for every element and selected between,
    so both must stay finite on every input — see the padding notes in
    `ExFsrs.Optimizer.Model.Batched`.
    """
    def next_stability(params, difficulty, stability, retrievability, rating)

    def next_stability(params, difficulty, stability, retrievability, 1) do
      Nx.max(forget_stability(params, difficulty, stability, retrievability), c(@stability_min))
    end

    def next_stability(params, difficulty, stability, retrievability, rating)
        when is_integer(rating) do
      Nx.max(
        recall_stability(params, difficulty, stability, retrievability, rating),
        c(@stability_min)
      )
    end

    def next_stability(params, difficulty, stability, retrievability, rating) do
      forget = forget_stability(params, difficulty, stability, retrievability)
      recall = recall_stability(params, difficulty, stability, retrievability, rating)

      Nx.max(Nx.select(Nx.equal(rating, 1), forget, recall), c(@stability_min))
    end

    # After a lapse, stability is the smaller of the long-term post-lapse
    # formula and the short-term Again result, so a lapse never raises it.
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

    @doc """
    Advances one card's `{stability, difficulty}` by one review.

    Pass `nil` as the state for a card's first review. Mirrors the three-way
    branch in `ExFsrs.Scheduler`: first review, same-day review, or a review a
    day or more later.
    """
    def step(params, state, rating, elapsed_days)

    def step(params, nil, rating, _elapsed_days) do
      {initial_stability(params, rating), initial_difficulty(params, rating)}
    end

    def step(params, {stability, difficulty}, rating, elapsed_days) when elapsed_days < 1 do
      {short_term_stability(params, stability, rating),
       next_difficulty(params, difficulty, rating)}
    end

    def step(params, {stability, difficulty}, rating, elapsed_days) do
      retrievability = retrievability(params, stability, elapsed_days)

      {next_stability(params, difficulty, stability, retrievability, rating),
       next_difficulty(params, difficulty, rating)}
    end
  end
end
