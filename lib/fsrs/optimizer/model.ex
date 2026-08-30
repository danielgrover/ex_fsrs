if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Model do
    @moduledoc """
    Differentiable FSRS-6 forward model.

    Mirrors the stability, difficulty and retrievability math of
    `ExFsrs.Scheduler`, expressed with `Nx` so that gradients flow back to the
    21 model weights.

    Only a subset of the scheduler is reproduced here. In FSRS-6 the card state
    machine (`:learning`/`:review`/`:relearning`), the step index, due dates,
    intervals and fuzzing never influence stability, difficulty or
    retrievability — see `ExFsrs.Scheduler.compute_stability_difficulty/4`,
    which reads only stability, difficulty and days-since-last-review. That
    makes the trainable model a pure function of
    `{stability, difficulty, elapsed_days, rating}`.

    Ratings are plain integers (1..4) and `elapsed_days` a plain integer, since
    both are known before tracing; only the weights and the card state are
    tensors.

    ## Kernel operators

    These functions run inside `Nx.Defn.value_and_grad/2` outside of `defn`,
    where `Kernel` operators are unavailable. Every arithmetic operation on a
    tensor must therefore go through an `Nx` function (`Nx.multiply/2`,
    `Nx.negate/1`, ...) rather than `*` or `-`.
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
      d = decay(params)
      Nx.subtract(Nx.pow(c(0.9), Nx.divide(1, d)), 1)
    end

    @doc """
    Predicted probability of recall after `elapsed_days` at the given stability.

    Both `decay` and `factor` derive from `w[20]`, which is itself trained, so
    gradients flow through them too.
    """
    def retrievability(params, stability, elapsed_days) do
      elapsed = c(max(elapsed_days, 0) * 1.0)

      Nx.pow(
        Nx.add(1, Nx.divide(Nx.multiply(factor(params), elapsed), stability)),
        decay(params)
      )
    end

    @doc "Initial stability for a card's first review, `w[rating - 1]`."
    def initial_stability(params, rating) do
      Nx.max(params[rating - 1], c(@stability_min))
    end

    @doc "Initial difficulty for a card's first review, clamped to 1.0..10.0."
    def initial_difficulty(params, rating) do
      params
      |> initial_difficulty_unclamped(rating)
      |> Nx.clip(c(@min_difficulty), c(@max_difficulty))
    end

    defp initial_difficulty_unclamped(params, rating) do
      Nx.add(Nx.subtract(params[4], Nx.exp(Nx.multiply(params[5], rating - 1))), 1)
    end

    @doc "Difficulty after a review, with linear damping and mean reversion."
    def next_difficulty(params, difficulty, rating) do
      delta_difficulty = Nx.negate(Nx.multiply(params[6], rating - 3))

      damped =
        Nx.divide(Nx.multiply(Nx.subtract(c(10.0), difficulty), delta_difficulty), c(9.0))

      arg1 = initial_difficulty_unclamped(params, 4)
      arg2 = Nx.add(difficulty, damped)

      params[7]
      |> Nx.multiply(arg1)
      |> Nx.add(Nx.multiply(Nx.subtract(1, params[7]), arg2))
      |> Nx.clip(c(@min_difficulty), c(@max_difficulty))
    end

    @doc "Stability after a same-day (< 1 day elapsed) review."
    def short_term_stability(params, stability, rating) do
      increase =
        Nx.multiply(
          Nx.exp(Nx.multiply(params[17], Nx.add(rating - 3, params[18]))),
          Nx.pow(stability, Nx.negate(params[19]))
        )

      # A successful same-day review must never shrink stability.
      increase = if rating in [2, 3, 4], do: Nx.max(increase, c(1.0)), else: increase

      Nx.max(Nx.multiply(stability, increase), c(@stability_min))
    end

    @doc "Stability after a review at least a day later."
    def next_stability(params, difficulty, stability, retrievability, rating) do
      next =
        if rating == 1 do
          next_forget_stability(params, difficulty, stability, retrievability)
        else
          next_recall_stability(params, difficulty, stability, retrievability, rating)
        end

      Nx.max(next, c(@stability_min))
    end

    defp next_forget_stability(params, difficulty, stability, retrievability) do
      long_term =
        params[11]
        |> Nx.multiply(Nx.pow(difficulty, Nx.negate(params[12])))
        |> Nx.multiply(Nx.subtract(Nx.pow(Nx.add(stability, 1), params[13]), 1))
        |> Nx.multiply(Nx.exp(Nx.multiply(Nx.subtract(1, retrievability), params[14])))

      short_term = Nx.divide(stability, Nx.exp(Nx.multiply(params[17], params[18])))

      Nx.min(long_term, short_term)
    end

    defp next_recall_stability(params, difficulty, stability, retrievability, rating) do
      hard_penalty = if rating == 2, do: params[15], else: 1
      easy_bonus = if rating == 4, do: params[16], else: 1

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
    Advances a card's `{stability, difficulty}` by one review.

    Pass `nil` as the state for a card's first review. Mirrors the three-way
    branch in `ExFsrs.Scheduler.compute_stability_difficulty/4`.
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
