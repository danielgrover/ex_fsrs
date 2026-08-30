if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Loss do
    @moduledoc """
    Loss functions for parameter optimization.

    The objective is binary cross-entropy between the predicted retrievability
    of a card and whether it was actually recalled (`:again` is a lapse, every
    other rating a success).
    """

    # torch's BCELoss clamps its log outputs to >= -100 so that an infinitely
    # confident wrong prediction yields a finite loss and gradient.
    @log_min -100.0

    defp c(value), do: Nx.tensor(value, type: :f64)

    @doc """
    Binary cross-entropy between a predicted probability and a 0/1 label.

    Matches `torch.nn.BCELoss` for a single element, including its log clamping.
    """
    def binary_cross_entropy(prediction, label) do
      log_p = Nx.max(Nx.log(prediction), c(@log_min))
      log_1_p = Nx.max(Nx.log(Nx.subtract(1, prediction)), c(@log_min))

      Nx.negate(
        Nx.add(
          Nx.multiply(label, log_p),
          Nx.multiply(Nx.subtract(1, label), log_1_p)
        )
      )
    end

    # Expected spread of each parameter across collections, from fsrs-rs. A
    # parameter that naturally varies a lot is penalized less for moving, so the
    # penalty means the same thing for w[3] (initial stability for Easy, which
    # ranges over tens of days) as for w[7] (difficulty mean reversion, which
    # barely moves). fsrs-optimizer omits this and penalizes every parameter
    # equally.
    @params_stddev [
      6.43,
      9.66,
      17.58,
      27.85,
      0.57,
      0.28,
      0.6,
      0.12,
      0.39,
      0.18,
      0.33,
      0.3,
      0.09,
      0.16,
      0.57,
      0.25,
      1.03,
      0.31,
      0.32,
      0.14,
      0.27
    ]

    @doc "Per-parameter standard deviations used to scale the L2 penalty."
    def params_stddev, do: @params_stddev

    @doc """
    L2 penalty pulling parameters toward the values training started from.

    `gamma * batch_size / total_size * sum((w - initial)^2 / stddev^2)`, as
    `fsrs-rs` computes it. The `batch_size / total_size` factor spreads one
    collection-wide penalty across the minibatches, so the total pull over an
    epoch does not depend on how the data happens to be divided.

    Anchoring to the *starting* weights is what makes this worth having
    alongside `:initialize` — with the initial-stability weights fitted from
    data, the anchor is a measurement rather than a generic default.
    """
    def l2_penalty(params, initial, gamma, batch_size, total_size) do
      params
      |> Nx.subtract(initial)
      |> Nx.pow(2)
      |> Nx.divide(Nx.pow(stddev(), 2))
      |> Nx.sum()
      |> Nx.multiply(c(gamma * batch_size / total_size))
    end

    @doc """
    Gradient of `l2_penalty/5` with respect to the parameters.

    `2 * gamma * batch_size / total_size * (w - initial) / stddev^2`. Derived
    rather than differentiated: the penalty is a quadratic in the parameters, so
    its gradient is exact in closed form and there is no reason to trace it.
    """
    def l2_gradient(params, initial, gamma, batch_size, total_size) do
      params
      |> Nx.subtract(initial)
      |> Nx.divide(Nx.pow(stddev(), 2))
      |> Nx.multiply(c(2 * gamma * batch_size / total_size))
    end

    defp stddev, do: Nx.tensor(@params_stddev, type: :f64)
  end
end
