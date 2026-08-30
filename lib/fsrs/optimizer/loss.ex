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

    @doc """
    L2 penalty pulling parameters toward their starting values.

    Unused by the py-fsrs-compatible optimizer, which has no regularization.
    Present so that `fsrs-optimizer`'s
    `sum(square(w - init_w)) * gamma * batch_size / train_set_size` can be added
    later without changing any call signature.
    """
    def l2_penalty(params, initial_params, gamma) do
      params
      |> Nx.subtract(initial_params)
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.multiply(gamma)
    end
  end
end
