if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Adam do
    @moduledoc """
    Adam with a cosine-annealing learning rate.

    Mirrors `torch.optim.Adam` at its defaults (no weight decay, no amsgrad)
    together with `torch.optim.lr_scheduler.CosineAnnealingLR` at `eta_min = 0`,
    so that an optimization run can be compared step by step against py-fsrs.

    The learning rate uses the schedule's closed form,
    `base_lr * (1 + cos(pi * t / t_max)) / 2`, which agrees with torch's
    recursive implementation to within 7.0e-18 over a full 55-step run.
    """

    @beta1 0.9
    @beta2 0.999
    @eps 1.0e-8

    defstruct [:m, :v, :t, :base_learning_rate, :t_max]

    # A bare float literal becomes an f32 tensor, which would round the beta
    # coefficients and shift each update by ~1e-8 relative — enough to move the
    # final parameters in the 4th decimal place after 55 steps.
    defp c(value), do: Nx.tensor(value, type: :f64)

    @doc """
    Creates an optimizer for a parameter vector of `size` elements.

    * `:learning_rate` — base rate before annealing (default `4.0e-2`)
    * `:t_max` — steps over which the rate anneals to zero
    """
    def new(size, opts \\ []) do
      %__MODULE__{
        m: Nx.broadcast(Nx.tensor(0.0, type: :f64), {size}),
        v: Nx.broadcast(Nx.tensor(0.0, type: :f64), {size}),
        t: 0,
        base_learning_rate: Keyword.get(opts, :learning_rate, 4.0e-2),
        t_max: Keyword.fetch!(opts, :t_max)
      }
    end

    @doc "The learning rate that the next `step/3` will use."
    def learning_rate(%__MODULE__{} = adam) do
      adam.base_learning_rate * (1 + :math.cos(:math.pi() * adam.t / adam.t_max)) / 2
    end

    @doc """
    Applies one update, returning `{optimizer, new_params}`.

    Bias correction follows torch: the moment estimates are corrected by
    `1 - beta^t` with `t` counting from 1 at the first update.
    """
    def step(%__MODULE__{} = adam, params, gradient) do
      lr = learning_rate(adam)
      t = adam.t + 1

      m =
        Nx.add(
          Nx.multiply(adam.m, c(@beta1)),
          Nx.multiply(gradient, c(1 - @beta1))
        )

      v =
        Nx.add(
          Nx.multiply(adam.v, c(@beta2)),
          Nx.multiply(Nx.pow(gradient, 2), c(1 - @beta2))
        )

      bias_correction1 = 1 - :math.pow(@beta1, t)
      bias_correction2 = 1 - :math.pow(@beta2, t)

      denominator =
        Nx.add(
          Nx.divide(Nx.sqrt(v), Nx.tensor(:math.sqrt(bias_correction2), type: :f64)),
          Nx.tensor(@eps, type: :f64)
        )

      step_size = Nx.tensor(lr / bias_correction1, type: :f64)

      params = Nx.subtract(params, Nx.multiply(step_size, Nx.divide(m, denominator)))

      {%{adam | m: m, v: v, t: t}, params}
    end
  end
end
