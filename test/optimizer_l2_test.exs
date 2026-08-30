defmodule ExFsrs.Optimizer.L2Test do
  @moduledoc """
  Checks the L2 penalty against fsrs-rs's own unit test for it
  (`l2_regularization` in `src/training.rs`), which ships expected values
  inline rather than depending on a dataset.

  fsrs-rs computes in f32, so its expected numbers carry f32 precision; ours are
  f64 and agree to that limit.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Optimizer.Loss

  # From fsrs-rs's test: the weights after one Adam step, and the defaults they
  # started from.
  @weights [
    0.252,
    1.3331,
    2.3464994,
    8.2556,
    6.3733,
    0.87340003,
    2.9794,
    0.040999997,
    1.8322,
    0.20660001,
    0.756,
    1.5235,
    0.021400042,
    0.3029,
    1.6882998,
    0.64140004,
    1.8329,
    0.5025,
    0.13119997,
    0.1058,
    0.1142
  ]

  @initial ExFsrs.Scheduler.new().parameters

  # fsrs-rs: l2_regularization(init_w, params_stddev, 512, 1000, 2.0)
  @expected_penalty 0.67711145
  @expected_gradient_head [0.0019813816, 0.00087788026, 0.00026506148, -0.000105618295]

  defp params, do: Nx.tensor(@weights, type: :f64)
  defp initial, do: Nx.tensor(@initial, type: :f64)

  test "penalty matches fsrs-rs" do
    penalty = Loss.l2_penalty(params(), initial(), 2.0, 512, 1000)

    # f32 vs f64: agreement to fsrs-rs's own precision is all that is available.
    assert_in_delta Nx.to_number(penalty), @expected_penalty, 1.0e-7
  end

  test "gradient matches fsrs-rs" do
    gradient = Loss.l2_gradient(params(), initial(), 2.0, 512, 1000)

    for {expected, index} <- Enum.with_index(@expected_gradient_head) do
      assert_in_delta Nx.to_number(gradient[index]), expected, 1.0e-9
    end
  end

  test "gradient agrees with a numerical derivative of the penalty" do
    # Guards the closed form against the penalty it is supposed to differentiate.
    epsilon = 1.0e-7
    analytical = Loss.l2_gradient(params(), initial(), 2.0, 512, 1000)

    for index <- 0..20 do
      bump =
        0..20
        |> Enum.map(&if(&1 == index, do: epsilon, else: 0.0))
        |> Nx.tensor(type: :f64)

      forward = Nx.to_number(Loss.l2_penalty(Nx.add(params(), bump), initial(), 2.0, 512, 1000))

      backward =
        Nx.to_number(Loss.l2_penalty(Nx.subtract(params(), bump), initial(), 2.0, 512, 1000))

      assert_in_delta Nx.to_number(analytical[index]),
                      (forward - backward) / (2 * epsilon),
                      1.0e-6
    end
  end

  test "is zero at the anchor and grows with distance from it" do
    assert Nx.to_number(Loss.l2_penalty(initial(), initial(), 1.0, 512, 1000)) == 0.0

    near = Loss.l2_penalty(params(), initial(), 1.0, 512, 1000)

    far =
      Loss.l2_penalty(
        Nx.add(initial(), Nx.multiply(Nx.subtract(params(), initial()), 2)),
        initial(),
        1.0,
        512,
        1000
      )

    assert Nx.to_number(far) > Nx.to_number(near)
  end

  test "scales linearly with gamma and with the batch fraction" do
    base = Nx.to_number(Loss.l2_penalty(params(), initial(), 1.0, 512, 1000))

    assert_in_delta Nx.to_number(Loss.l2_penalty(params(), initial(), 2.0, 512, 1000)),
                    base * 2,
                    1.0e-12

    assert_in_delta Nx.to_number(Loss.l2_penalty(params(), initial(), 1.0, 256, 1000)),
                    base / 2,
                    1.0e-12
  end
end
