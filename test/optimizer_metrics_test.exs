defmodule ExFsrs.Optimizer.MetricsTest do
  @moduledoc """
  RMSE(bins) is the metric the FSRS project quotes, so these check it behaves
  the way that metric is supposed to: zero for a calibrated model, proportional
  to how far off a miscalibrated one is, and — the property a global average
  lacks — non-zero for a model that is right overall while wrong in every group.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Optimizer.Metrics

  # Bins are keyed on delta_t, review index and lapse count, so outcomes must be
  # independent of those for a calibrated model to score zero.
  defp sample(prediction, true_rate, count, seed) do
    :rand.seed(:exsss, seed)

    for _ <- 1..count do
      outcome = if :rand.uniform() < true_rate, do: 1.0, else: 0.0

      {prediction, outcome, :rand.uniform(60), :rand.uniform(12), :rand.uniform(4) - 1}
    end
  end

  describe "rmse_bins" do
    test "is zero when predictions are exactly right" do
      observations =
        for i <- 1..500 do
          outcome = rem(i, 2) * 1.0
          {outcome, outcome, rem(i, 30) + 1, rem(i, 10) + 1, 0}
        end

      assert Metrics.rmse_bins(observations) == 0.0
    end

    test "is near zero for a calibrated predictor" do
      assert Metrics.rmse_bins(sample(0.9, 0.9, 20_000, {1, 2, 3})) < 0.03
    end

    test "tracks how far off a miscalibrated predictor is" do
      over = Metrics.rmse_bins(sample(0.99, 0.9, 20_000, {1, 2, 3}))
      under = Metrics.rmse_bins(sample(0.70, 0.9, 20_000, {1, 2, 3}))

      # Off by 0.09 and 0.20 respectively.
      assert_in_delta over, 0.09, 0.02
      assert_in_delta under, 0.20, 0.02
      assert under > over
    end

    test "catches a model that is right on average but wrong in every bin" do
      # Half the reviews are short-interval and always recalled, half are
      # long-interval and never recalled. Predicting 0.5 everywhere matches the
      # global rate exactly, so a global average would score it perfect.
      observations =
        for i <- 1..20_000 do
          short? = rem(i, 2) == 0

          {0.5, if(short?, do: 1.0, else: 0.0),
           if(short?, do: rem(i, 3) + 1, else: 200 + rem(i, 50)), rem(i, 12) + 1, 0}
        end

      global_prediction = 0.5
      global_outcome = Enum.sum(Enum.map(observations, &elem(&1, 1))) / length(observations)
      assert_in_delta global_prediction, global_outcome, 0.01

      assert_in_delta Metrics.rmse_bins(observations), 0.5, 0.01
    end

    test "is zero for no observations" do
      assert Metrics.rmse_bins([]) == 0.0
    end
  end

  describe "log_loss" do
    test "matches the closed form" do
      observations = [{0.9, 1.0, 5, 2, 0}, {0.2, 0.0, 5, 2, 0}]

      expected = (-:math.log(0.9) + -:math.log(0.8)) / 2

      assert_in_delta Metrics.log_loss(observations), expected, 1.0e-12
    end

    test "is finite for a confidently wrong prediction" do
      assert Metrics.log_loss([{0.0, 1.0, 5, 2, 0}]) < 100.0
    end
  end

  describe "the two metrics can disagree" do
    test "a confident model can win on log loss and lose on calibration" do
      # Predicts 0.99 for everything; truth is 90%.
      confident = sample(0.99, 0.9, 5_000, {7, 7, 7})
      # Predicts 0.9 for everything; truth is 90%.
      calibrated = sample(0.9, 0.9, 5_000, {7, 7, 7})

      assert Metrics.rmse_bins(calibrated) < Metrics.rmse_bins(confident)
      assert Metrics.log_loss(calibrated) < Metrics.log_loss(confident)
    end
  end
end
