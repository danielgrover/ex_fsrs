if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Metrics do
    @moduledoc """
    The metrics `srs-benchmark` reports, so results here can be compared with
    the numbers the FSRS project publishes.

    Log loss is what the optimizer minimizes, but it is hard to interpret: a
    change of 0.002 says nothing about whether anyone's schedule improved.
    RMSE(bins) is the metric FSRS quotes instead, and it has units you can
    reason about — **the average gap between the recall probability the model
    predicted and the recall rate actually observed.** An RMSE of 0.05 means
    predictions are off by about five percentage points.

    ## How the bins work

    Comparing a single prediction against a single 0/1 outcome is meaningless,
    so reviews are grouped and compared in aggregate. `srs-benchmark` groups on
    three things, each on a log scale so that small values get fine buckets and
    large ones get coarse:

      * `delta_t` — days since the card's last review
      * `i` — how many times the card has been reviewed
      * `lapses` — how often it has been forgotten

    Within each bucket it takes the mean prediction and the mean outcome, then
    reports the weighted RMSE between them. Grouping this way stops a model
    being rewarded for being right on average while being wrong everywhere in
    particular — over-predicting short intervals and under-predicting long ones
    would cancel out in a global average but not here.

    The bin edges are `srs-benchmark`'s, reproduced exactly rather than chosen.
    """

    @doc """
    RMSE(bins) between predictions and outcomes.

    Takes `{prediction, outcome, delta_t, review_index, lapses}` tuples, where
    `outcome` is 1.0 for a recall and 0.0 for a lapse.
    """
    def rmse_bins(observations) do
      grouped =
        Enum.group_by(observations, fn {_p, _y, delta_t, index, lapses} ->
          {delta_t_bin(delta_t), index_bin(index), lapse_bin(lapses)}
        end)

      {weighted_error, total} =
        Enum.reduce(grouped, {0.0, 0}, fn {_bin, members}, {error, total} ->
          count = Enum.count(members)
          mean_prediction = Enum.sum(Enum.map(members, fn {p, _y, _t, _i, _l} -> p end)) / count
          mean_outcome = Enum.sum(Enum.map(members, fn {_p, y, _t, _i, _l} -> y end)) / count

          {error + :math.pow(mean_prediction - mean_outcome, 2) * count, total + count}
        end)

      if total == 0, do: 0.0, else: :math.sqrt(weighted_error / total)
    end

    @doc """
    Mean binary cross-entropy, the quantity the optimizer minimizes.

    Reported alongside RMSE(bins) because `srs-benchmark` reports both, and
    because they can disagree: a model can predict the right average while
    misjudging which cards are at risk.
    """
    def log_loss(observations) do
      count = Enum.count(observations)

      total =
        Enum.reduce(observations, 0.0, fn {prediction, outcome, _t, _i, _l}, total ->
          prediction = min(max(prediction, 1.0e-12), 1.0 - 1.0e-12)

          total - (outcome * :math.log(prediction) + (1 - outcome) * :math.log(1 - prediction))
        end)

      if count == 0, do: 0.0, else: total / count
    end

    # srs-benchmark's bin edges, kept verbatim so numbers are comparable.
    defp delta_t_bin(delta_t) do
      value = max(delta_t, 1.0e-6)

      Float.round(2.48 * :math.pow(3.62, Float.floor(:math.log(value) / :math.log(3.62))), 2)
    end

    defp index_bin(index) do
      value = max(index, 1)

      Float.round(1.99 * :math.pow(1.89, Float.floor(:math.log(value) / :math.log(1.89))), 0)
    end

    defp lapse_bin(0), do: 0.0

    defp lapse_bin(lapses) do
      Float.round(1.65 * :math.pow(1.73, Float.floor(:math.log(lapses) / :math.log(1.73))), 0)
    end
  end
end
