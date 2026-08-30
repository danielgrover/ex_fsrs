defmodule ExFsrs.Optimizer.Initialization do
  @moduledoc """
  Fits the initial-stability parameters `w[0..3]` from review history.

  `w[0..3]` are the stability a card gets after its very first review, one per
  rating. py-fsrs leaves them at the defaults and lets gradient descent move
  them; `fsrs-optimizer` and `fsrs-rs` instead measure them, which is the single
  largest accuracy difference between those implementations and ours.

  The measurement is a forgetting curve fit. Group every card by the rating of
  its first review, look at how many were still recalled after each interval,
  and find the stability whose curve best explains those outcomes.

  Needs no `Nx` — the fit is one-dimensional and bounded, so a direct search
  beats autodiff here.

  ## What counts as an observation

  A card contributes one observation: its first rating, and the outcome of its
  first *long-term* review — the first review at least a day later. Same-day
  reviews are excluded because the forgetting curve is not what governs them.

  `fsrs-optimizer` derives this from its own preprocessing pipeline; this is
  `fsrs-rs`'s definition, which says the same thing without depending on that
  pipeline.
  """

  # The forgetting curve used for the fit is the FSRS-6 default, not the trained
  # w[20]: this runs before there is anything trained.
  @decay -0.1542
  @stability_min 0.001
  @stability_max 100.0

  # Pulls the fit toward the default so a rating with thin data cannot wander.
  @l1_scale 16.0

  # Interpolation weights for filling in a rating the user never pressed.
  @w1 0.41
  @w2 0.54

  @doc """
  Fitted `w[0..3]` for the given sequences, or the defaults where data is thin.

  Returns the four initial-stability parameters, ready to be spliced over the
  first four entries of a parameter list.
  """
  def initial_stability(sequences, defaults) do
    observations = observations(sequences)
    average_recall = average_recall(observations)

    fitted =
      observations
      |> Enum.group_by(fn {rating, _delta_t, _recalled} -> rating end)
      |> Map.new(fn {rating, rating_observations} ->
        {rating, fit_rating(rating_observations, average_recall, Enum.at(defaults, rating - 1))}
      end)

    fitted
    |> enforce_monotonicity()
    |> fill_missing(defaults)
    |> Enum.map(&clamp/1)
  end

  @doc """
  One `{first_rating, delta_t, recalled?}` per card that has a long-term review.

  Exposed so the bucketing can be checked independently of the curve fit.
  """
  def observations(sequences) do
    Enum.flat_map(sequences, fn {_card_id, reviews} -> observation(reviews) end)
  end

  defp observation([first | rest]) do
    case Enum.find(rest, & &1.counts_for_loss?) do
      nil -> []
      review -> [{first.rating, review.elapsed_days, review.label}]
    end
  end

  defp observation([]), do: []

  # The overall pass rate, used to smooth thinly-populated intervals.
  defp average_recall([]), do: 0.9

  defp average_recall(observations) do
    Enum.sum(Enum.map(observations, fn {_rating, _delta_t, label} -> label end)) /
      Enum.count(observations)
  end

  @doc """
  Interval buckets for one rating: `{delta_t, smoothed_recall, count}`.

  Each interval's raw pass rate is smoothed toward the collection's average by
  one pseudo-observation — Laplace smoothing — so an interval seen twice does
  not carry the same weight as one seen two hundred times.
  """
  def buckets(rating_observations, average_recall) do
    rating_observations
    |> Enum.group_by(
      fn {_rating, delta_t, _label} -> delta_t end,
      fn {_rating, _delta_t, label} -> label end
    )
    |> Enum.sort_by(fn {delta_t, _labels} -> delta_t end)
    |> Enum.map(fn {delta_t, labels} ->
      count = Enum.count(labels)
      mean = Enum.sum(labels) / count

      {delta_t, (mean * count + average_recall) / (count + 1), count}
    end)
  end

  defp fit_rating(rating_observations, average_recall, default) do
    buckets = buckets(rating_observations, average_recall)
    count = Enum.sum(Enum.map(buckets, fn {_t, _recall, n} -> n end))

    {fit_stability(buckets, default), count}
  end

  @doc """
  The stability whose forgetting curve best fits these buckets.

  The raw fit for one rating, before monotonicity across ratings is applied.
  Exposed so the fit can be checked against a reference optimizer on identical
  buckets, independently of the ordering rules layered on top.
  """
  def fit_stability(buckets, default) do
    minimize(&loss(&1, buckets, default))
  end

  @doc """
  Weighted log-loss of a candidate stability against the observed buckets.

  The `|s - default| / 16` term is `fsrs-optimizer`'s: it keeps a rating with
  few observations near its default rather than letting one sparse interval
  dictate the answer.
  """
  def loss(stability, buckets, default) do
    log_loss =
      Enum.reduce(buckets, 0.0, fn {delta_t, recall, count}, total ->
        predicted = retrievability(delta_t, stability)

        total +
          -(recall * :math.log(predicted) + (1 - recall) * :math.log(1 - predicted)) * count
      end)

    log_loss + abs(stability - default) / @l1_scale
  end

  @doc "Probability of recall after `delta_t` days at the given stability."
  def retrievability(delta_t, stability) do
    factor = :math.pow(0.9, 1 / @decay) - 1

    :math.pow(1 + factor * delta_t / stability, @decay)
  end

  # A bounded one-dimensional minimum. The loss is smooth but not guaranteed
  # unimodal over four orders of magnitude, so a log-spaced scan brackets the
  # minimum before golden-section refines it. `fsrs-optimizer` uses L-BFGS-B
  # from the default; this does not reproduce its path, only its answer.
  defp minimize(loss) do
    scan =
      for i <- 0..96 do
        stability =
          @stability_min *
            :math.pow(@stability_max / @stability_min, i / 96)

        {loss.(stability), stability}
      end

    {_value, best} = Enum.min_by(scan, fn {value, _stability} -> value end)

    index = Enum.find_index(scan, fn {_value, stability} -> stability == best end)
    {_v, low} = Enum.at(scan, max(index - 1, 0))
    {_v, high} = Enum.at(scan, min(index + 1, 96))

    golden_section(loss, low, high, 80)
  end

  @invphi 0.6180339887498949

  defp golden_section(_loss, low, high, 0), do: (low + high) / 2

  defp golden_section(loss, low, high, iterations) do
    span = high - low
    c = high - @invphi * span
    d = low + @invphi * span

    if loss.(c) < loss.(d) do
      golden_section(loss, low, d, iterations - 1)
    else
      golden_section(loss, c, high, iterations - 1)
    end
  end

  # Higher ratings must not produce lower initial stability. Where they do, the
  # rating backed by fewer observations gives way to the better-evidenced one.
  defp enforce_monotonicity(fitted) do
    pairs = [{1, 2}, {2, 3}, {3, 4}, {1, 3}, {2, 4}, {1, 4}]

    Enum.reduce(pairs, fitted, fn {small, big}, acc ->
      with {small_stability, small_count} <- Map.get(acc, small),
           {big_stability, big_count} <- Map.get(acc, big),
           true <- small_stability > big_stability do
        if small_count > big_count do
          Map.put(acc, big, {small_stability, big_count})
        else
          Map.put(acc, small, {big_stability, small_count})
        end
      else
        _ -> acc
      end
    end)
  end

  defp fill_missing(fitted, defaults) do
    stabilities = Map.new(fitted, fn {rating, {stability, _count}} -> {rating, stability} end)

    case map_size(stabilities) do
      0 -> Enum.take(defaults, 4)
      4 -> Enum.map(1..4, &stabilities[&1])
      1 -> scale_defaults(stabilities, defaults)
      _ -> interpolate(stabilities) |> then(&Enum.map(1..4, fn r -> &1[r] end))
    end
  end

  # With a single rating observed, the others keep their default ratios.
  defp scale_defaults(stabilities, defaults) do
    [{rating, stability}] = Map.to_list(stabilities)
    factor = stability / Enum.at(defaults, rating - 1)

    defaults |> Enum.take(4) |> Enum.map(&(&1 * factor))
  end

  # Ratings the user never pressed are interpolated from the ones they did,
  # using fsrs-optimizer's weights between adjacent ratings.
  defp interpolate(stabilities) do
    stabilities
    |> fill(2, [{1, 3}], fn s -> :math.pow(s[1], @w1) * :math.pow(s[3], 1 - @w1) end)
    |> fill(3, [{2, 4}], fn s -> :math.pow(s[2], 1 - @w2) * :math.pow(s[4], @w2) end)
    |> fill(1, [{2, 3}], fn s ->
      :math.pow(s[2], 1 / @w1) * :math.pow(s[3], 1 - 1 / @w1)
    end)
    |> fill(4, [{2, 3}], fn s ->
      :math.pow(s[2], 1 - 1 / @w2) * :math.pow(s[3], 1 / @w2)
    end)
    |> fill_remaining()
  end

  defp fill(stabilities, rating, required, formula) do
    needed = Enum.flat_map(required, fn {a, b} -> [a, b] end)

    if Map.has_key?(stabilities, rating) or
         not Enum.all?(needed, &Map.has_key?(stabilities, &1)) do
      stabilities
    else
      Map.put(stabilities, rating, formula.(stabilities))
    end
  end

  # Anything still missing after interpolation takes the nearest known rating,
  # which keeps the result ordered without inventing a value.
  defp fill_remaining(stabilities) do
    Enum.reduce(1..4, stabilities, fn rating, acc ->
      if Map.has_key?(acc, rating) do
        acc
      else
        nearest =
          acc
          |> Map.keys()
          |> Enum.min_by(&abs(&1 - rating))

        Map.put(acc, rating, acc[nearest])
      end
    end)
  end

  defp clamp(stability), do: stability |> min(@stability_max) |> max(@stability_min)
end
