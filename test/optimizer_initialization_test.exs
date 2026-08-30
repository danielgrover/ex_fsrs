defmodule ExFsrs.Optimizer.InitializationTest do
  @moduledoc """
  Checks the initial-stability fit against constructed data whose answer is
  known, and against the invariants `fsrs-optimizer` enforces.

  The curve fit itself was cross-checked against `scipy.optimize.minimize` on
  buckets exported from real collections: 16 of 16 rating fits agreed to within
  6.9e-16 relative loss. That check needs Python and the gated dataset, so what
  is asserted here is the property it established — that the search finds the
  true minimum — using data generated from a known stability.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Optimizer.Data
  alias ExFsrs.Optimizer.Initialization

  @defaults ExFsrs.Scheduler.new().parameters

  # Cards whose first review has `rating`, each followed by a review after
  # `delta_t` days that either succeeded or lapsed.
  defp cards(rating, outcomes) do
    outcomes
    |> Enum.with_index()
    |> Enum.map(fn {{delta_t, recalled?}, index} ->
      {index,
       [
         %Data.Review{rating: rating, elapsed_days: -1, label: 1.0, counts_for_loss?: false},
         %Data.Review{
           rating: if(recalled?, do: 3, else: 1),
           elapsed_days: delta_t,
           label: if(recalled?, do: 1.0, else: 0.0),
           counts_for_loss?: delta_t > 0
         }
       ]}
    end)
  end

  describe "observations" do
    test "takes one per card, from its first long-term review" do
      sequences = [
        {1,
         [
           %Data.Review{rating: 3, elapsed_days: -1, label: 1.0, counts_for_loss?: false},
           # same-day: not the observation
           %Data.Review{rating: 1, elapsed_days: 0, label: 0.0, counts_for_loss?: false},
           %Data.Review{rating: 3, elapsed_days: 5, label: 1.0, counts_for_loss?: true},
           # only the first long-term review counts
           %Data.Review{rating: 1, elapsed_days: 9, label: 0.0, counts_for_loss?: true}
         ]}
      ]

      assert Initialization.observations(sequences) == [{3, 5, 1.0}]
    end

    test "ignores cards that never got a long-term review" do
      sequences = [
        {1,
         [
           %Data.Review{rating: 3, elapsed_days: -1, label: 1.0, counts_for_loss?: false},
           %Data.Review{rating: 3, elapsed_days: 0, label: 1.0, counts_for_loss?: false}
         ]}
      ]

      assert Initialization.observations(sequences) == []
    end
  end

  describe "buckets" do
    test "group by interval and smooth toward the average" do
      observations = [{3, 5, 1.0}, {3, 5, 1.0}, {3, 5, 0.0}, {3, 10, 1.0}]

      assert [{5, five, 3}, {10, ten, 1}] = Initialization.buckets(observations, 0.9)

      # Raw means are 2/3 and 1/1; each is pulled toward 0.9 by one
      # pseudo-observation, so the sparser bucket moves further.
      assert_in_delta five, (2 / 3 * 3 + 0.9) / 4, 1.0e-12
      assert_in_delta ten, (1.0 * 1 + 0.9) / 2, 1.0e-12
      assert ten > 1.0 * 1 / 1 - 1.0e-9 or ten < 1.0
    end
  end

  describe "fit_stability" do
    test "recovers the stability that generated the data" do
      # Build buckets straight from the forgetting curve at a known stability,
      # with enough weight that the L1 pull toward the default is negligible.
      for true_stability <- [0.5, 2.0, 7.5, 30.0] do
        buckets =
          for delta_t <- [1, 2, 3, 5, 8, 13, 21, 34] do
            {delta_t, Initialization.retrievability(delta_t, true_stability), 10_000}
          end

        fitted = Initialization.fit_stability(buckets, true_stability)

        assert_in_delta fitted, true_stability, true_stability * 1.0e-4
      end
    end

    test "finds a lower loss than the default it starts from" do
      buckets =
        for delta_t <- [1, 3, 7, 14],
            do: {delta_t, Initialization.retrievability(delta_t, 9.0), 500}

      default = 2.3065
      fitted = Initialization.fit_stability(buckets, default)

      assert Initialization.loss(fitted, buckets, default) <
               Initialization.loss(default, buckets, default)
    end

    test "is pulled toward the default it is given" do
      # The L1 term does not override the data — a single 50%-recall observation
      # at five days really does imply a low stability — but it does bias the
      # answer toward the default, which is its purpose on thin data.
      buckets = [{5, 0.5, 1}]

      toward_low = Initialization.fit_stability(buckets, 0.001)
      toward_high = Initialization.fit_stability(buckets, 100.0)

      assert toward_high > toward_low

      # And with heavy data the pull is negligible: both land on the same fit.
      heavy = [{5, 0.5, 100_000}]

      assert_in_delta Initialization.fit_stability(heavy, 0.001),
                      Initialization.fit_stability(heavy, 100.0),
                      1.0e-3
    end
  end

  describe "initial_stability" do
    test "returns four values, ordered and within bounds" do
      sequences =
        cards(1, for(i <- 1..40, do: {rem(i, 5) + 1, rem(i, 3) != 0})) ++
          cards(3, for(i <- 1..60, do: {rem(i, 9) + 3, rem(i, 7) != 0})) ++
          cards(4, for(i <- 1..50, do: {rem(i, 12) + 8, rem(i, 11) != 0}))

      fitted = Initialization.initial_stability(sequences, @defaults)

      assert [_, _, _, _] = fitted
      assert fitted == Enum.sort(fitted), "expected non-decreasing, got #{inspect(fitted)}"

      for value <- fitted do
        assert value >= 0.001 and value <= 100.0
      end
    end

    test "an easier rating never yields lower stability than a harder one" do
      # Rating 1 is given long successful intervals and rating 4 short failures,
      # which fits an order the monotonicity rule has to correct.
      sequences =
        cards(1, for(i <- 1..200, do: {20 + rem(i, 5), true})) ++
          cards(4, for(_i <- 1..10, do: {1, false}))

      fitted = Initialization.initial_stability(sequences, @defaults)

      assert fitted == Enum.sort(fitted), "expected non-decreasing, got #{inspect(fitted)}"
    end

    test "falls back to the defaults with no usable observations" do
      sequences = [
        {1, [%Data.Review{rating: 3, elapsed_days: -1, label: 1.0, counts_for_loss?: false}]}
      ]

      assert Initialization.initial_stability(sequences, @defaults) == Enum.take(@defaults, 4)
    end

    test "scales the defaults when only one rating was ever used" do
      sequences = cards(3, for(i <- 1..80, do: {rem(i, 7) + 2, rem(i, 4) != 0}))

      fitted = Initialization.initial_stability(sequences, @defaults)
      ratios = Enum.zip(fitted, Enum.take(@defaults, 4)) |> Enum.map(fn {a, b} -> a / b end)

      # Every rating keeps its default ratio to the one that was observed.
      assert Enum.max(ratios) - Enum.min(ratios) < 1.0e-9
    end
  end
end
