defmodule ExFsrs.Optimizer.DataTest do
  @moduledoc """
  The two input paths into the optimizer — timestamped logs and pre-computed
  elapsed days — and the recency weighting layered on top of them.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Optimizer.Data

  @start ~U[2024-01-01 09:00:00Z]

  defp days_later(days), do: DateTime.add(@start, days, :day)

  describe "build_sequences_from_elapsed/1" do
    test "keeps per-card order and derives labels and scoring from elapsed days" do
      reviews = [
        {2, :good, -1},
        {1, 3, -1},
        {1, 1, 0},
        {1, :again, 4},
        {2, :easy, 7}
      ]

      assert [{1, [first, same_day, lapse]}, {2, [_first, easy]}] =
               Data.build_sequences_from_elapsed(reviews)

      assert %Data.Review{rating: 3, elapsed_days: -1, counts_for_loss?: false} = first
      assert %Data.Review{rating: 1, elapsed_days: 0, counts_for_loss?: false} = same_day
      assert %Data.Review{rating: 1, elapsed_days: 4, label: +0.0, counts_for_loss?: true} = lapse
      assert %Data.Review{rating: 4, elapsed_days: 7, label: 1.0, counts_for_loss?: true} = easy
    end

    test "truncates a card at the maximum sequence length" do
      reviews = for i <- 0..99, do: {1, :good, if(i == 0, do: -1, else: 1)}

      assert [{1, kept}] = Data.build_sequences_from_elapsed(reviews)
      assert length(kept) == Data.max_seq_len()
    end
  end

  describe "build_sequences/2 with recency: true" do
    test "ranks reviews chronologically, not by DateTime field order" do
      # These dates are chosen so that comparing DateTime structs field by
      # field (day, then month, then year) would order them differently from
      # their actual order in time.
      logs = [
        {1, :good, ~U[2023-12-15 00:00:00Z]},
        {1, :good, ~U[2024-01-31 00:00:00Z]},
        {1, :good, ~U[2024-02-01 00:00:00Z]},
        {1, :good, ~U[2024-03-05 00:00:00Z]}
      ]

      assert [{1, [_first | scored]}] = Data.build_sequences(logs, recency: true)

      weights = Enum.map(scored, & &1.weight)
      assert weights == Enum.sort(weights), "expected ascending, got #{inspect(weights)}"
      assert [0.25, _, 1.0] = weights
    end

    test "ranks across cards and leaves unscored reviews at weight 1.0" do
      logs = [
        {1, :good, days_later(0)},
        {1, :good, days_later(1)},
        {2, :good, days_later(2)},
        {2, :good, days_later(3)},
        {1, :good, days_later(4)}
      ]

      assert [{1, [c1_first, c1_second, c1_third]}, {2, [c2_first, c2_second]}] =
               Data.build_sequences(logs, recency: true)

      # First reviews are never scored and keep the neutral weight.
      assert c1_first.weight == 1.0
      assert c2_first.weight == 1.0

      # The three scored reviews rank day 1 < day 3 < day 4 across both cards.
      assert c1_second.weight == 0.25
      assert c1_third.weight == 1.0
      assert c2_second.weight > 0.25 and c2_second.weight < 1.0
    end

    test "is off by default" do
      logs = [{1, :good, days_later(0)}, {1, :good, days_later(1)}]

      assert [{1, reviews}] = Data.build_sequences(logs)
      assert Enum.all?(reviews, &(&1.weight == 1.0))
    end
  end

  describe "apply_recency_weights/2" do
    test "follows the fsrs-rs ramp with numeric day offsets" do
      sequences =
        Data.build_sequences_from_elapsed(
          for i <- 0..5, do: {1, :good, if(i == 0, do: -1, else: 1)}
        )

      [{1, reviews}] = Data.apply_recency_weights(sequences, %{1 => Enum.to_list(0..5)})

      # Five scored reviews ranked 0..4: 0.25 + 0.75 * (rank / 4)^3.
      expected = for rank <- 0..4, do: 0.25 + 0.75 * :math.pow(rank / 4, 3)

      for {review, weight} <- Enum.zip(tl(reviews), expected) do
        assert_in_delta review.weight, weight, 1.0e-12
      end
    end

    test "a lone scored review gets full weight" do
      sequences = Data.build_sequences_from_elapsed([{1, :good, -1}, {1, :good, 2}])

      assert [{1, [_first, %{weight: 1.0}]}] =
               Data.apply_recency_weights(sequences, %{1 => [0, 2]})
    end
  end
end
