defmodule ExFsrs.Optimizer.OutliersTest do
  @moduledoc """
  Checks outlier removal against the behaviour `fsrs-optimizer`'s
  `remove_outliers` specifies: a bounded amount dropped unconditionally,
  sparsest first, and beyond that only genuinely thin or implausibly long
  buckets.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Optimizer.Data
  alias ExFsrs.Optimizer.Data.Review

  defp review(rating, elapsed_days) do
    %Review{
      rating: rating,
      elapsed_days: elapsed_days,
      label: if(rating == 1, do: 0.0, else: 1.0),
      counts_for_loss?: elapsed_days > 0
    }
  end

  # `count` cards whose first review has `first_rating` and whose second review
  # comes after `delta_t` days.
  defp cards(first_rating, delta_t, count, id_base) do
    for i <- 1..count do
      {id_base + i, [review(first_rating, -1), review(3, delta_t)]}
    end
  end

  defp scored(sequences) do
    for {_id, reviews} <- sequences, r <- reviews, r.counts_for_loss?, do: r.elapsed_days
  end

  test "keeps a collection whose intervals are all well populated" do
    # Every bucket has 200 reviews at ordinary intervals; the budget is 5%, and
    # no whole bucket fits inside it.
    sequences =
      Enum.flat_map([1, 3, 7, 14], fn delta_t ->
        cards(3, delta_t, 200, delta_t * 1000)
      end)

    before = scored(sequences)
    after_removal = scored(Data.remove_outliers(sequences))

    assert Enum.count(after_removal) == Enum.count(before)
  end

  test "drops implausibly long intervals" do
    sequences =
      Enum.flat_map([1, 3, 7], fn delta_t -> cards(3, delta_t, 300, delta_t * 1000) end) ++
        cards(3, 400, 50, 900_000)

    kept = scored(Data.remove_outliers(sequences))

    refute 400 in kept, "a 400-day interval should not survive for a Good first rating"
    assert 7 in kept
  end

  test "allows longer intervals for cards first rated Easy" do
    # The same 200-day bucket, differing only in the card's first rating.
    good = Enum.flat_map([1, 3, 7], &cards(3, &1, 300, &1 * 1000)) ++ cards(3, 200, 50, 900_000)
    easy = Enum.flat_map([1, 3, 7], &cards(4, &1, 300, &1 * 1000)) ++ cards(4, 200, 50, 900_000)

    refute 200 in scored(Data.remove_outliers(good))
    assert 200 in scored(Data.remove_outliers(easy))
  end

  test "drops thin buckets" do
    sequences =
      Enum.flat_map([1, 3, 7], fn delta_t -> cards(3, delta_t, 400, delta_t * 1000) end) ++
        cards(3, 9, 2, 900_000)

    refute 9 in scored(Data.remove_outliers(sequences)),
           "a bucket with two reviews should not survive"
  end

  test "can remove everything when no interval is well populated" do
    # A sharp edge inherited from fsrs-optimizer: past the budget, any bucket
    # with fewer than 6 reviews is dropped, so a collection whose intervals are
    # all thinly populated loses all of them. Real collections repeat common
    # intervals hundreds of times, but a caller should not assume survival.
    sequences = Enum.flat_map(1..60, fn delta_t -> cards(3, delta_t, 5, delta_t * 1000) end)

    assert Data.num_reviews(sequences) > 0
    assert Data.num_reviews(Data.remove_outliers(sequences)) == 0
  end

  test "the budget bounds removal when buckets are well populated" do
    # Same 60 intervals, but with 50 reviews each. Only the budget's worth goes.
    sequences = Enum.flat_map(1..60, fn delta_t -> cards(3, delta_t, 50, delta_t * 1000) end)

    before = Data.num_reviews(sequences)
    after_removal = Data.num_reviews(Data.remove_outliers(sequences))

    assert after_removal > 0
    assert after_removal < before
    # Intervals over 100 days would also go, but none here exceed 60.
    assert after_removal > before * 0.9,
           "expected roughly the 5% budget, lost #{before - after_removal} of #{before}"
  end

  test "dropped reviews stay in the sequence and still advance state" do
    sequences =
      Enum.flat_map([1, 3, 7], fn delta_t -> cards(3, delta_t, 300, delta_t * 1000) end) ++
        cards(3, 500, 40, 900_000)

    cleaned = Data.remove_outliers(sequences)

    # Same number of reviews, fewer scored.
    count = fn seqs -> Enum.sum(Enum.map(seqs, fn {_id, rs} -> Enum.count(rs) end)) end
    assert count.(cleaned) == count.(sequences)
    assert Data.num_reviews(cleaned) < Data.num_reviews(sequences)

    # And the dropped ones are still present, just unscored.
    dropped = for {_id, rs} <- cleaned, r <- rs, r.elapsed_days == 500, do: r
    assert Enum.count(dropped) == 40
    assert Enum.all?(dropped, &(not &1.counts_for_loss?))
  end
end
