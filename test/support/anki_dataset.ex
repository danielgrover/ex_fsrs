defmodule ExFsrs.AnkiDataset do
  @moduledoc """
  Loads per-user review histories fetched from the Anki Revlogs 10K dataset.

  Fetch them first with `bench/fetch_anki_revlogs.py`, which needs a Hugging
  Face token because the dataset is gated. The CSVs are not committed — the
  dataset carries its own licence and is not ours to redistribute — so anything
  depending on them must skip when `available?/0` is false.

  The dataset records `elapsed_days` per review rather than absolute timestamps,
  which is what `ExFsrs.Optimizer.Data.build_sequences_from_elapsed/1` takes.
  """

  @fixtures Path.join(__DIR__, "../fixtures/anki")

  @doc "Whether any user data has been fetched."
  def available?, do: users() != []

  @doc "User ids that have been fetched locally, ascending."
  def users do
    case File.ls(@fixtures) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.match?(&1, ~r/^user_\d+\.csv$/))
        |> Enum.map(
          &(&1
            |> String.replace_prefix("user_", "")
            |> String.replace_suffix(".csv", ""))
        )
        |> Enum.map(&String.to_integer/1)
        |> Enum.sort()

      {:error, _reason} ->
        []
    end
  end

  @doc """
  A user's reviews as `{card_id, rating, elapsed_days}`, in dataset order.

  The dataset is already sorted chronologically, and `-1` marks a card's first
  review, so the tuples can go straight into
  `ExFsrs.Optimizer.Data.build_sequences_from_elapsed/1`.
  """
  def reviews(user_id) do
    @fixtures
    |> Path.join("user_#{user_id}.csv")
    |> File.stream!()
    |> Stream.drop(1)
    |> Enum.map(fn line ->
      [card_id, rating, elapsed_days | _rest] = line |> String.trim() |> String.split(",")

      {String.to_integer(card_id), String.to_integer(rating), String.to_integer(elapsed_days)}
    end)
  end

  @doc """
  A user's reviews as `{card_id, rating, elapsed_days, day_offset}`.

  `day_offset` counts days since that user's first review and is the only
  absolute timeline the dataset carries, so it is what a temporal split needs.
  """
  def reviews_with_time(user_id) do
    path = Path.join(@fixtures, "user_#{user_id}.csv")

    # Older exports had no day_offset column. Reading position 4 from one of
    # those silently yields `state`, whose 0..4 values look like a plausible
    # timeline, so check the header rather than trust the position.
    header = path |> File.stream!() |> Enum.take(1) |> List.first() |> String.trim()

    unless String.contains?(header, "day_offset") do
      raise """
      #{Path.basename(path)} predates the day_offset column and cannot be split       by time. Re-fetch it:

          python bench/fetch_anki_revlogs.py --user-ids #{user_id}
      """
    end

    path
    |> File.stream!()
    |> Stream.drop(1)
    |> Enum.map(fn line ->
      [card_id, rating, elapsed_days, day_offset | _rest] =
        line |> String.trim() |> String.split(",")

      {String.to_integer(card_id), String.to_integer(rating), String.to_integer(elapsed_days),
       String.to_integer(day_offset)}
    end)
  end

  @doc """
  Splits a collection in time: reviews before `day` train, the rest is scored.

  Predicting the future from the past is what a scheduler actually does, so this
  is the split that matters — `srs-benchmark` uses the same idea. It is not a
  partition of the data, because a card usually has reviews on both sides of the
  cut:

  * the training set is each card truncated to its pre-cut reviews;
  * the test set is each card's **whole** history, with only its post-cut
    reviews scored. The earlier reviews still have to be replayed, because a
    card's stability at the cut depends on everything that came before it.

  Returns `{train_sequences, test_sequences}`.
  """
  def temporal_split(user_id, quantile \\ 0.8) do
    reviews = reviews_with_time(user_id)
    cut = cut_day(reviews, quantile)

    train =
      reviews
      |> Enum.filter(fn {_id, _rating, _elapsed, day} -> day < cut end)
      |> Enum.map(fn {id, rating, elapsed, _day} -> {id, rating, elapsed} end)
      |> ExFsrs.Optimizer.Data.build_sequences_from_elapsed()

    {train, test_sequences(reviews, cut)}
  end

  # The day by which `quantile` of the scoreable reviews have happened.
  defp cut_day(reviews, quantile) do
    days =
      reviews
      |> Enum.filter(fn {_id, _rating, elapsed, _day} -> elapsed > 0 end)
      |> Enum.map(fn {_id, _rating, _elapsed, day} -> day end)
      |> Enum.sort()

    case days do
      [] -> 0
      _ -> Enum.at(days, min(round(length(days) * quantile), length(days) - 1))
    end
  end

  # Every card's whole history, with scoring switched off before the cut. The
  # earlier reviews stay so that state still accumulates through them; they just
  # do not count toward the loss.
  defp test_sequences(reviews, cut) do
    limit = ExFsrs.Optimizer.Data.max_seq_len()

    by_card =
      Enum.group_by(
        reviews,
        fn {id, _rating, _elapsed, _day} -> id end,
        fn {_id, rating, elapsed, day} -> {rating, elapsed, day} end
      )

    sequences =
      reviews
      |> Enum.map(fn {id, rating, elapsed, _day} -> {id, rating, elapsed} end)
      |> ExFsrs.Optimizer.Data.build_sequences_from_elapsed()

    Enum.map(sequences, fn {card_id, card_reviews} ->
      # Both sides take the card's reviews in order and truncate identically,
      # so position i lines up with position i.
      days =
        by_card
        |> Map.fetch!(card_id)
        |> Enum.take(limit)
        |> Enum.map(fn {_rating, _elapsed, day} -> day end)

      scored =
        card_reviews
        |> Enum.zip(days)
        |> Enum.map(fn {review, day} ->
          %{review | counts_for_loss?: review.counts_for_loss? and day >= cut}
        end)

      {card_id, scored}
    end)
  end

  @doc "A user's reviews as optimizer sequences."
  def sequences(user_id) do
    user_id |> reviews() |> ExFsrs.Optimizer.Data.build_sequences_from_elapsed()
  end

  @doc """
  A summary of every fetched user: cards, total reviews, and scored reviews.

  Useful for picking a collection small enough to iterate on — an optimization
  run costs roughly in proportion to the scored count.
  """
  def summary do
    Enum.map(users(), fn user_id ->
      sequences = sequences(user_id)

      %{
        user_id: user_id,
        cards: Enum.count(sequences),
        reviews: Enum.sum(Enum.map(sequences, fn {_id, reviews} -> Enum.count(reviews) end)),
        scored: ExFsrs.Optimizer.Data.num_reviews(sequences)
      }
    end)
  end
end
