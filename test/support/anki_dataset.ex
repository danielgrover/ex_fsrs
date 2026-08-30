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
