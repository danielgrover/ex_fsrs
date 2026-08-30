defmodule ExFsrs.Fixtures do
  @moduledoc """
  Loads the py-fsrs reference fixtures.

  `revlogs_josh.csv` is the review history py-fsrs ships as
  `tests/review_logs_josh_1711744352250_to_1728234780857.csv`.
  `pyfsrs_trace.json` is a recording of py-fsrs's optimizer running over it —
  see `test/fixtures/generate_trace.py` for how it was produced.
  """

  @fixtures Path.join(__DIR__, "../fixtures")

  @doc "The reference review logs as `{card_id, rating, review_datetime}` tuples."
  def review_log_tuples do
    @fixtures
    |> Path.join("revlogs_josh.csv")
    |> File.stream!()
    |> Stream.drop(1)
    |> Enum.map(fn line ->
      [card_id, rating, review_time, _duration] =
        line |> String.trim() |> String.split(",")

      {:ok, datetime, _offset} = DateTime.from_iso8601(review_time)

      {String.to_integer(card_id), String.to_integer(rating), datetime}
    end)
  end

  @doc "The same logs as `%ExFsrs.ReviewLog{}` structs."
  def review_logs do
    Enum.map(review_log_tuples(), fn {card_id, rating, datetime} ->
      ExFsrs.ReviewLog.new(ExFsrs.new(card_id: card_id), rating_atom(rating), datetime)
    end)
  end

  defp rating_atom(1), do: :again
  defp rating_atom(2), do: :hard
  defp rating_atom(3), do: :good
  defp rating_atom(4), do: :easy

  @doc "The recorded py-fsrs optimization trace."
  def trace do
    @fixtures
    |> Path.join("pyfsrs_trace.json")
    |> File.read!()
    |> JSON.decode!()
  end
end
