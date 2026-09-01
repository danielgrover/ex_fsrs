defmodule ExFsrs.ReviewLog do
  @moduledoc """
  A record of one review: the card as it was, the rating, and when.

  `ExFsrs.Scheduler.review_card/5` returns one of these with every review.
  Persisting them is what makes `ExFsrs.Scheduler.reschedule_card/3` and
  `ExFsrs.Optimizer` possible, since both work from a card's history rather
  than its current state.

  The `card` field is the card *before* the review was applied. Only its
  `card_id` is needed for rescheduling and optimizing; the rest is there so a
  review can be inspected or undone.

  `to_map/1` and `from_map/1` mirror `ExFsrs.to_map/1` and `ExFsrs.from_map/1`.
  """

  @typedoc "A review log."
  @type t :: %__MODULE__{
          card: ExFsrs.t(),
          rating: ExFsrs.rating(),
          review_datetime: DateTime.t(),
          review_duration: non_neg_integer() | nil
        }

  @ratings %{"again" => :again, "hard" => :hard, "good" => :good, "easy" => :easy}

  defstruct [:card, :rating, :review_datetime, :review_duration]

  @doc """
  Builds a review log.

  `review_duration` is how long the review took in milliseconds, if known. The
  scheduler never reads it.
  """
  @spec new(ExFsrs.t(), ExFsrs.rating(), DateTime.t(), non_neg_integer() | nil) :: t()
  def new(%ExFsrs{} = card, rating, review_datetime \\ DateTime.utc_now(), review_duration \\ nil)
      when rating in [:again, :hard, :good, :easy] do
    %__MODULE__{
      card: card,
      rating: rating,
      review_datetime: review_datetime,
      review_duration: review_duration
    }
  end

  @doc "Converts a review log to a string-keyed map for storage."
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = log) do
    %{
      "card" => ExFsrs.to_map(log.card),
      "rating" => Atom.to_string(log.rating),
      "review_datetime" => DateTime.to_iso8601(log.review_datetime),
      "review_duration" => log.review_duration
    }
  end

  @doc """
  Builds a review log from a map produced by `to_map/1`.

  Keys may be strings or atoms. Raises `ArgumentError` on a missing card, an
  unrecognised rating, or an unparseable datetime.
  """
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    %__MODULE__{
      card: parse_card(ExFsrs.field(map, :card)),
      rating: parse_rating(ExFsrs.field(map, :rating)),
      review_datetime:
        ExFsrs.parse_datetime!(ExFsrs.field(map, :review_datetime), "review_datetime"),
      review_duration: ExFsrs.field(map, :review_duration)
    }
  end

  defp parse_card(%ExFsrs{} = card), do: card
  defp parse_card(map) when is_map(map), do: ExFsrs.from_map(map)
  defp parse_card(value), do: raise(ArgumentError, "invalid card: #{inspect(value)}")

  defp parse_rating(rating) when rating in [:again, :hard, :good, :easy], do: rating

  defp parse_rating(rating) when is_binary(rating) and is_map_key(@ratings, rating),
    do: Map.fetch!(@ratings, rating)

  defp parse_rating(rating), do: raise(ArgumentError, "invalid rating: #{inspect(rating)}")
end
