defmodule ExFsrs.ReviewLog do
  @moduledoc """
  Review log functionality for FSRS.

  This module handles the storage and retrieval of review history
  for cards in the spaced repetition system.
  """

  alias ExFsrs

  @type t :: %__MODULE__{
          card: ExFsrs.t(),
          rating: ExFsrs.rating(),
          review_datetime: DateTime.t(),
          review_duration: integer() | nil
        }

  defstruct [
    :card,
    :rating,
    :review_datetime,
    :review_duration
  ]

  @doc """
  Creates a new review log entry.

  ## Parameters
    - card: ExFsrs struct that was reviewed
    - rating: Rating given to the card
    - review_datetime: DateTime of the review
    - review_duration: Duration of the review in milliseconds

  ## Returns
    - A new ReviewLog struct
  """
  def new(card, rating, review_datetime \\ DateTime.utc_now(), review_duration \\ nil) do
    %__MODULE__{
      card: card,
      rating: rating,
      review_datetime: review_datetime,
      review_duration: review_duration
    }
  end

  @doc """
  Converts a review log to a map for storage.

  ## Parameters
    - log: ReviewLog struct to convert

  ## Returns
    - Map containing the review log data
  """
  def to_map(%__MODULE__{} = log) do
    %{
      "card" => ExFsrs.to_map(log.card),
      "rating" => Atom.to_string(log.rating),
      "review_datetime" => DateTime.to_iso8601(log.review_datetime),
      "review_duration" => log.review_duration
    }
  end

  @doc """
  Creates a review log from a map.

  ## Parameters
    - map: Map containing review log data

  ## Returns
    - ReviewLog struct
  """
  def from_map(map) do
    %__MODULE__{
      card: parse_card(map["card"] || map[:card]),
      rating: parse_rating(map["rating"] || map[:rating]),
      review_datetime: parse_review_datetime(map["review_datetime"] || map[:review_datetime]),
      review_duration: map["review_duration"] || map[:review_duration]
    }
  end

  defp parse_review_datetime(nil), do: DateTime.utc_now()

  defp parse_review_datetime(value) when is_binary(value) do
    case DateTime.from_iso8601(value) do
      {:ok, datetime, 0} -> datetime
      _ -> raise "Invalid ISO8601 datetime format for review_datetime"
    end
  end

  defp parse_review_datetime(value), do: value

  defp parse_rating(value) when is_atom(value), do: value
  defp parse_rating(value) when is_binary(value), do: String.to_existing_atom(value)
  defp parse_rating(_), do: :good

  defp parse_card(nil), do: ExFsrs.new()
  defp parse_card(data), do: ExFsrs.from_map(data)
end
