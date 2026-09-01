defmodule ExFsrs do
  @moduledoc """
  A flashcard as FSRS-6 sees it, and the simplest way to review one.

  The struct holds everything the scheduler needs to know about a card: where
  it is in the learning state machine, its memory state (stability and
  difficulty), and when it is next due. Cards are plain data; nothing here
  holds state between calls.

      card = ExFsrs.new()
      {card, log} = ExFsrs.review_card(card, :good)
      card.due          #=> ten minutes from now, at the second learning step
      card.stability    #=> 2.3065

  The functions in this module use a default `ExFsrs.Scheduler`. To change
  retention, steps, fuzzing or the model weights, build a scheduler and call
  `ExFsrs.Scheduler.review_card/5` instead.

  ## Fields

    * `card_id` — any integer identifying the card. Defaults to the current
      time in milliseconds.
    * `state` — `:learning` for a new card, `:review` once it has graduated,
      `:relearning` after a lapse.
    * `step` — position in the learning or relearning steps, or `nil` in
      `:review`.
    * `stability` — days until retrievability is expected to fall to 90%.
      `nil` before the first review.
    * `difficulty` — 1.0 (easiest) to 10.0. `nil` before the first review.
    * `due` — when the card should next be shown.
    * `last_review` — when it was last shown, or `nil`.

  ## Storage

  `to_map/1` and `from_map/1` convert to and from string-keyed maps with
  ISO 8601 datetimes, ready for JSON or a database. `from_map/1` also accepts
  atom keys and already-parsed `DateTime`s, and raises `ArgumentError` on a
  value it cannot interpret rather than substituting a default.
  """

  @typedoc "A card."
  @type t :: %__MODULE__{
          card_id: integer(),
          state: state(),
          step: non_neg_integer() | nil,
          stability: float() | nil,
          difficulty: float() | nil,
          due: DateTime.t(),
          last_review: DateTime.t() | nil
        }

  @typedoc "How the review went, from a lapse to an effortless recall."
  @type rating :: :again | :hard | :good | :easy

  @typedoc "Where the card is in the learning state machine."
  @type state :: :learning | :review | :relearning

  @states %{"learning" => :learning, "review" => :review, "relearning" => :relearning}

  defstruct [:card_id, :state, :step, :stability, :difficulty, :due, :last_review]

  @doc """
  Builds a card.

  With no options this is a new card: in `:learning` at step 0, due now, with
  no memory state. Any field can be overridden by keyword, which is how a card
  is reconstructed from storage when `from_map/1` is not a fit.

      iex> card = ExFsrs.new(card_id: 7, due: ~U[2024-01-01 00:00:00Z])
      iex> {card.card_id, card.state, card.step, card.stability}
      {7, :learning, 0, nil}
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      card_id: Keyword.get_lazy(opts, :card_id, fn -> System.system_time(:millisecond) end),
      state: Keyword.get(opts, :state, :learning),
      step: Keyword.get(opts, :step, 0),
      stability: opts[:stability],
      difficulty: opts[:difficulty],
      due: Keyword.get_lazy(opts, :due, &DateTime.utc_now/0),
      last_review: opts[:last_review]
    }
  end

  @doc """
  Reviews a card with the default scheduler.

  See `ExFsrs.Scheduler.review_card/5` for the details and for reviewing with
  a custom scheduler. Returns `{updated_card, review_log}`.
  """
  @spec review_card(t(), rating(), DateTime.t(), non_neg_integer() | nil) ::
          {t(), ExFsrs.ReviewLog.t()}
  def review_card(card, rating, review_datetime \\ DateTime.utc_now(), review_duration \\ nil) do
    ExFsrs.Scheduler.review_card(
      ExFsrs.Scheduler.new(),
      card,
      rating,
      review_datetime,
      review_duration
    )
  end

  @doc """
  The probability that the card is still remembered at `datetime`, under the
  default scheduler.

  `0.0` for a card that has never been reviewed. See
  `ExFsrs.Scheduler.get_retrievability/3`.
  """
  @spec get_retrievability(t(), DateTime.t()) :: float()
  def get_retrievability(%__MODULE__{} = card, datetime \\ DateTime.utc_now()) do
    ExFsrs.Scheduler.get_retrievability(card, datetime, ExFsrs.Scheduler.new())
  end

  @doc """
  Rebuilds a card by replaying its review logs through the default scheduler.

  See `ExFsrs.Scheduler.reschedule_card/3`.
  """
  @spec reschedule_card(t(), [ExFsrs.ReviewLog.t() | map()]) :: t()
  def reschedule_card(%__MODULE__{} = card, review_logs) when is_list(review_logs) do
    ExFsrs.Scheduler.reschedule_card(ExFsrs.Scheduler.new(), card, review_logs)
  end

  @doc """
  Converts a card to a string-keyed map for storage.

      iex> card = ExFsrs.new(card_id: 7, due: ~U[2024-01-01 00:00:00Z])
      iex> ExFsrs.to_map(card)
      %{
        "card_id" => 7,
        "state" => "learning",
        "step" => 0,
        "stability" => nil,
        "difficulty" => nil,
        "due" => "2024-01-01T00:00:00Z",
        "last_review" => nil
      }
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = card) do
    %{
      "card_id" => card.card_id,
      "state" => Atom.to_string(card.state),
      "step" => card.step,
      "stability" => card.stability,
      "difficulty" => card.difficulty,
      "due" => DateTime.to_iso8601(card.due),
      "last_review" => card.last_review && DateTime.to_iso8601(card.last_review)
    }
  end

  @doc """
  Builds a card from a map produced by `to_map/1`.

  Keys may be strings or atoms; datetimes may be ISO 8601 strings or
  `DateTime`s. Raises `ArgumentError` on an unrecognised state or an
  unparseable datetime.

      iex> card = ExFsrs.new(card_id: 7, due: ~U[2024-01-01 00:00:00Z])
      iex> card |> ExFsrs.to_map() |> ExFsrs.from_map() == card
      true
  """
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    %__MODULE__{
      card_id: field(map, :card_id),
      state: parse_state(field(map, :state)),
      step: field(map, :step),
      stability: field(map, :stability),
      difficulty: field(map, :difficulty),
      due: parse_datetime!(field(map, :due), "due"),
      last_review: parse_optional_datetime!(field(map, :last_review), "last_review")
    }
  end

  @doc false
  # Reads a field by atom key, falling back to its string form.
  def field(map, key) do
    case Map.fetch(map, key) do
      {:ok, value} -> value
      :error -> Map.get(map, Atom.to_string(key))
    end
  end

  @doc false
  def parse_datetime!(%DateTime{} = datetime, _field), do: datetime

  def parse_datetime!(value, field) when is_binary(value) do
    case DateTime.from_iso8601(value) do
      {:ok, datetime, _offset} ->
        datetime

      {:error, _reason} ->
        raise ArgumentError, "invalid ISO 8601 datetime for #{field}: #{inspect(value)}"
    end
  end

  def parse_datetime!(value, field) do
    raise ArgumentError, "invalid datetime for #{field}: #{inspect(value)}"
  end

  defp parse_optional_datetime!(nil, _field), do: nil
  defp parse_optional_datetime!(value, field), do: parse_datetime!(value, field)

  defp parse_state(state) when state in [:learning, :review, :relearning], do: state

  defp parse_state(state) when is_binary(state) and is_map_key(@states, state),
    do: Map.fetch!(@states, state)

  defp parse_state(state), do: raise(ArgumentError, "invalid card state: #{inspect(state)}")
end
