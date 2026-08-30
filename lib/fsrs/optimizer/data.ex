if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Data do
    @moduledoc """
    Turns review logs into the per-card sequences the optimizer trains on.

    Each card becomes one sequence of reviews in chronological order. A review
    contributes to the loss only when it follows an earlier review by at least
    one full day: same-day repeats update the card's state but are not
    predictions the model is scored on.

    Ordering matters for parity with py-fsrs, whose minibatch boundaries — and
    therefore its whole optimization trajectory — depend on it. Cards are
    ordered by ascending `card_id` and reviews within a card by ascending
    datetime, with ties keeping input order (both `Enum.sort_by/3` and Python's
    `sorted` are stable).
    """

    @max_seq_len 64

    @rating_numbers %{again: 1, hard: 2, good: 3, easy: 4}

    defmodule Review do
      @moduledoc """
      A single review, prepared for training.

      `weight` scales this review's contribution to the loss. It is 1.0 unless
      something has re-weighted the collection — see
      `ExFsrs.Optimizer.Data.apply_recency_weights/2`.
      """
      defstruct [:rating, :elapsed_days, :label, :counts_for_loss?, weight: 1.0]
    end

    @doc "Maximum reviews used per card, matching py-fsrs."
    def max_seq_len, do: @max_seq_len

    @doc """
    Builds training sequences from review logs.

    Accepts `%ExFsrs.ReviewLog{}` structs, `{card_id, rating, review_datetime}`
    tuples, or any enumerable of either. Returns
    `[{card_id, [%Review{}]}]` ordered by ascending `card_id`.
    """
    def build_sequences(logs) do
      logs
      |> Enum.map(&normalize/1)
      |> Enum.group_by(fn {card_id, _rating, _datetime} -> card_id end)
      |> Enum.sort_by(fn {card_id, _reviews} -> card_id end)
      |> Enum.map(fn {card_id, reviews} ->
        {card_id, to_sequence(reviews)}
      end)
    end

    defp normalize(%ExFsrs.ReviewLog{card: card, rating: rating, review_datetime: datetime}) do
      {card.card_id, rating_number(rating), datetime}
    end

    defp normalize({card_id, rating, datetime}) do
      {card_id, rating_number(rating), datetime}
    end

    defp rating_number(rating) when rating in [:again, :hard, :good, :easy] do
      @rating_numbers[rating]
    end

    defp rating_number(rating) when rating in 1..4, do: rating

    defp to_sequence(reviews) do
      reviews
      |> Enum.sort_by(fn {_card_id, _rating, datetime} -> datetime end, DateTime)
      |> Enum.take(@max_seq_len)
      |> Enum.map_reduce(nil, fn {_card_id, rating, datetime}, last_review ->
        elapsed_days =
          case last_review do
            # -1 marks a first review: no elapsed time, no prediction to score.
            nil -> -1
            previous -> DateTime.diff(datetime, previous, :day)
          end

        {review(rating, elapsed_days), datetime}
      end)
      |> elem(0)
    end

    defp review(rating, elapsed_days) do
      %Review{
        rating: rating,
        elapsed_days: elapsed_days,
        label: if(rating == 1, do: 0.0, else: 1.0),
        counts_for_loss?: elapsed_days > 0
      }
    end

    @doc """
    Builds training sequences from reviews whose elapsed time is already known.

    Some sources record the gap since a card's previous review rather than an
    absolute timestamp. The Anki revlogs dataset is one, and its `elapsed_days`
    field is exactly what a `Review` needs, so nothing has to be reconstructed
    from synthesized datetimes.

    Takes `{card_id, rating, elapsed_days}` tuples, already in chronological
    order per card, with `-1` marking a card's first review. Cards come back
    ordered by ascending `card_id`, matching `build_sequences/1`.
    """
    def build_sequences_from_elapsed(reviews) do
      reviews
      |> Enum.group_by(
        fn {card_id, _rating, _elapsed} -> card_id end,
        fn {_card_id, rating, elapsed} -> {rating_number(rating), elapsed} end
      )
      |> Enum.sort_by(fn {card_id, _reviews} -> card_id end)
      |> Enum.map(fn {card_id, card_reviews} ->
        reviews =
          card_reviews
          |> Enum.take(@max_seq_len)
          |> Enum.map(fn {rating, elapsed_days} -> review(rating, elapsed_days) end)

        {card_id, reviews}
      end)
    end

    @doc """
    Weights reviews by how recent they are, as `fsrs-rs` does.

    A collection's older reviews describe a memory that has since changed — the
    material got easier, the user got better at it, their settings moved. Recent
    reviews are more likely to describe the present, so `fsrs-rs` ramps weights
    from 0.25 for the oldest scored review to 1.0 for the newest, cubically:
    `0.25 + 0.75 * (rank / (n - 1))^3`. The cube keeps most of the collection
    near the floor and lets only the tail carry full weight.

    `days` maps each card id to the day offsets of its reviews, in the same
    order as that card's reviews in `sequences`. Ranking is over scored reviews
    only, since unscored ones contribute nothing to weight.
    """
    def apply_recency_weights(sequences, days) do
      ranked =
        for {card_id, reviews} <- sequences,
            {review, index} <- Enum.with_index(reviews),
            review.counts_for_loss?,
            do: {days |> Map.fetch!(card_id) |> Enum.at(index), card_id, index}

      total = Enum.count(ranked)
      last = max(total - 1, 1)

      weights =
        ranked
        |> Enum.sort()
        |> Enum.with_index()
        |> Map.new(fn {{_day, card_id, index}, rank} ->
          {{card_id, index}, 0.25 + 0.75 * :math.pow(rank / last, 3)}
        end)

      Enum.map(sequences, fn {card_id, reviews} ->
        reviews =
          reviews
          |> Enum.with_index()
          |> Enum.map(fn {review, index} ->
            %{review | weight: Map.get(weights, {card_id, index}, 1.0)}
          end)

        {card_id, reviews}
      end)
    end

    @doc """
    Counts the reviews that contribute to the loss.

    py-fsrs uses this both to decide whether there is enough data to train at
    all and to size the cosine-annealing schedule.
    """
    def num_reviews(sequences) do
      Enum.reduce(sequences, 0, fn {_card_id, reviews}, total ->
        total + Enum.count(reviews, & &1.counts_for_loss?)
      end)
    end
  end
end
