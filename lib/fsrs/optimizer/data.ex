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

    With `recency: true` the reviews are also weighted by how recent they are —
    see `apply_recency_weights/2`.
    """
    def build_sequences(logs, opts \\ []) do
      by_card =
        logs
        |> Enum.map(&normalize/1)
        |> Enum.group_by(fn {card_id, _rating, _datetime} -> card_id end)
        |> Enum.sort_by(fn {card_id, _reviews} -> card_id end)
        |> Enum.map(fn {card_id, reviews} -> {card_id, to_sequence(reviews)} end)

      sequences = Enum.map(by_card, fn {card_id, {reviews, _times}} -> {card_id, reviews} end)

      if Keyword.get(opts, :recency, false) do
        times = Map.new(by_card, fn {card_id, {_reviews, times}} -> {card_id, times} end)
        apply_recency_weights(sequences, times)
      else
        sequences
      end
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

    # One card's reviews in time order, plus the moment of each as a unix
    # timestamp so they can later be ranked against other cards' reviews.
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

        {{review(rating, elapsed_days), DateTime.to_unix(datetime, :microsecond)}, datetime}
      end)
      |> elem(0)
      |> Enum.unzip()
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

    `times` maps each card id to when its reviews happened, in the same order as
    that card's reviews in `sequences`. Values may be `DateTime`s or numbers
    (day offsets, unix timestamps); only their order matters. Ranking is over
    scored reviews only, as unscored ones contribute nothing.
    """
    def apply_recency_weights(sequences, times) do
      ranked =
        for {card_id, reviews} <- sequences,
            {review, index} <- Enum.with_index(reviews),
            review.counts_for_loss?,
            do: {times |> Map.fetch!(card_id) |> Enum.at(index) |> time_key(), card_id, index}

      last = Enum.count(ranked) - 1

      weights =
        ranked
        |> Enum.sort()
        |> Enum.with_index()
        |> Map.new(fn {{_time, card_id, index}, rank} ->
          {{card_id, index}, recency_weight(rank, last)}
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

    # The newest review always carries full weight, including when it is the
    # only one.
    defp recency_weight(_rank, 0), do: 1.0
    defp recency_weight(rank, last), do: 0.25 + 0.75 * :math.pow(rank / last, 3)

    # Ranking sorts on Erlang term order, which for a `DateTime` struct compares
    # field by field alphabetically — day before month before year — and is
    # not chronological. Reduce it to a number first.
    defp time_key(%DateTime{} = datetime), do: DateTime.to_unix(datetime, :microsecond)
    defp time_key(number) when is_number(number), do: number

    @doc """
    Stops scoring reviews that sit in outlier intervals, as `fsrs-optimizer` does.

    > #### Measured harmful {: .warning}
    >
    > Across 83 collections this turned a significant 2% improvement into a
    > non-significant 0.8% one, and lost head to head on 48 of 83. It hurts most
    > on collections of 2,000-8,000 reviews, which is where the other upgrades
    > help most. See the README. It is implemented for completeness rather than
    > because it is recommended.

    A collection accumulates reviews that say more about the user's life than
    their memory: the card they came back to after a year away, or an interval
    that only two cards ever had. Fitting a forgetting curve through those
    drags the parameters toward explaining noise.

    Reviews are grouped by the card's first rating and by elapsed days. Buckets
    are then considered sparsest-first, longest-interval-first, and dropped:

      * unconditionally, while fewer than `max(5% of reviews, 20)` have been
        dropped so far;
      * after that budget is spent, only if the bucket is genuinely thin (fewer
        than 6 reviews) or the interval is implausibly long — over 100 days, or
        over 365 for cards first rated Easy, which legitimately get long ones.

    Note the sharp edge, inherited from `fsrs-optimizer`: past the budget the
    "fewer than 6" rule applies to every remaining bucket, so a collection whose
    intervals are *all* thinly populated loses all of them. Real collections
    repeat common intervals hundreds of times and are unaffected, but callers
    should check `num_reviews/1` afterwards rather than assume survival — the
    optimizer returns the default parameters below its training threshold, so
    this degrades rather than failing.

    Dropped reviews stay in the sequence and still advance the
    card's state; they simply stop contributing to the loss, which is what
    `fsrs-optimizer` does by dropping the training row while leaving the card's
    history intact.

    `fsrs-optimizer` also has `remove_non_continuous_rows`, which truncates a
    card at the first gap in its review index. That has no analogue here:
    sequences are built contiguously from a card's own reviews, so there are no
    gaps to find.
    """
    def remove_outliers(sequences) do
      scored =
        for {card_id, reviews} <- sequences,
            {review, index} <- Enum.with_index(reviews),
            review.counts_for_loss?,
            do: {first_rating(reviews), review.elapsed_days, card_id, index}

      dropped = buckets_to_drop(scored)

      Enum.map(sequences, fn {card_id, reviews} ->
        first = first_rating(reviews)
        {card_id, Enum.map(reviews, &unscore_if_dropped(&1, first, dropped))}
      end)
    end

    defp unscore_if_dropped(%Review{counts_for_loss?: true} = review, first_rating, dropped) do
      if MapSet.member?(dropped, {first_rating, review.elapsed_days}),
        do: %{review | counts_for_loss?: false},
        else: review
    end

    defp unscore_if_dropped(review, _first_rating, _dropped), do: review

    defp first_rating([%Review{rating: rating} | _rest]), do: rating

    defp buckets_to_drop(scored) do
      total = Enum.count(scored)
      budget = max(total * 0.05, 20)

      scored
      |> Enum.group_by(fn {rating, _delta_t, _id, _i} -> rating end)
      |> Enum.flat_map(fn {rating, rating_scored} ->
        drop_for_rating(rating, rating_scored, budget)
      end)
      |> MapSet.new()
    end

    defp drop_for_rating(rating, rating_scored, budget) do
      # Sparsest first; among equally sparse, the longest interval first.
      counts =
        rating_scored
        |> Enum.frequencies_by(fn {_rating, delta_t, _id, _i} -> delta_t end)
        |> Enum.sort_by(fn {delta_t, count} -> {count, -delta_t} end)

      # Cards first rated Easy legitimately reach much longer intervals.
      limit = if rating == 4, do: 365, else: 100

      {dropped, _removed} =
        Enum.reduce(counts, {[], 0}, fn {delta_t, count}, {dropped, removed} ->
          cond do
            removed + count < budget ->
              {[{rating, delta_t} | dropped], removed + count}

            count < 6 or delta_t > limit ->
              {[{rating, delta_t} | dropped], removed + count}

            true ->
              {dropped, removed}
          end
        end)

      dropped
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
