defmodule ExFsrs.ReferenceTest do
  @moduledoc """
  Reference tests ported from py-fsrs (open-spaced-repetition/py-fsrs) test suite.
  These verify our FSRS-6 implementation matches the canonical Python implementation.
  """
  use ExUnit.Case, async: true

  @start_datetime ~U[2022-11-29 12:30:00Z]

  defp review_sequence(ratings, opts \\ []) do
    scheduler_opts = Keyword.get(opts, :scheduler_opts, [])
    start = Keyword.get(opts, :start_datetime, @start_datetime)

    scheduler = ExFsrs.Scheduler.new([enable_fuzzing: false] ++ scheduler_opts)
    card = ExFsrs.new(state: :learning, step: 0)

    {final_card, _now, intervals} =
      Enum.reduce(ratings, {card, start, []}, fn rating, {card, now, intervals} ->
        {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)
        interval_days = DateTime.diff(updated_card.due, now, :day)
        {updated_card, updated_card.due, intervals ++ [interval_days]}
      end)

    {final_card, intervals}
  end

  describe "py-fsrs test_review_card" do
    test "interval history matches reference for Good*6, Again*2, Good*5" do
      ratings = [
        :good,
        :good,
        :good,
        :good,
        :good,
        :good,
        :again,
        :again,
        :good,
        :good,
        :good,
        :good,
        :good
      ]

      {_card, intervals} = review_sequence(ratings)

      assert intervals == [0, 2, 11, 46, 163, 498, 0, 0, 2, 4, 7, 12, 21]
    end
  end

  describe "py-fsrs test_memo_state" do
    test "stability and difficulty match after Again, Good*5 at specified intervals" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)

      # py-fsrs reviews at start + cumulative sum of ivl_history days
      # ivl_history is the INPUT schedule, not the output
      ratings_and_ivls = [
        {:again, 0},
        {:good, 0},
        {:good, 1},
        {:good, 3},
        {:good, 8},
        {:good, 21}
      ]

      review_datetime = @start_datetime

      {final_card, _} =
        Enum.reduce(ratings_and_ivls, {card, review_datetime}, fn {rating, ivl}, {card, dt} ->
          review_at = DateTime.add(dt, ivl, :day)
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, review_at)
          {updated, review_at}
        end)

      assert_in_delta final_card.stability, 53.62691, 0.01
      assert_in_delta final_card.difficulty, 6.3574867, 0.001
    end
  end

  describe "py-fsrs test_repeated_correct_reviews" do
    test "difficulty floors at 1.0 after many easy reviews" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      now = @start_datetime

      {final_card, _now} =
        Enum.reduce(1..10, {card, now}, fn _, {card, now} ->
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, now)
          {updated, DateTime.add(now, 1, :second)}
        end)

      assert final_card.difficulty == 1.0
    end
  end

  describe "py-fsrs test_stability_lower_bound" do
    test "stability never drops below 0.001 after many again ratings" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      now = @start_datetime

      # First get card into review state
      {card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)
      now = card.due
      {card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)
      now = card.due

      # Hit with :again 1000 times
      {final_card, _now} =
        Enum.reduce(1..1000, {card, now}, fn _, {card, now} ->
          next_now = DateTime.add(now, 1, :day)
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :again, next_now)
          {updated, next_now}
        end)

      assert final_card.stability >= 0.001
    end
  end

  describe "py-fsrs test_retrievability" do
    test "returns 0.9 when elapsed_days equals stability" do
      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      now = DateTime.add(@start_datetime, 10, :day)
      r = ExFsrs.get_retrievability(card, now)

      assert_in_delta r, 0.9, 0.001
    end
  end

  describe "py-fsrs test_maximum_interval" do
    test "all intervals respect maximum_interval" do
      max_ivl = 100

      scheduler =
        ExFsrs.Scheduler.new(enable_fuzzing: false, maximum_interval: max_ivl)

      card = ExFsrs.new(state: :learning, step: 0)
      now = @start_datetime

      # Do many good reviews, all intervals should be <= max_ivl
      {_card, _now} =
        Enum.reduce(1..20, {card, now}, fn _, {card, now} ->
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)
          interval_days = DateTime.diff(updated.due, now, :day)

          if updated.state == :review do
            assert interval_days <= max_ivl,
                   "interval #{interval_days} exceeds maximum #{max_ivl}"
          end

          {updated, updated.due}
        end)
    end
  end

  describe "py-fsrs test_no_learning_steps" do
    test "again goes straight to review with empty learning steps" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, learning_steps: [])
      card = ExFsrs.new(state: :learning, step: 0)
      now = @start_datetime

      {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :again, now)
      assert updated.state == :review
    end
  end

  describe "py-fsrs test_no_relearning_steps" do
    test "again stays in review with empty relearning steps" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, relearning_steps: [])

      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: DateTime.add(@start_datetime, -10, :day)
        )

      {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :again, @start_datetime)
      assert updated.state == :review
    end
  end

  describe "py-fsrs learning step transitions" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      {:ok, scheduler: scheduler, card: card, now: @start_datetime}
    end

    test "good advances step then graduates", %{scheduler: scheduler, card: card, now: now} do
      # First good: stay in learning, advance to step 1 (~10 min)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)
      assert card.state == :learning
      assert card.step == 1
      assert DateTime.diff(card.due, now, :minute) == 10

      # Second good: graduate to review (>= 1 day)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      assert card.state == :review
      assert card.step == nil
    end

    test "again always resets to step 0", %{scheduler: scheduler, card: card, now: now} do
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, now)
      assert card.state == :learning
      assert card.step == 0
      assert DateTime.diff(card.due, now, :minute) == 1
    end

    test "hard stays at current step with interpolated interval", %{
      scheduler: scheduler,
      card: card,
      now: now
    } do
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, now)
      assert card.state == :learning
      assert card.step == 0
      # (1 + 10) / 2 = 5.5 minutes, rounded to 6
      assert DateTime.diff(card.due, now, :minute) == 6
    end

    test "easy immediately graduates to review", %{scheduler: scheduler, card: card, now: now} do
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, now)
      assert card.state == :review
      assert card.step == nil
      assert DateTime.diff(card.due, now, :day) >= 1
    end
  end

  describe "py-fsrs relearning transitions" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = @start_datetime

      # Get a card into relearning state
      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: DateTime.add(now, -10, :day)
        )

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, now)
      assert card.state == :relearning

      {:ok, scheduler: scheduler, card: card, now: now}
    end

    test "again resets to step 0 with relearning step interval", %{
      scheduler: scheduler,
      card: card,
      now: now
    } do
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, card.due)
      assert card.state == :relearning
      assert card.step == 0
      assert DateTime.diff(card.due, now, :minute) == 20
    end

    test "good graduates back to review", %{scheduler: scheduler, card: card} do
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      assert card.state == :review
      assert card.step == nil
    end
  end

  describe "py-fsrs hard with one learning step" do
    test "hard gives 1.5x the single step" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, learning_steps: [10.0])
      card = ExFsrs.new(state: :learning, step: 0)

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, @start_datetime)
      assert card.state == :learning
      # 10 * 1.5 = 15 minutes
      assert DateTime.diff(card.due, @start_datetime, :minute) == 15
    end
  end

  describe "py-fsrs hard with one relearning step" do
    test "hard gives 1.5x the single relearning step" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, relearning_steps: [10.0])

      card =
        ExFsrs.new(
          state: :relearning,
          step: 0,
          stability: 5.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, @start_datetime)
      assert card.state == :relearning
      # 10 * 1.5 = 15 minutes
      assert DateTime.diff(card.due, @start_datetime, :minute) == 15
    end
  end

  describe "py-fsrs test_review_state" do
    test "good reviews stay in review, again moves to relearning" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)

      # Graduate through learning
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      assert card.state == :review
      assert card.step == nil

      # Good in review stays in review, due >= 1 day
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      assert card.state == :review
      assert DateTime.diff(card.due, prev_due, :day) >= 1

      # Again moves to relearning, due in 10 minutes
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, card.due)
      assert card.state == :relearning
      assert DateTime.diff(card.due, prev_due, :minute) == 10
    end
  end

  describe "py-fsrs test_relearning" do
    test "again stays in relearning step 0, good graduates to review" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)

      # Graduate through learning to review
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)

      # Again into relearning
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, card.due)
      assert card.state == :relearning
      assert card.step == 0
      assert DateTime.diff(card.due, prev_due, :minute) == 10

      # Again stays in relearning step 0
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, card.due)
      assert card.state == :relearning
      assert card.step == 0
      assert DateTime.diff(card.due, prev_due, :minute) == 10

      # Good graduates to review, due >= 1 day
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      assert card.state == :review
      assert card.step == nil
      assert DateTime.diff(card.due, prev_due, :day) >= 1
    end
  end

  describe "py-fsrs test_one_card_multiple_schedulers" do
    test "card works with different scheduler configurations" do
      two_learning = ExFsrs.Scheduler.new(learning_steps: [1.0, 10.0])
      one_learning = ExFsrs.Scheduler.new(learning_steps: [1.0])
      no_learning = ExFsrs.Scheduler.new(learning_steps: [])

      two_relearning = ExFsrs.Scheduler.new(relearning_steps: [1.0, 10.0])
      one_relearning = ExFsrs.Scheduler.new(relearning_steps: [1.0])
      no_relearning = ExFsrs.Scheduler.new(relearning_steps: [])

      now = @start_datetime
      card = ExFsrs.new(state: :learning, step: 0)

      # Two learning steps: good -> learning step 1
      {card, _} = ExFsrs.Scheduler.review_card(two_learning, card, :good, now)
      assert card.state == :learning
      assert card.step == 1

      # One learning step: again -> learning step 0
      {card, _} = ExFsrs.Scheduler.review_card(one_learning, card, :again, now)
      assert card.state == :learning
      assert card.step == 0

      # No learning steps: hard -> review
      {card, _} = ExFsrs.Scheduler.review_card(no_learning, card, :hard, now)
      assert card.state == :review
      assert card.step == nil

      # Two relearning steps: again -> relearning step 0
      {card, _} = ExFsrs.Scheduler.review_card(two_relearning, card, :again, now)
      assert card.state == :relearning
      assert card.step == 0

      # Two relearning steps: good -> relearning step 1
      {card, _} = ExFsrs.Scheduler.review_card(two_relearning, card, :good, now)
      assert card.state == :relearning
      assert card.step == 1

      # One relearning step: again -> relearning step 0
      {card, _} = ExFsrs.Scheduler.review_card(one_relearning, card, :again, now)
      assert card.state == :relearning
      assert card.step == 0

      # No relearning steps: hard -> review
      {card, _} = ExFsrs.Scheduler.review_card(no_relearning, card, :hard, now)
      assert card.state == :review
      assert card.step == nil
    end
  end

  describe "py-fsrs test_learning_card_rate_hard_second_learning_step" do
    test "hard at step 1 gives the second step interval" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, learning_steps: [1.0, 10.0])
      card = ExFsrs.new(state: :learning, step: 0)

      # Good to advance to step 1
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)
      assert card.state == :learning
      assert card.step == 1

      # Hard at step 1: interval should be the second step (10 min)
      due_after_first = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, due_after_first)
      assert card.state == :learning
      assert card.step == 1
      assert DateTime.diff(card.due, due_after_first, :minute) == 10
    end
  end

  describe "py-fsrs test_long_term_stability_learning_state" do
    test "relearning card reviewed a day late uses long-term stability" do
      scheduler = ExFsrs.Scheduler.new()
      card = ExFsrs.new(state: :learning, step: 0)

      # Graduate to review with easy
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, card.due)
      assert card.state == :review

      # Lapse to relearning
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, card.due)
      assert card.state == :relearning

      # Review a full day after due (triggers long-term stability calculation)
      late_review = DateTime.add(card.due, 1, :day)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, late_review)
      assert card.state == :review
    end
  end

  # Helper to build review logs from a card through a sequence of ratings
  defp build_review_logs(card, ratings, scheduler) do
    now = @start_datetime

    {_final_card, logs, _now} =
      Enum.reduce(ratings, {card, [], now}, fn rating, {card, logs, now} ->
        {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)

        log_entry = %{
          card: %{card_id: card.card_id},
          rating: rating,
          review_datetime: now
        }

        {updated_card, logs ++ [log_entry], updated_card.due}
      end)

    logs
  end

  describe "py-fsrs test_reschedule_card_same_scheduler" do
    test "rescheduling with same scheduler produces equivalent card" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      now = @start_datetime
      ratings = [:good, :good, :good, :again, :good]

      # Review the card through all ratings
      {final_card, logs, _now} =
        Enum.reduce(ratings, {card, [], now}, fn rating, {card, logs, now} ->
          {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)

          log_entry = %{
            card: %{card_id: card.card_id},
            rating: rating,
            review_datetime: now
          }

          {updated_card, logs ++ [log_entry], updated_card.due}
        end)

      # Reschedule with the same scheduler
      rescheduled = ExFsrs.Scheduler.reschedule_card(scheduler, card, logs)

      # Should produce equivalent state
      assert rescheduled.card_id == final_card.card_id
      assert rescheduled.state == final_card.state
      assert rescheduled.step == final_card.step
      assert_in_delta rescheduled.stability, final_card.stability, 0.0001
      assert_in_delta rescheduled.difficulty, final_card.difficulty, 0.0001
      assert rescheduled.due == final_card.due
    end
  end

  describe "py-fsrs test_reschedule_card_different_parameters" do
    test "different parameters change stability and difficulty" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      ratings = [:good, :good, :good, :again, :good]

      logs = build_review_logs(card, ratings, scheduler)

      # Review with original scheduler to get final card
      now = @start_datetime

      {final_card, _now} =
        Enum.reduce(ratings, {card, now}, fn rating, {card, now} ->
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)
          {updated, updated.due}
        end)

      # Reschedule with different parameters
      different_params = [
        0.1,
        0.7,
        1.5,
        6.0,
        5.5,
        0.5,
        2.0,
        0.01,
        1.5,
        0.2,
        0.9,
        1.2,
        0.05,
        0.3,
        1.5,
        0.7,
        1.5,
        0.6,
        0.1,
        0.07,
        0.16
      ]

      different_scheduler =
        ExFsrs.Scheduler.new(parameters: different_params, enable_fuzzing: false)

      rescheduled = ExFsrs.Scheduler.reschedule_card(different_scheduler, card, logs)

      assert rescheduled.card_id == final_card.card_id
      assert rescheduled.state == final_card.state
      assert rescheduled.step == final_card.step
      # Stability and difficulty should be different
      assert rescheduled.stability != final_card.stability
      assert rescheduled.difficulty != final_card.difficulty
      # Due date should be different
      assert rescheduled.due != final_card.due
    end
  end

  describe "py-fsrs test_reschedule_card_different_desired_retention" do
    test "different retention changes due but not stability/difficulty" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      ratings = [:good, :good, :good, :again, :good]

      logs = build_review_logs(card, ratings, scheduler)

      # Review with original scheduler
      now = @start_datetime

      {final_card, _now} =
        Enum.reduce(ratings, {card, now}, fn rating, {card, now} ->
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)
          {updated, updated.due}
        end)

      # Reschedule with different desired_retention
      different_scheduler =
        ExFsrs.Scheduler.new(desired_retention: 0.8, enable_fuzzing: false)

      rescheduled = ExFsrs.Scheduler.reschedule_card(different_scheduler, card, logs)

      # Stability and difficulty should be the same (same parameters)
      assert_in_delta rescheduled.stability, final_card.stability, 0.0001
      assert_in_delta rescheduled.difficulty, final_card.difficulty, 0.0001
      # Due date should be different (lower retention = longer intervals)
      assert DateTime.compare(final_card.due, rescheduled.due) == :lt
    end
  end

  describe "py-fsrs test_reschedule_card_different_learning_steps" do
    test "different learning steps change state and step" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      ratings = [:good, :good, :good, :again, :good]

      logs = build_review_logs(card, ratings, scheduler)

      # Review with original scheduler
      now = @start_datetime

      {final_card, _now} =
        Enum.reduce(ratings, {card, now}, fn rating, {card, now} ->
          {updated, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)
          {updated, updated.due}
        end)

      # Reschedule with many more learning steps (forces card to stay in learning longer)
      different_scheduler =
        ExFsrs.Scheduler.new(
          learning_steps: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          enable_fuzzing: false
        )

      rescheduled = ExFsrs.Scheduler.reschedule_card(different_scheduler, card, logs)

      # State and step should be different
      assert rescheduled.state != final_card.state or rescheduled.step != final_card.step
      # Stability and difficulty should be same (same parameters)
      assert_in_delta rescheduled.stability, final_card.stability, 0.0001
      assert_in_delta rescheduled.difficulty, final_card.difficulty, 0.0001
      # Due should be different (different steps)
      assert DateTime.compare(final_card.due, rescheduled.due) == :gt
    end
  end

  describe "py-fsrs test_reschedule_card_wrong_review_logs" do
    test "raises error when review log card_id doesn't match" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(card_id: 1, state: :learning, step: 0)

      logs = [
        %{card: %{card_id: 1}, rating: :good, review_datetime: @start_datetime},
        %{card: %{card_id: 2}, rating: :good, review_datetime: @start_datetime}
      ]

      assert_raise ArgumentError, ~r/ReviewLog card_id 2 does not match Card card_id 1/, fn ->
        ExFsrs.Scheduler.reschedule_card(scheduler, card, logs)
      end
    end
  end

  describe "reschedule_card with out-of-order logs" do
    test "produces same result regardless of log order" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      ratings = [:good, :good, :good, :again, :good]

      ordered_logs = build_review_logs(card, ratings, scheduler)
      shuffled_logs = Enum.shuffle(ordered_logs)

      rescheduled_ordered = ExFsrs.Scheduler.reschedule_card(scheduler, card, ordered_logs)
      rescheduled_shuffled = ExFsrs.Scheduler.reschedule_card(scheduler, card, shuffled_logs)

      assert_in_delta rescheduled_ordered.stability, rescheduled_shuffled.stability, 0.0001
      assert_in_delta rescheduled_ordered.difficulty, rescheduled_shuffled.difficulty, 0.0001
      assert rescheduled_ordered.state == rescheduled_shuffled.state
      assert rescheduled_ordered.due == rescheduled_shuffled.due
    end
  end

  describe "empty learning_steps with all ratings" do
    test "all ratings graduate directly to review" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, learning_steps: [])
      now = @start_datetime

      for rating <- [:again, :hard, :good, :easy] do
        card = ExFsrs.new(state: :learning, step: 0)
        {updated, _} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)

        assert updated.state == :review,
               "#{rating} with empty learning_steps should graduate to review"

        assert updated.step == nil
        assert DateTime.diff(updated.due, now, :day) >= 1
      end
    end
  end

  describe "py-fsrs test_relearning_card_rate_hard_two_relearning_steps" do
    test "hard at step 0 gives avg of steps, hard at step 1 gives step 1" do
      scheduler =
        ExFsrs.Scheduler.new(enable_fuzzing: false, relearning_steps: [1.0, 10.0])

      card = ExFsrs.new(state: :learning, step: 0)

      # Graduate to review, then lapse to relearning
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, card.due)
      assert card.state == :review

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, card.due)
      assert card.state == :relearning
      assert card.step == 0

      # Hard at step 0: (1 + 10) / 2 = 5.5 minutes, rounded to 6
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, prev_due)
      assert card.state == :relearning
      assert card.step == 0
      assert DateTime.diff(card.due, prev_due, :minute) == 6

      # Good to advance to step 1
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, card.due)
      assert card.state == :relearning
      assert card.step == 1

      # Hard at step 1: interval = step 1 = 10 min
      prev_due = card.due
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, prev_due)
      assert card.state == :relearning
      assert card.step == 1
      assert DateTime.diff(card.due, prev_due, :minute) == 10

      # Easy graduates to review
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, prev_due)
      assert card.state == :review
      assert card.step == nil
    end
  end
end
