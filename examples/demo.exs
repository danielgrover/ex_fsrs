# ExFsrs Interactive Demo
#
# This script demonstrates the FSRS-6 algorithm by running cards through
# various state transitions with different ratings.
#
# Usage:
#   iex -S mix
#   c("examples/demo.exs")
#   ExFsrs.Demo.run()
#
# Or from a script:
#   Code.eval_file("examples/demo.exs")

defmodule ExFsrs.Demo do
  def run do
    scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

    test_learning_progression(scheduler)
    test_review_ratings(scheduler)
    test_relearning(scheduler)
    test_extreme_values(scheduler)

    IO.puts("\nDone.")
  end

  defp test_learning_progression(scheduler) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("TEST 1: LEARNING STATE PROGRESSION")
    IO.puts(String.duplicate("=", 50))

    card = ExFsrs.new(state: :learning, step: 0)
    ratings = [:again, :hard, :good, :easy]

    Enum.reduce(ratings, card, fn rating, card ->
      IO.puts("\n" <> String.duplicate("-", 30))
      IO.puts("Rating: #{rating}")
      IO.puts(String.duplicate("-", 30))

      IO.puts("\nBefore review:")
      print_card_state(card, "  ")

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, rating)

      IO.puts("\nAfter review:")
      print_card_state(card, "  ")

      card
    end)
  end

  defp test_review_ratings(scheduler) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("TEST 2: REVIEW STATE WITH DIFFERENT RATINGS")
    IO.puts(String.duplicate("=", 50))

    card =
      ExFsrs.new(
        state: :review,
        stability: 10.0,
        difficulty: 5.0,
        due: DateTime.add(DateTime.utc_now(), 10, :day)
      )

    ratings = [:again, :hard, :good, :easy]

    Enum.reduce(ratings, card, fn rating, card ->
      IO.puts("\n" <> String.duplicate("-", 30))
      IO.puts("Rating: #{rating}")
      IO.puts(String.duplicate("-", 30))

      IO.puts("\nBefore review:")
      print_card_state(card, "  ")

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, rating)

      IO.puts("\nAfter review:")
      print_card_state(card, "  ")

      card
    end)
  end

  defp test_relearning(scheduler) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("TEST 3: RELEARNING STATE")
    IO.puts(String.duplicate("=", 50))

    card =
      ExFsrs.new(
        state: :relearning,
        step: 0,
        stability: 5.0,
        difficulty: 7.0,
        due: DateTime.add(DateTime.utc_now(), 1, :day)
      )

    ratings = [:again, :hard, :good, :easy]

    Enum.reduce(ratings, card, fn rating, card ->
      IO.puts("\n" <> String.duplicate("-", 30))
      IO.puts("Rating: #{rating}")
      IO.puts(String.duplicate("-", 30))

      IO.puts("\nBefore review:")
      print_card_state(card, "  ")

      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, rating)

      IO.puts("\nAfter review:")
      print_card_state(card, "  ")

      card
    end)
  end

  defp test_extreme_values(scheduler) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("TEST 4: EXTREME VALUES")
    IO.puts(String.duplicate("=", 50))

    card =
      ExFsrs.new(
        state: :review,
        stability: 1000.0,
        difficulty: 10.0,
        due: DateTime.add(DateTime.utc_now(), 1000, :day)
      )

    IO.puts("\nBefore review:")
    print_card_state(card, "  ")

    {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good)

    IO.puts("\nAfter review:")
    print_card_state(card, "  ")
  end

  defp print_card_state(card, prefix) do
    IO.puts("#{prefix}State: #{card.state}")
    IO.puts("#{prefix}Step: #{card.step}")

    IO.puts(
      "#{prefix}Stability: #{if card.stability, do: Float.round(card.stability, 4), else: "nil"}"
    )

    IO.puts(
      "#{prefix}Difficulty: #{if card.difficulty, do: Float.round(card.difficulty, 4), else: "nil"}"
    )

    IO.puts("#{prefix}Due: #{card.due}")
    IO.puts("#{prefix}Days until due: #{DateTime.diff(card.due, DateTime.utc_now(), :day)}")
    IO.puts("#{prefix}Last review: #{card.last_review}")
  end
end

# Uncomment to run automatically when loaded:
# ExFsrs.Demo.run()
