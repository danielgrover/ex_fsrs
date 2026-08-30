defmodule ExFsrs.OptimizerTest do
  @moduledoc """
  Checks the optimizer against a recorded run of py-fsrs's own optimizer over
  py-fsrs's own review-log fixture (see `test/fixtures/generate_trace.py`).

  The recorded run reproduces py-fsrs's published `test_optimal_parameters` to
  8.9e-16, so the trace is ground truth rather than a reimplementation.

  Exact agreement depends on replaying py-fsrs's card ordering, which sets the
  minibatch boundaries and therefore the whole Adam trajectory. The trace
  records that ordering per epoch and the tests inject it via `:card_orders`.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Fixtures
  alias ExFsrs.Optimizer
  alias ExFsrs.Optimizer.Data

  setup_all do
    trace = Fixtures.trace()
    logs = Fixtures.review_log_tuples()

    {:ok, trace: trace, logs: logs, sequences: Data.build_sequences(logs)}
  end

  describe "data pipeline" do
    test "reproduces py-fsrs's card and scored-review counts", ctx do
      assert Enum.count_until(ctx.sequences, 1206) == 1205
      assert Data.num_reviews(ctx.sequences) == ctx.trace["num_reviews"]
    end

    test "review log structs and tuples produce identical sequences", ctx do
      from_tuples = ctx.sequences
      from_structs = Data.build_sequences(Fixtures.review_logs())

      assert from_tuples == from_structs
    end

    test "a review is scored only when it follows an earlier one by a full day" do
      start = ~U[2024-01-01 09:00:00Z]

      logs = [
        {1, :good, start},
        # same day: updates state, not scored
        {1, :good, DateTime.add(start, 3600)},
        {1, :again, DateTime.add(start, 3 * 86_400)}
      ]

      assert [{1, [first, same_day, next_day]}] = Data.build_sequences(logs)

      refute first.counts_for_loss?
      assert first.elapsed_days == -1

      refute same_day.counts_for_loss?
      assert same_day.elapsed_days == 0

      assert next_day.counts_for_loss?
      # Measured from the previous review at 10:00 on day 1, not from the first.
      assert next_day.elapsed_days == 2
      assert next_day.label == 0.0
    end
  end

  describe "minibatch partitioning" do
    test "matches py-fsrs's minibatch sizes for every epoch", ctx do
      expected =
        Enum.map(ctx.trace["epochs"], fn epoch ->
          ctx.trace["steps"]
          |> Enum.slice(epoch["first_step"]..epoch["last_step"])
          |> Enum.map(& &1["n_losses"])
        end)

      actual =
        Enum.map(ctx.trace["epochs"], fn epoch ->
          Optimizer.minibatch_sizes(ctx.sequences, epoch["card_order"])
        end)

      assert actual == expected
    end
  end

  describe "batch_loss/2" do
    test "matches py-fsrs at the default parameters", ctx do
      loss = Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      assert_in_delta loss, ctx.trace["batch_loss_at_defaults"], 1.0e-12
    end

    test "matches py-fsrs at its optimized parameters", ctx do
      loss = Optimizer.batch_loss(ctx.sequences, ctx.trace["final_parameters"])

      assert_in_delta loss, ctx.trace["batch_loss_at_expected"], 1.0e-12
    end

    test "accepts raw review logs as well as built sequences", ctx do
      from_sequences = Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)
      from_logs = Optimizer.batch_loss(ctx.logs, ExFsrs.Scheduler.new().parameters)

      assert from_sequences == from_logs
    end

    test "raises when nothing can be scored" do
      logs = [{1, :good, ~U[2024-01-01 09:00:00Z]}]

      assert_raise ArgumentError, ~r/no reviews to score/, fn ->
        Optimizer.batch_loss(logs, ExFsrs.Scheduler.new().parameters)
      end
    end

    test "optimized parameters beat the defaults", ctx do
      optimized = Optimizer.batch_loss(ctx.sequences, ctx.trace["final_parameters"])
      defaults = Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      assert optimized < defaults
    end
  end

  describe "compute_optimal_parameters/2" do
    test "returns the defaults unchanged when there is too little data" do
      start = ~U[2024-01-01 09:00:00Z]

      logs =
        for card_id <- 1..10, review <- 0..4 do
          {card_id, :good, DateTime.add(start, review * 86_400)}
        end

      assert Optimizer.compute_optimal_parameters(logs) ==
               ExFsrs.Scheduler.new().parameters
    end

    # One full optimization run takes ~90s, so this covers every property of the
    # result in a single pass rather than re-running it per assertion.
    @tag timeout: 600_000
    test "reproduces py-fsrs's optimized parameters", ctx do
      orders = Enum.map(ctx.trace["epochs"], & &1["card_order"])

      parameters = Optimizer.compute_optimal_parameters(ctx.logs, card_orders: orders)

      assert Enum.count_until(parameters, 22) == 21

      for {{actual, expected}, index} <-
            Enum.with_index(Enum.zip(parameters, ctx.trace["final_parameters"])) do
        assert_in_delta actual,
                        expected,
                        1.0e-9,
                        "w[#{index}] diverged from py-fsrs: #{actual} vs #{expected}"
      end

      # Properties py-fsrs's own test suite asserts.
      assert parameters != ExFsrs.Scheduler.new().parameters

      assert Optimizer.batch_loss(ctx.sequences, parameters) <
               Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      for {{value, lower}, upper} <-
            Enum.zip(Enum.zip(parameters, Optimizer.lower_bounds()), Optimizer.upper_bounds()) do
        assert value >= lower
        assert value <= upper
      end

      # The result must be usable as scheduler parameters.
      scheduler = ExFsrs.Scheduler.new(parameters: parameters, enable_fuzzing: false)

      {reviewed, _log} =
        ExFsrs.Scheduler.review_card(
          scheduler,
          ExFsrs.new(state: :learning, step: 0),
          :good,
          ~U[2024-01-01 09:00:00Z]
        )

      assert reviewed.stability > 0
      assert reviewed.difficulty >= 1.0 and reviewed.difficulty <= 10.0
    end
  end
end
