defmodule ExFsrs.AnkiDatasetTest do
  @moduledoc """
  Exercises the optimizer against real collections from the Anki Revlogs 10K
  dataset, when they have been fetched locally.

  The dataset is gated and separately licensed, so it is not committed. These
  tests skip unless `bench/fetch_anki_revlogs.py` has been run.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.AnkiDataset
  alias ExFsrs.Optimizer
  alias ExFsrs.Optimizer.Data

  @moduletag :dataset

  setup_all do
    if AnkiDataset.available?() do
      {:ok, users: AnkiDataset.users()}
    else
      {:ok, users: []}
    end
  end

  test "loads real collections into optimizer sequences", ctx do
    if ctx.users == [] do
      IO.puts("\n  no Anki data fetched; run bench/fetch_anki_revlogs.py")
    else
      for user_id <- ctx.users do
        sequences = AnkiDataset.sequences(user_id)

        assert match?([_ | _], sequences)

        # A card's first review has no elapsed time and cannot be scored.
        for {_card_id, [first | _rest]} <- sequences do
          assert first.elapsed_days == -1
          refute first.counts_for_loss?
        end

        assert Data.num_reviews(sequences) > 0
      end
    end
  end

  @tag timeout: 1_800_000
  test "optimized parameters beat the defaults on a real collection", ctx do
    if ctx.users == [] do
      IO.puts("\n  no Anki data fetched; run bench/fetch_anki_revlogs.py")
    else
      user_id = hd(ctx.users)
      sequences = AnkiDataset.sequences(user_id)
      defaults = ExFsrs.Scheduler.new().parameters

      parameters = Optimizer.compute_optimal_parameters(sequences)

      if parameters == defaults do
        # Under 512 scored reviews the optimizer returns the defaults by design.
        assert Data.num_reviews(sequences) < 512
      else
        assert Optimizer.batch_loss(sequences, parameters) <
                 Optimizer.batch_loss(sequences, defaults)

        for {{value, lower}, upper} <-
              Enum.zip(Enum.zip(parameters, Optimizer.lower_bounds()), Optimizer.upper_bounds()) do
          assert value >= lower and value <= upper
        end
      end
    end
  end
end
