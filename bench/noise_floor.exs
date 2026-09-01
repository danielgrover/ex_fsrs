# Needs an Nx carrying the `while` f64 gradient fix (bench/NX_WHILE_GRAD_F64.md):
# this script trains with `model: :loop` for speed, and stock Nx computes wrong
# gradients through `while`. Point EX_FSRS_NX_PATH at a patched checkout before
# `mix deps.get`; on stock Nx the optimizer refuses `:loop` rather than train on
# bad gradients, and the script stops with that error.
#
# How much does the same configuration vary between runs?
#
# The optimizer shuffles card order per epoch, so two runs of identical settings
# land on different parameters. If that spread is comparable to the differences
# between variants, then those differences are not measurable at this sample
# size — and that is a fact about the experiment, not about the variants.
#
#     MIX_ENV=test mix run bench/noise_floor.exs
alias ExFsrs.AnkiDataset
alias ExFsrs.Optimizer
alias ExFsrs.Optimizer.Data

base = [model: :loop, compiler: EXLA, initialize: true, regularization: 1.0]
repeats = 5

IO.puts("Same configuration, #{repeats} runs each, temporal split\n")

IO.puts(
  "  user | " <>
    Enum.map_join(1..repeats, " | ", &String.pad_leading("run #{&1}", 8)) <> " |   spread"
)

spreads =
  AnkiDataset.summary()
  |> Enum.sort_by(& &1.scored)
  |> Enum.take(8)
  |> Enum.flat_map(fn %{user_id: id} ->
    {train, test} = AnkiDataset.temporal_split(id)

    if Data.num_reviews(train) < 512 do
      []
    else
      losses =
        for _ <- 1..repeats do
          Optimizer.batch_loss(test, Optimizer.compute_optimal_parameters(train, base))
        end

      spread = (Enum.max(losses) - Enum.min(losses)) / Enum.min(losses) * 100

      IO.puts(
        "#{String.pad_leading("#{id}", 6)} | " <>
          Enum.map_join(
            losses,
            " | ",
            &String.pad_leading(Float.to_string(Float.round(&1, 6)), 8)
          ) <>
          " | #{String.pad_leading(Float.to_string(Float.round(spread, 3)), 7)}%"
      )

      [spread]
    end
  end)

IO.puts("\nrun-to-run spread on identical settings:")
IO.puts("  median #{Float.round(Enum.at(Enum.sort(spreads), div(length(spreads), 2)), 3)}%")
IO.puts("  max    #{Float.round(Enum.max(spreads), 3)}%")
IO.puts("\nFor comparison, the largest difference between any two variants in")
IO.puts("bench/temporal_eval.exs is about 2%, so a spread this size must be pinned")
IO.puts("(via :card_orders) before variants can be compared.")
