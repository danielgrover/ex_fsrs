# Needs an Nx carrying the `while` f64 gradient fix (bench/NX_WHILE_GRAD_F64.md):
# this script trains with `model: :loop` for speed, and stock Nx computes wrong
# gradients through `while`. Point EX_FSRS_NX_PATH at a patched checkout before
# `mix deps.get`; on stock Nx the optimizer refuses `:loop` rather than train on
# bad gradients, and the script stops with that error.
#
# Compares optimizer variants on held-out cards across every fetched collection.
#
# Regularization trades training fit for generalization, so it can only be
# judged on data the optimizer did not see. Every 5th card is held out; the rest
# train. Collections are processed smallest-first so partial results arrive
# early, and each row prints as it completes.
#
#     MIX_ENV=test mix run bench/holdout_eval.exs
alias ExFsrs.AnkiDataset
alias ExFsrs.Optimizer
alias ExFsrs.Optimizer.Data

base = [model: :loop, compiler: EXLA]

variants = [
  {"py-fsrs", []},
  {"+init", [initialize: true]},
  {"+init+L2", [initialize: true, regularization: 1.0]}
]

split = fn sequences ->
  {train, test} =
    sequences
    |> Enum.with_index()
    |> Enum.split_with(fn {_sequence, index} -> rem(index, 5) != 0 end)

  {Enum.map(train, &elem(&1, 0)), Enum.map(test, &elem(&1, 0))}
end

header =
  "  user | train scored | " <>
    Enum.map_join(variants, " | ", &String.pad_leading(elem(&1, 0), 9)) <> " | best"

IO.puts("HELD-OUT loss, 20% of cards never trained on\n")
IO.puts(header)
IO.puts(String.duplicate("-", String.length(header)))

rows =
  AnkiDataset.summary()
  |> Enum.sort_by(& &1.scored)
  |> Enum.flat_map(fn %{user_id: id} ->
    {train, test} = split.(AnkiDataset.sequences(id))
    train_scored = Data.num_reviews(train)

    if train_scored < 512 do
      []
    else
      losses =
        Enum.map(variants, fn {_name, opts} ->
          parameters = Optimizer.compute_optimal_parameters(train, opts ++ base)
          Optimizer.batch_loss(test, parameters)
        end)

      {_loss, best} =
        losses |> Enum.zip(Enum.map(variants, &elem(&1, 0))) |> Enum.min_by(&elem(&1, 0))

      IO.puts(
        "#{String.pad_leading("#{id}", 6)} | #{String.pad_leading("#{train_scored}", 12)} | " <>
          Enum.map_join(
            losses,
            " | ",
            &String.pad_leading(Float.to_string(Float.round(&1, 6)), 9)
          ) <>
          " | #{best}"
      )

      [losses]
    end
  end)

count = length(rows)
means = rows |> Enum.zip() |> Enum.map(fn t -> Enum.sum(Tuple.to_list(t)) / count end)

IO.puts("\n#{count} collections")
IO.puts("mean held-out loss:")

for {{name, _opts}, mean} <- Enum.zip(variants, means) do
  IO.puts("  #{String.pad_trailing(name, 10)} #{Float.round(mean, 6)}")
end

wins =
  Enum.map(Enum.with_index(variants), fn {{name, _}, i} ->
    {name, Enum.count(rows, fn r -> Enum.min(r) == Enum.at(r, i) end)}
  end)

IO.puts("\nbest on:  " <> Enum.map_join(wins, "   ", fn {n, c} -> "#{n}=#{c}/#{count}" end))

[_p, init_mean, l2_mean] = means

IO.puts(
  "\nL2 vs init alone: #{Float.round((init_mean - l2_mean) / init_mean * 100, 3)}% (positive = L2 better)"
)

IO.puts("L2 beat init alone on #{Enum.count(rows, fn [_p, i, l] -> l < i end)}/#{count}")
