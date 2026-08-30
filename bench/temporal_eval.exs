# Compares optimizer variants under a temporal split: train on a collection's
# past, score its future.
#
# This is the split that matches what a scheduler actually does, and the one
# srs-benchmark uses. It differs from a card holdout in an important way — a
# card usually has reviews on both sides of the cut, so the test set replays
# each card's whole history and scores only the post-cut reviews.
#
#     MIX_ENV=test mix run bench/temporal_eval.exs
alias ExFsrs.AnkiDataset
alias ExFsrs.Optimizer
alias ExFsrs.Optimizer.Data

base = [model: :loop, compiler: EXLA]

variants = [
  {"py-fsrs", []},
  {"+init", [initialize: true]},
  {"+init+L2", [initialize: true, regularization: 1.0]}
]

scored_count = fn sequences ->
  Enum.reduce(sequences, 0, fn {_id, reviews}, total ->
    total + Enum.count(reviews, & &1.counts_for_loss?)
  end)
end

header =
  "  user | train | test  | " <>
    Enum.map_join(variants, " | ", &String.pad_leading(elem(&1, 0), 9)) <> " | best"

IO.puts("TEMPORAL split: train on the first 80% of the timeline, score the rest\n")
IO.puts(header)
IO.puts(String.duplicate("-", String.length(header)))

rows =
  AnkiDataset.summary()
  |> Enum.sort_by(& &1.scored)
  |> Enum.flat_map(fn %{user_id: id} ->
    {train, test} = AnkiDataset.temporal_split(id)
    train_scored = Data.num_reviews(train)
    test_scored = scored_count.(test)

    if train_scored < 512 or test_scored < 50 do
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
        "#{String.pad_leading("#{id}", 6)} | #{String.pad_leading("#{train_scored}", 5)} | " <>
          "#{String.pad_leading("#{test_scored}", 5)} | " <>
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

IO.puts("\n#{count} collections. Mean loss on the future:")

for {{name, _opts}, mean} <- Enum.zip(variants, means) do
  wins =
    Enum.count(rows, fn r ->
      Enum.min(r) == Enum.at(r, Enum.find_index(variants, &(elem(&1, 0) == name)))
    end)

  IO.puts("  #{String.pad_trailing(name, 10)} #{Float.round(mean, 6)}   best on #{wins}/#{count}")
end

[plain, init, l2] = means
IO.puts("\ninit vs py-fsrs: #{Float.round((plain - init) / plain * 100, 3)}%")
IO.puts("L2 vs init:      #{Float.round((init - l2) / init * 100, 3)}%  (positive = better)")
IO.puts("L2 beat init on #{Enum.count(rows, fn [_p, i, l] -> l < i end)}/#{count}")
