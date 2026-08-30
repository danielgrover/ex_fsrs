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

# The optimizer shuffles card order each epoch, and that shuffle alone moves
# held-out loss by ~2% run to run — as much as the largest difference between
# any two variants. Pinning one ordering per collection and giving it to every
# variant removes that noise from the comparison entirely, so what is left is
# the variant.
fixed_orders = fn sequences ->
  card_ids = Enum.map(sequences, fn {card_id, _reviews} -> card_id end)

  {orders, _seed} =
    Enum.map_reduce(1..5, :rand.seed_s(:exsss, {17, 23, 42}), fn _epoch, seed ->
      Enum.reduce(card_ids, {[], seed}, fn _card, {shuffled, seed} ->
        {shuffled, seed}
      end)
      |> then(fn {_, seed} ->
        {shuffled, seed} =
          Enum.reduce(card_ids, {[], seed}, fn card, {acc, seed} ->
            {position, seed} = :rand.uniform_s(length(card_ids), seed)
            {[{position, card} | acc], seed}
          end)

        {shuffled |> Enum.sort() |> Enum.map(&elem(&1, 1)), seed}
      end)
    end)

  orders
end

# Every upgrade, plus a gamma sweep, under the split that matters.
variants = [
  {"py-fsrs", []},
  {"+init", [initialize: true]},
  {"L2 γ1", [regularization: 1.0]},
  {"init+L2γ1", [initialize: true, regularization: 1.0]},
  {"init+L2γ2", [initialize: true, regularization: 2.0]},
  {"init+L2+rec", [initialize: true, regularization: 1.0, recency: true]}
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
      weighted = AnkiDataset.recency_weighted(train, id)

      losses =
        Enum.map(variants, fn {_name, opts} ->
          {recency, opts} = Keyword.pop(opts, :recency, false)
          set = if recency, do: weighted, else: train

          parameters =
            Optimizer.compute_optimal_parameters(
              set,
              [card_orders: fixed_orders.(train)] ++ opts ++ base
            )

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

[plain, init, l2_only, l2, l2g2, rec] = means

IO.puts("\nrelative to py-fsrs (positive = better):")
IO.puts("  +init            #{Float.round((plain - init) / plain * 100, 3)}%")
IO.puts("  L2 alone         #{Float.round((plain - l2_only) / plain * 100, 3)}%")
IO.puts("  init+L2 γ=1      #{Float.round((plain - l2) / plain * 100, 3)}%")
IO.puts("  init+L2 γ=2      #{Float.round((plain - l2g2) / plain * 100, 3)}%")
IO.puts("  init+L2+recency  #{Float.round((plain - rec) / plain * 100, 3)}%")

IO.puts("\nhead-to-head win counts:")

IO.puts(
  "  L2 γ=1 beat init alone on #{Enum.count(rows, fn r -> Enum.at(r, 3) < Enum.at(r, 1) end)}/#{count}"
)

IO.puts(
  "  γ=2 beat γ=1 on           #{Enum.count(rows, fn r -> Enum.at(r, 4) < Enum.at(r, 3) end)}/#{count}"
)

IO.puts(
  "  recency beat no recency on #{Enum.count(rows, fn r -> Enum.at(r, 5) < Enum.at(r, 3) end)}/#{count}"
)
