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

# The two configurations worth resolving against the baseline: the one with the
# best mean in the earlier sweep, and the one that won on the most collections.
variants = [
  {"py-fsrs", []},
  {"best", [initialize: true, regularization: 1.0, recency: true]},
  {"best+outliers", [initialize: true, regularization: 1.0, recency: true, outliers: true]}
]

# Runtime scales with scored reviews, and a handful of very large collections
# would otherwise dominate the wall clock while contributing one data point each
# — the same weight as a collection a fiftieth their size.
max_train_scored =
  case System.get_env("MAX_TRAIN_SCORED") do
    nil -> 25_000
    value -> String.to_integer(value)
  end

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

    if train_scored < 512 or test_scored < 50 or train_scored > max_train_scored do
      []
    else
      losses =
        Enum.map(variants, fn {_name, opts} ->
          {recency, opts} = Keyword.pop(opts, :recency, false)
          {outliers, opts} = Keyword.pop(opts, :outliers, false)

          # Outliers are removed before weighting, so the recency ranking runs
          # over the reviews that will actually be scored.
          set = if outliers, do: Data.remove_outliers(train), else: train
          set = if recency, do: AnkiDataset.recency_weighted(set, id), else: set

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

[plain, l2g2, rec] = means

# Three aggregations, because they answer different questions and disagree. The
# ratio of totals is dominated by the collections with the largest losses; the
# mean of per-collection ratios weights every collection equally and is pulled
# by a few big winners; the median says what a typical collection gets.
ratios = fn index ->
  Enum.map(rows, fn row ->
    (Enum.at(row, 0) - Enum.at(row, index)) / Enum.at(row, 0) * 100
  end)
end

report = fn name, index, mean ->
  values = ratios.(index)
  sorted = Enum.sort(values)
  median = Enum.at(sorted, div(length(sorted), 2))
  improved = Enum.count(values, &(&1 > 0))

  IO.puts(
    "  #{String.pad_trailing(name, 16)} " <>
      "total #{Float.round((plain - mean) / plain * 100, 3)}%   " <>
      "per-collection mean #{Float.round(Enum.sum(values) / length(values), 3)}%   " <>
      "median #{Float.round(median, 3)}%   " <>
      "improved #{improved}/#{count}"
  )
end

IO.puts("\nrelative to py-fsrs (positive = better):")
report.("best", 1, l2g2)
report.("best+outliers", 2, rec)
