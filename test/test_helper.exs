# Tests tagged :dataset need collections fetched from the gated Anki Revlogs 10K
# dataset, which is not committed. Run them with:
#
#     mix test --include dataset
#
# after fetching data with bench/fetch_anki_revlogs.py.
ExUnit.start(exclude: [:dataset])
