"""
Fetches per-user review histories from the Anki Revlogs 10K dataset.

The dataset is one parquet file per user (revlogs/user_id=N/data.parquet), so a
single collection can be pulled without touching the other 727 million reviews.
Users are picked smallest-first by file size, because a full optimization run
scales with review count and the point of this is to have something that
finishes in seconds rather than hours.

Writes one CSV per user into test/fixtures/anki/ (gitignored — the dataset is
licensed separately and is not ours to redistribute).

    export HF_TOKEN=hf_...
    python bench/fetch_anki_revlogs.py --users 5
    python bench/fetch_anki_revlogs.py --user-ids 1,42,1337

The dataset is gated: log in at
https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k,
accept the terms (access is granted automatically), then create a read token at
https://huggingface.co/settings/tokens.
"""

import argparse
import io
import os
import sys
import urllib.error
import urllib.request

DATASET = "open-spaced-repetition/anki-revlogs-10k"
API = f"https://huggingface.co/api/datasets/{DATASET}"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "fixtures", "anki")


def token():
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not value:
        sys.exit(
            "No HF_TOKEN in the environment.\n\n"
            f"The dataset is gated. Accept the terms at\n"
            f"  https://huggingface.co/datasets/{DATASET}\n"
            "then create a read token at https://huggingface.co/settings/tokens and\n"
            "  export HF_TOKEN=hf_..."
        )
    return value


def get(url, auth):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {auth}"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            sys.exit(
                f"{error.code} from {url}\n\n"
                "The token was rejected. Confirm you accepted the dataset terms at\n"
                f"  https://huggingface.co/datasets/{DATASET}\n"
                "while logged in as the account that owns this token."
            )
        raise


def revlog_files(auth):
    """Every user's revlog path with its size, smallest first."""
    import json

    tree = json.loads(get(f"{API}/tree/main/revlogs?recursive=true", auth))
    files = [
        (entry["path"], entry.get("size") or entry.get("lfs", {}).get("size") or 0)
        for entry in tree
        if entry["type"] == "file" and entry["path"].endswith(".parquet")
    ]

    return sorted(files, key=lambda pair: pair[1])


def user_id_of(path):
    return int(path.split("user_id=")[1].split("/")[0])


def write_user(path, auth, out_dir):
    import pyarrow.parquet as pq

    user = user_id_of(path)
    table = pq.read_table(io.BytesIO(get(f"{RESOLVE}/{path}", auth)))
    columns = table.column_names

    card_ids = table.column("card_id").to_pylist()
    ratings = table.column("rating").to_pylist()
    elapsed = table.column("elapsed_days").to_pylist()
    states = table.column("state").to_pylist() if "state" in columns else [None] * len(card_ids)

    out = os.path.join(out_dir, f"user_{user}.csv")
    with open(out, "w") as f:
        f.write("card_id,rating,elapsed_days,state\n")
        for card_id, rating, days, state in zip(card_ids, ratings, elapsed, states):
            f.write(f"{card_id},{rating},{days},{'' if state is None else state}\n")

    cards = len(set(card_ids))
    scored = sum(1 for days in elapsed if days is not None and days > 0)
    print(f"user {user:>5}: {len(card_ids):>7} reviews  {cards:>6} cards  {scored:>7} scored -> {os.path.basename(out)}")

    return len(card_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=3, help="how many users to fetch, smallest first")
    parser.add_argument("--user-ids", help="comma-separated user ids, overriding --users")
    args = parser.parse_args()

    auth = token()
    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    if args.user_ids:
        wanted = [int(u) for u in args.user_ids.split(",")]
        paths = [f"revlogs/user_id={u}/data.parquet" for u in wanted]
    else:
        print("listing users by size...")
        paths = [path for path, _size in revlog_files(auth)[: args.users]]

    total = sum(write_user(path, auth, out_dir) for path in paths)
    print(f"\n{len(paths)} users, {total} reviews -> {out_dir}")


main()
