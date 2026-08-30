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


def get(url, auth, with_headers=False):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {auth}"})
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            return (body, response.headers) if with_headers else body
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            sys.exit(
                f"{error.code} from {url}\n\n"
                "The token was rejected. Confirm you accepted the dataset terms at\n"
                f"  https://huggingface.co/datasets/{DATASET}\n"
                "while logged in as the account that owns this token."
            )
        raise


def next_page(headers):
    """The API pages at 1000 entries and points at the next with a Link header."""
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None

    for part in link.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")

    return None


def revlog_files(auth):
    """Every user's revlog path with its size, smallest first.

    The tree is listed recursively, which interleaves directories with files and
    pages at 1000 entries, so every page has to be walked.
    """
    import json

    url = f"{API}/tree/main/revlogs?recursive=true"
    files = []
    pages = 0

    while url:
        body, headers = get(url, auth, with_headers=True)
        pages += 1

        for entry in json.loads(body):
            if entry.get("type") != "file" or not entry["path"].endswith(".parquet"):
                continue

            size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
            files.append((entry["path"], size))

        url = next_page(headers)
        print(f"\r  {pages} pages, {len(files)} files", end="", flush=True)

    print()

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
    parser.add_argument("--users", type=int, default=3, help="how many users to fetch")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="skip this many of the smallest users first. The smallest collections "
        "have too few scored reviews to train at all (the optimizer returns the "
        "defaults under 512), so skip ahead for something that exercises training.",
    )
    parser.add_argument(
        "--until",
        type=int,
        help="stop considering users at this rank (default: all of them)",
    )
    parser.add_argument(
        "--spread",
        action="store_true",
        help="pick users evenly spaced across the considered range rather than "
        "consecutively, so the sample covers a range of collection sizes",
    )
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
        ranked = revlog_files(auth)
        until = args.until if args.until is not None else len(ranked)
        window = ranked[args.skip : until]

        if args.spread and len(window) > args.users:
            step = len(window) / args.users
            window = [window[int(i * step)] for i in range(args.users)]
        else:
            window = window[: args.users]

        paths = [path for path, _size in window]

    total = sum(write_user(path, auth, out_dir) for path in paths)
    print(f"\n{len(paths)} users, {total} reviews -> {out_dir}")


main()
