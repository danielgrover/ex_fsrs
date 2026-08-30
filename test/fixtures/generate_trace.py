"""
Reproduces py-fsrs's Optimizer.compute_optimal_parameters, instrumented to
record everything our Elixir port needs to check itself step by step.

The body below is a transcription of fsrs/optimizer.py's
compute_optimal_parameters with only recording statements added. Its nested
_update_parameters is a closure, so it cannot be monkeypatched from outside --
hence the copy. The assertion at the end (final params == py-fsrs's published
test_optimal_parameters) is what proves the transcription did not drift.
"""

import json
import math
import sys
from datetime import datetime
from random import Random

import pandas as pd
import torch
from torch import optim
from torch.nn import BCELoss

from fsrs.card import Card
from fsrs.optimizer import Optimizer, learning_rate, max_seq_len, mini_batch_size, num_epochs
from fsrs.review_log import Rating, ReviewLog
from fsrs.scheduler import (
    DEFAULT_PARAMETERS,
    LOWER_BOUNDS_PARAMETERS,
    UPPER_BOUNDS_PARAMETERS,
    Scheduler,
)

CSV = sys.argv[1]
OUT = sys.argv[2]

# py-fsrs's published expectation, from tests/test_optimizer.py
EXPECTED = [
    0.12340357383516173, 1.2931, 2.397673571899466, 8.2956, 6.686820427099132,
    0.45021679958387956, 3.077875127553957, 0.053520395733247045,
    1.6539992229052127, 0.1466206769107436, 0.6300772488850335,
    1.611965002575047, 0.012840136810798864, 0.34853762746216305,
    1.8878958285806287, 0.8546376191171063, 1.8729, 0.6748536823468675,
    0.20451266082721842, 0.22622814695113844, 0.46030603398979064,
]


def get_revlogs():
    df = pd.read_csv(CSV)
    return [
        ReviewLog(
            card_id=row["card_id"],
            rating=Rating(row["review_rating"]),
            review_datetime=datetime.fromisoformat(row["review_time"]),
            review_duration=row["review_duration"],
        )
        for _, row in df.iterrows()
    ]


def main():
    opt = Optimizer(get_revlogs())

    lower = torch.tensor(LOWER_BOUNDS_PARAMETERS, dtype=torch.float64)
    upper = torch.tensor(UPPER_BOUNDS_PARAMETERS, dtype=torch.float64)

    trace = {
        "source": "py-fsrs Optimizer.compute_optimal_parameters",
        "torch_version": torch.__version__,
        "hyperparameters": {
            "num_epochs": num_epochs,
            "mini_batch_size": mini_batch_size,
            "learning_rate": learning_rate,
            "max_seq_len": max_seq_len,
        },
        "default_parameters": list(DEFAULT_PARAMETERS),
        "lower_bounds": list(LOWER_BOUNDS_PARAMETERS),
        "upper_bounds": list(UPPER_BOUNDS_PARAMETERS),
        "epochs": [],
    }

    # Deterministic: no shuffle, no Adam. The easiest target to hit first.
    trace["batch_loss_at_defaults"] = opt._compute_batch_loss(
        parameters=list(DEFAULT_PARAMETERS)
    )
    trace["batch_loss_at_expected"] = opt._compute_batch_loss(parameters=EXPECTED)

    # --- transcription of compute_optimal_parameters, with recording ---
    rng = Random(42)
    card_ids = list(opt._revlogs_train.keys())

    num_reviews = 0
    probe = Scheduler()
    for card_id in card_ids:
        history = opt._revlogs_train[card_id][:max_seq_len]
        for i, review in enumerate(history):
            review_datetime, rating = review[0][0], review[0][1]
            if i == 0:
                card = Card(card_id=card_id, due=review_datetime)
            if card.last_review and (review_datetime - card.last_review).days > 0:
                num_reviews += 1
            card, _ = probe.review_card(card, rating, review_datetime, None)

    trace["num_reviews"] = num_reviews
    assert num_reviews >= mini_batch_size

    params = torch.tensor(DEFAULT_PARAMETERS, requires_grad=True, dtype=torch.float64)
    loss_fn = BCELoss()
    adam = optim.Adam([params], lr=learning_rate)
    t_max = math.ceil(num_reviews / mini_batch_size) * num_epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=adam, T_max=t_max)
    trace["t_max"] = t_max

    steps = []

    def update(step_losses):
        mini_batch_loss = torch.sum(torch.stack(step_losses))
        adam.zero_grad()
        mini_batch_loss.backward()
        grad = params.grad.detach().clone().tolist()
        adam.step()
        with torch.no_grad():
            params.clamp_(min=lower, max=upper)
        lr_scheduler.step()
        steps.append(
            {
                "n_losses": len(step_losses),
                "loss_sum": mini_batch_loss.item(),
                "lr": adam.param_groups[0]["lr"],
                "grad": grad,
                "params_after": params.detach().clone().tolist(),
            }
        )

    best_params, best_loss = None, math.inf

    for epoch in range(num_epochs):
        rng.shuffle(card_ids)
        epoch_record = {"card_order": [int(c) for c in card_ids], "first_step": len(steps)}

        scheduler = Scheduler(parameters=params)
        step_losses = []

        for card_id in card_ids:
            history = opt._revlogs_train[card_id][:max_seq_len]
            for i, review in enumerate(history):
                x_date, u_rating = review[0][0], review[0][1]
                y = review[1]
                if i == 0:
                    card = Card(card_id=card_id, due=x_date)

                y_pred = scheduler.get_card_retrievability(card, current_datetime=x_date)
                y_t = torch.tensor(y, dtype=torch.float64)

                if card.last_review and (x_date - card.last_review).days > 0:
                    step_losses.append(loss_fn(y_pred, y_t))

                card, _ = scheduler.review_card(card, u_rating, x_date, None)

                if len(step_losses) == mini_batch_size:
                    update(step_losses)
                    scheduler = Scheduler(parameters=params)
                    step_losses = []
                    card.stability = card.stability.detach()
                    card.difficulty = card.difficulty.detach()

        if len(step_losses) > 0:
            update(step_losses)

        detached = [x.detach().item() for x in list(params.detach())]
        with torch.no_grad():
            epoch_loss = opt._compute_batch_loss(parameters=detached)

        epoch_record["last_step"] = len(steps) - 1
        epoch_record["batch_loss"] = epoch_loss
        epoch_record["params_after"] = detached
        trace["epochs"].append(epoch_record)

        if epoch_loss < best_loss:
            best_loss, best_params = epoch_loss, detached

        print(f"epoch {epoch}: batch_loss={epoch_loss:.12f} steps={len(steps)}", flush=True)

    trace["steps"] = steps
    trace["best_loss"] = best_loss
    trace["final_parameters"] = best_params

    # Proves the transcription above still is py-fsrs's algorithm.
    max_diff = max(abs(a - b) for a, b in zip(best_params, EXPECTED))
    trace["max_abs_diff_vs_published"] = max_diff
    print(f"\nmax abs diff vs published test_optimal_parameters: {max_diff:.3e}")
    assert max_diff < 1e-8, f"transcription drifted from py-fsrs: {max_diff}"

    with open(OUT, "w") as f:
        json.dump(trace, f)
    print(f"wrote {OUT} ({len(steps)} gradient steps)")


main()
