"""Periodic recovery saves, plus initial and final recovery states.

An improving FID also triggers a current-state save before best.pt is linked.
Evaluation and preview schedules are independent of this interval.
"""


def checkpoint_due(update, total, interval):
    if interval < 1:
        raise ValueError('Checkpoint interval must be positive')
    return update == 3 or update % interval == 0 or update == total
