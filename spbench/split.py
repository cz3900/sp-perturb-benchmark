import numpy as np


def split_by_perturbation(data, unseen: list, seed: int = 0) -> dict:
    """Split by PERTURBATION (not random cells). Unlabeled + control cells always go to
    train (spatial context). Perturbed cells of `unseen` perturbations go to test only;
    the rest of perturbed cells are split per-cell into seen-train / seen-test."""
    rng = np.random.default_rng(seed)
    unseen = set(unseen)
    is_unseen_pert = np.array([p in unseen for p in data.perturbation])
    seen_pert = data.is_perturbed & ~is_unseen_pert

    seen_idx = np.where(seen_pert)[0]
    rng.shuffle(seen_idx)
    cut = int(0.75 * len(seen_idx))
    seen_train, seen_test = seen_idx[:cut], seen_idx[cut:]

    ctx = np.where(data.is_control | data.is_unlabeled)[0]   # always train
    train_idx = np.sort(np.concatenate([ctx, seen_train]))
    test_idx = np.sort(np.concatenate([np.where(is_unseen_pert)[0], seen_test]))
    return {"train": data.subset(train_idx), "test": data.subset(test_idx),
            "test_seen": seen_test, "test_unseen": np.where(is_unseen_pert)[0]}
