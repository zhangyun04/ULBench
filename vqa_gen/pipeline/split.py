import random


def build_class_level_splits(wnids, ratios, seed):
    wnids_unique = sorted(set(wnids))
    rng = random.Random(seed)
    rng.shuffle(wnids_unique)

    total = len(wnids_unique)
    forget_n = int(total * ratios["forget"])
    test_n = int(total * ratios["test"])
    retain_n = total - forget_n - test_n
    if retain_n < 0:
        raise ValueError("Invalid split ratios: retain becomes negative.")

    forget_set = set(wnids_unique[:forget_n])
    retain_set = set(wnids_unique[forget_n : forget_n + retain_n])
    test_set = set(wnids_unique[forget_n + retain_n :])

    mapping = {}
    for wnid in wnids_unique:
        if wnid in forget_set:
            mapping[wnid] = "forget"
        elif wnid in retain_set:
            mapping[wnid] = "retain"
        elif wnid in test_set:
            mapping[wnid] = "test"
        else:
            raise RuntimeError(f"Split assignment missing for class: {wnid}")
    return mapping
