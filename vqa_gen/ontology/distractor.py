import random

from vqa_gen.ontology.normalize import normalize_choice_text


def _sample_unique_wnids(candidates, all_classes, used_names, k, rng):
    sampled = []
    pool = list(candidates)
    rng.shuffle(pool)

    for wnid in pool:
        if wnid not in all_classes:
            continue
        class_name = all_classes[wnid]
        key = normalize_choice_text(class_name)
        if key in used_names:
            continue
        sampled.append(wnid)
        used_names.add(key)
        if len(sampled) == k:
            break
    return sampled


def sample_distractors(gt_wnid, hard_pool, all_classes, hard_k, easy_k, seed):
    rng = random.Random(seed)

    gt_name_key = normalize_choice_text(all_classes[gt_wnid])
    used_names = {gt_name_key}

    hard_candidates = [wnid for wnid in hard_pool if wnid != gt_wnid]
    hard_sampled = _sample_unique_wnids(
        hard_candidates, all_classes, used_names, hard_k, rng
    )

    if len(hard_sampled) < hard_k:
        fallback_candidates = [
            wnid for wnid in all_classes.keys() if wnid not in set(hard_sampled) and wnid != gt_wnid
        ]
        hard_sampled.extend(
            _sample_unique_wnids(
                fallback_candidates,
                all_classes,
                used_names,
                hard_k - len(hard_sampled),
                rng,
            )
        )

    easy_candidates = [
        wnid for wnid in all_classes.keys() if wnid not in set(hard_sampled) and wnid != gt_wnid
    ]
    easy_sampled = _sample_unique_wnids(
        easy_candidates, all_classes, used_names, easy_k, rng
    )

    if len(easy_sampled) < easy_k:
        extra_candidates = [
            wnid for wnid in all_classes.keys() if wnid not in set(hard_sampled + easy_sampled) and wnid != gt_wnid
        ]
        easy_sampled.extend(
            _sample_unique_wnids(
                extra_candidates, all_classes, used_names, easy_k - len(easy_sampled), rng
            )
        )

    sampled_wnids = hard_sampled + easy_sampled
    return [all_classes[wnid] for wnid in sampled_wnids]
