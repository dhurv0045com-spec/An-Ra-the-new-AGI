"""One-step engineering fixture; not an AGI or generalization benchmark."""
import hashlib


def make_fixture(data):
    excluded = {tuple(sorted(r['canonical_pair']))
                for key in ('train', 'dev_controller', 'dev_measurement', 'sealed_reserved')
                for r in data[key]}
    pairs = [(a, b) for a in range(10, 80) for b in range(a + 1, 80)
             if a + b <= 99 and (a, b) not in excluded]
    pairs.sort(key=lambda p: hashlib.sha256(f'one-step-8121/{p[0]}/{p[1]}'.encode()).hexdigest())
    def rows(selected, role):
        return [dict(world_id=f'one-step/{role}/{a}/{b}', a=a, b=b,
                     canonical_pair=[a, b], prompt=f'{a} + {b} = ', answer=str(a + b))
                for a, b in selected]
    if len(pairs) < 164:
        raise ValueError('insufficient disjoint pairs')
    return dict(train=rows(pairs[:64], 'train'), holdout=rows(pairs[64:164], 'holdout'))


def exact_flags(predicted, expected):
    if len(predicted) != len(expected) or not expected:
        raise ValueError('empty or unequal row counts')
    return [p == e for p, e in zip(predicted, expected)]
