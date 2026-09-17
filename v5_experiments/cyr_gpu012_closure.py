"""Full-exposure compact closure; independent paired commutation diagnostics."""
from __future__ import annotations

from v5_experiments import cyr_gpu011 as base


def make_battery(data):
    battery = base.make_reasoning_battery(data)
    excluded = {tuple(r['canonical_pair'])
                for key in ('train', 'dev_controller', 'dev_measurement', 'sealed_reserved')
                for r in data[key]}
    for name, bands in (('COMMUTATION_MATCHED_BAND', (1, 2, 3, 4)),
                        ('COMMUTATION_OOD_BAND', (6, 7))):
        rows = []
        for band in bands:
            for a in range(10 * band, 10 * band + 10):
                for b in range(a + 1, 10 * band + 10):
                    if a % 10 + b % 10 > 9 or (a, b) in excluded:
                        continue
                    pair = f'{name}/{a}/{b}'
                    for role, x, y in (('base', a, b), ('reversed', b, a)):
                        rows.append(dict(world_id=f'{pair}/{role}', pair_id=pair,
                                         role=role, a=x, b=y, band=band,
                                         canonical_pair=[a, b], answer=str(a + b),
                                         prompt=f'{x} + {y} = '))
        battery[name] = rows
    battery['COMMUTED'] = [{**r, 'a': r['b'], 'b': r['a']}
                           for r in battery['COMMUTED']]
    battery['_manifest'] = [dict(name=k, count=len(v), sha256=base.stable_sha(v))
                            for k, v in battery.items() if k != '_manifest']
    return battery


def score_pairs(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row['pair_id'], {})
        if row['role'] in group:
            raise ValueError('duplicate pair role')
        group[row['role']] = row
    if not groups or any(set(g) != {'base', 'reversed'} for g in groups.values()):
        raise ValueError('empty or incomplete pair evidence')
    content = valid = correct = 0
    for g in groups.values():
        a, b = g['base'], g['reversed']
        same = a['generated'] == b['generated']
        eos = a['stop'] == b['stop'] == 'EOS'
        content += int(same)
        valid += int(same and eos)
        correct += int(eos and a['generated'] == a['expected']
                       and b['generated'] == b['expected'])
    return dict(pairs=len(groups), content_consistency=content / len(groups),
                eos_valid_consistency=valid / len(groups),
                both_exact_with_eos=correct / len(groups))


def decide(acquisition):
    if acquisition.get('updates') != 18000 or acquisition.get('row_presentations') != 1152000:
        return 'INCOMPLETE_EXPOSURE'
    trace = [r for r in acquisition['trace'] if 'dev_controller' in r]
    final = acquisition['reasoning_battery_final']['STANDARD']['complete_exact_with_valid_stop']
    sustained = len(trace) >= 3 and all(
        r['dev_controller']['complete_exact_with_valid_stop'] >= .9 for r in trace[-3:])
    if sustained and final >= .9:
        return 'COMPACT_G90_FULL_EXPOSURE'
    if sustained or final >= .9 or acquisition.get('g90_confirm_update') is not None:
        return 'AMBIGUOUS_PARTIAL_OR_TRANSIENT_G90'
    return 'NO_G90_AT_FULL_EXPOSURE'
