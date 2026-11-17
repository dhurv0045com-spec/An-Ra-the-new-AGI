"""CPU query/bridge crossover design pilot. No learned model is evaluated."""
from collections import Counter
import json
import random
from string import ascii_uppercase


def traverse(rows, start):
    edges = {subject: value for subject, value, _ in rows}
    for _ in range(3):
        start = edges[start]
    return start


def run(seed=1801, families=256):
    rng = random.Random(seed)
    case_hits = Counter()
    family_hits = Counter()
    for _ in range(families):
        a, b, c, d, e, f, g, h = rng.sample(ascii_uppercase[:16], 8)
        order = rng.sample(range(6), 6)
        ticks = rng.sample(range(1000, 9000), 6)
        counts = Counter()
        for swapped in (False, True):
            edges = [(a,c), (b,d), (c,f if swapped else e), (d,e if swapped else f), (e,g), (f,h)]
            rows = [(*edges[i], ticks[i]) for i in order]
            for query, answer in ((a, h if swapped else g), (b, g if swapped else h)):
                guesses = {'traversal': traverse(rows, query), 'last_object': rows[-1][1],
                           'third_timestamp': sorted(rows, key=lambda row: row[2])[2][1],
                           'ignore_bridge_oracle_elsewhere': g if query == a else h,
                           'ignore_query_oracle_elsewhere': h if swapped else g}
                for policy, guess in guesses.items():
                    counts[policy] += int(guess == answer)
        for policy, count in counts.items():
            case_hits[policy] += count
            family_hits[policy] += int(count == 4)
    return dict(seed=seed, families=families, cases=4*families, correct=dict(case_hits),
                all_four_correct=dict(family_hits))


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
