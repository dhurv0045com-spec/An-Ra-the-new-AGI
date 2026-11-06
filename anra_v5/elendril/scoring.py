"""Score committed token streams without importing the learner or PyTorch."""

from collections import defaultdict
from collections.abc import Iterable

from .benchmark import Example
from .codec import BOS, EOS, PAD, Prediction, VOCAB_SIZE, answer_tokens


def score(rows: tuple[Example, ...], predictions: Iterable[Prediction]) -> dict:
    outputs = {}
    for prediction in predictions:
        if prediction.task_id in outputs:
            raise ValueError("duplicate prediction ID")
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE
               for token in prediction.tokens):
            raise ValueError("prediction contains an invalid token ID")
        outputs[prediction.task_id] = prediction.tokens
    expected_ids = {row.visible.task_id for row in rows}
    if not rows or len(expected_ids) != len(rows) or outputs.keys() != expected_ids:
        raise ValueError("prediction IDs must match the complete scoring fixture")

    clusters = defaultdict(list)
    query_swaps = defaultdict(list)
    graph_pair_clusters = {}
    positions = []
    exact = terminated = prefix_total = correct_content = total_content = 0
    for row in rows:
        task = row.visible
        if row.truth.task_id != task.task_id:
            raise ValueError("truth ID does not match the visible task")
        actual = outputs[task.task_id]
        gold = answer_tokens(row.truth.gold)
        valid_eos = (len(actual) >= 2 and actual[-1] == EOS
                     and not set(actual[:-1]) & {EOS, PAD, BOS})
        correct = valid_eos and actual == gold
        exact += correct
        terminated += valid_eos
        clusters[task.cluster_id].append(correct)
        if task.family == "binding" or (task.family == "abstention" and task.prompt[-2] in "XY"):
            key = task.prompt.split("?")[0]
            query_swaps[key].append(correct)
            if task.difficulty == "graph-v3":
                if graph_pair_clusters.setdefault(key, task.cluster_id) != task.cluster_id:
                    raise ValueError("query pair crosses graph topology clusters")
        content = actual[:actual.index(EOS)] if EOS in actual else actual
        prefix = 0
        for index, token in enumerate(gold[:-1]):
            if index == len(positions):
                positions.append([0, 0])
            hit = index < len(content) and content[index] == token
            correct_content += hit
            total_content += 1
            positions[index][0] += hit
            positions[index][1] += 1
            if hit and prefix == index:
                prefix += 1
        prefix_total += prefix

    if any(len(pair) != 2 for pair in query_swaps.values()):
        raise ValueError("query swap fixture must provide exactly two counterfactual views")
    result = {
        "query_swap_both_correct": [sum(all(pair) for pair in query_swaps.values()), len(query_swaps)],
        "exact_eos": [exact, len(rows)],
        "valid_eos": [terminated, len(rows)],
        "content_token_accuracy": [correct_content, total_content],
        "lcp": [prefix_total, total_content],
        "positions": positions,
        "all_views_correct": [sum(all(values) for values in clusters.values()), len(clusters)],
        "clusters": {key: [sum(values), len(values)] for key, values in sorted(clusters.items())},
    }
    if graph_pair_clusters:
        cluster_pairs = defaultdict(list)
        for key, cluster in graph_pair_clusters.items():
            cluster_pairs[cluster].append(all(query_swaps[key]))
        result["cluster_query_pairs"] = {
            key: [sum(values), len(values)] for key, values in sorted(cluster_pairs.items())
        }
    return result
