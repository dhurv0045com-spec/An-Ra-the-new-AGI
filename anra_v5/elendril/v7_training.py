"""Deterministic, matched-multiset V7 sampler plans over V6 training."""
from collections import defaultdict
import math
import random

from .codec import digest
from .v6_training import V6Trainer

METHODS = ('uniform', 'blocked', 'interleaved')


def ordered_indices(samples, seed, epoch, method='uniform'):
    """Return a permutation; interleaving follows each category's share.

    A seeded shuffle fixes the within-category order and tie order. At each
    position the category with the lowest next normalized rank is selected.
    Blocked reproduces V6's uncertainty-first first epoch, then uniform order.
    """
    if method not in METHODS or type(epoch) is not int or epoch < 0:
        raise ValueError('unknown method or invalid epoch')
    order = list(range(len(samples)))
    random.Random(int(digest([seed, epoch, 'foundation-order-v1']), 16)).shuffle(order)
    if method == 'blocked' and epoch == 0:
        order.sort(key=lambda i: samples[i].category not in ('missing', 'conflict'))
    elif method == 'interleaved':
        groups = defaultdict(list)
        for i in order:
            groups[samples[i].category].append(i)
        keys = list(groups)
        used = {key: 0 for key in keys}
        result = []
        # Compare ranks as integers, avoiding floating-point schedule drift.
        while len(result) < len(order):
            chosen = None
            for key in keys:
                if used[key] == len(groups[key]):
                    continue
                if chosen is None or (used[key] + 1) * len(groups[chosen]) < (used[chosen] + 1) * len(groups[key]):
                    chosen = key
            result.append(groups[chosen][used[chosen]])
            used[chosen] += 1
        order = result
    return tuple(order)


class V7Trainer(V6Trainer):
    """V6 updates with a source-bound, finite V7 ordering plan."""
    def __init__(self, *args, method='uniform', **kwargs):
        if method not in METHODS:
            raise ValueError('unknown V7 sampling method')
        if kwargs['recipe'].sampling != 'uniform':
            raise ValueError('V7 sampling requires a uniform recipe')
        provenance = dict(kwargs['data_provenance'])
        if provenance.get('v7_sampling_method', method) != method:
            raise ValueError('V7 sampling provenance mismatch')
        provenance['v7_sampling_method'] = method
        kwargs['data_provenance'] = provenance
        self.method = method
        super().__init__(*args, **kwargs)

    def indices(self, update):
        per_epoch = math.ceil(len(self.samples) / self.recipe.batch_size)
        epoch, batch = divmod(update, per_epoch)
        if epoch not in self._epoch_orders:
            self._epoch_orders[epoch] = ordered_indices(self.samples, self.recipe.seed, epoch, self.method)
        start = batch * self.recipe.batch_size
        return self._epoch_orders[epoch][start:start + self.recipe.batch_size]
