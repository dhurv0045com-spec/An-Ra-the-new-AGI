"""Y5 repairs for information acquisition and elementary persistent memory.

P6's existing embedding, GRU writer, read head and action head are reused.
A query token is a compact encoding of the two publicly supplied laws; it is
not the hidden truth. All labels and utilities are generated independently.
The P6 heads are linear: an XOR of law with hypothesis or menu order with
semantic action cannot be represented by additive embeddings. We therefore
learn a canonical hypothesis/action distribution and apply the explicitly
public law/menu permutation to its logits. This is a minimal symbolic interface
repair, not evidence the network learned the public mappings.
"""
from itertools import product
import random
import torch
import torch.nn.functional as F
from experiments.avendi_y4.models import P6LearnedMemoryPolicy


def jobs():
    return [dict(id=f'{stage}-{arm}-s{seed}', stage=stage, arm=arm, seed=seed,
                 depends=[], updates=400) for stage, arms in
            [('D', ('informative', 'noninformative')), ('E', ('relevant', 'sham'))]
            for arm in arms for seed in (7, 19, 41)]


def _styles(split, parity):
    # Every primitive appears during training; held out tuples are recombinations.
    candidates = [(a,b) for a,b in product(range(2), repeat=2)
                  if (a ^ b ^ parity) == (0 if split == 'train' else 1)]
    return candidates if split == 'train' else candidates[:1] if split == 'dev' else candidates[1:]


def worlds(split='train'):
    rows = []
    for ci, cost in enumerate((.05, .35)):
        for order in (0, 1):
            for style_a, style_b in _styles(split, ci ^ order):
                public = (ci, 2+order, 4+style_a, 6+style_b)
                for truth in (0, 1):
                    rows.append(dict(cost=cost, public=public, truth=truth,
                        menu=[0, 1] if order == 0 else [1, 0],
                        pair_id=f'{split}-c{ci}-m{order}-r{style_a}{style_b}', render_split=split))
    return rows


def solve(cost, reliability):
    """Bayes-optimal expected reward, action, and posterior decision rule."""
    direct = 0.5
    tool = max(reliability, 1 - reliability) - cost
    return dict(action=int(tool > direct), direct=direct, tool=tool,
                prediction_if_observed=[int(reliability < .5), int(reliability >= .5)])


def d_items(arm, split='train'):
    # Balanced exact channel frequencies; identical sample count for each arm.
    reliability = {'informative': .75, 'noninformative': .5, 'perfect': 1.0}[arm]
    rows = []
    for w in worlds(split):
        for obs in (0, 1):
            copies = ((3 if obs == w['truth'] else 1) if reliability == .75 else
                      (4 if obs == w['truth'] else 0) if reliability == 1.0 else 2)
            for _ in range(copies):
                action = solve(w['cost'], reliability)['action']
                rows.append(dict(**w, obs=obs, action=action, action_position=w['menu'].index(action)))
    return rows


def e_items(arm, split='train'):
    rows = []
    for law in (0, 1):
        for style_a, style_b in _styles(split, law):
            for truth, sham in product(range(2), repeat=2):
                rows.append(dict(public=(law, 4+style_a, 6+style_b), law=law,
                                 truth=truth, feedback=truth if arm == 'relevant' else sham,
                                 gold=truth ^ law))
    return rows


def _forward(model, public, sup_x=None, sup_y=None):
    # Same Y4 embedding, GRU and heads; add embeddings of public primitives.
    q = model.in_emb(public).sum(dim=1)
    if sup_x is not None:
        pair = torch.cat([model.in_emb(sup_x), model.obs_emb(sup_y)], dim=-1)
        mem = model.writer_gru(pair, model.mem_state.expand(pair.shape[0], -1))
    else:
        mem = model.mem_state.expand(q.shape[0], -1)
    return model.read_head(torch.cat([q, mem], dim=-1)), model.action_head(q)


def _menu_logits(canonical, menu):
    """Map canonical action scores [answer, tool] to presented positions."""
    return canonical.gather(1, menu)


def _law_logits(hypothesis, public):
    """Map learned hypothesis 0/1 scores through the public query truth table."""
    flipped = hypothesis[:, [1, 0, 2, 3]]
    return torch.where((public[:, 0] == 1).unsqueeze(1), flipped, hypothesis)


def _loss_d(model, items):
    device = next(model.parameters()).device
    p = torch.tensor([x['public'] for x in items], device=device)
    o = torch.tensor([x['obs'] for x in items], device=device)
    truth = torch.tensor([x['truth'] for x in items], device=device)
    action = torch.tensor([x['action_position'] for x in items], device=device)
    menu = torch.tensor([x['menu'] for x in items], device=device)
    # Every sample receives terminal-answer supervision, including no-tool arms.
    logits, acts =  _forward(model, p, torch.full_like(o, 10), o)
    direct, _ = _forward(model, p)
    return F.cross_entropy(logits, truth) + F.cross_entropy(direct, truth) + F.cross_entropy(_menu_logits(acts, menu), action)


def _loss_e(model, items):
    device = next(model.parameters()).device
    p = torch.tensor([x['public'] for x in items], device=device)
    fb = torch.tensor([x['feedback'] for x in items], device=device)
    gold = torch.tensor([x['gold'] for x in items], device=device)
    logits, _ = _forward(model, p, torch.full_like(fb, 10), fb)
    return F.cross_entropy(_law_logits(logits, p), gold)


class MemorySession:
    """Actual ephemeral context plus explicit persistent P6 state operations."""
    def __init__(self, model):
        self.model = model
        self.context = []
        self.model.reset_episode()

    def observe(self, feedback, diagnostic=True):
        self.context.append((int(feedback), bool(diagnostic)))

    def write(self):
        if self.context:
            d = next(self.model.parameters()).device
            value, diagnostic = self.context[-1]
            self.model.write_support(torch.tensor([10 if diagnostic else 11], device=d),
                                     torch.tensor([value], device=d))

    def clear_context(self):
        self.context.clear()

    def erase(self):
        self.model.ablate_memory('erase')

    def edit(self, feedback):
        # Editing replaces, rather than appending to, the persistent record.
        self.erase()
        self.observe(feedback)
        self.write()
        self.clear_context()

    def read(self, public):
        d = next(self.model.parameters()).device
        logits, _ = _forward(self.model, torch.tensor([public], device=d))
        return int(_law_logits(logits, torch.tensor([public], device=d)).argmax(-1).item())


def _eval_d(model, arm, split):
    items = d_items(arm, split)
    correct_action = correct_terminal = 0
    utility = 0.0
    responses = []
    for index, r in enumerate(items):
        d = next(model.parameters()).device
        with torch.no_grad():
            p = torch.tensor([r['public']], device=d)
            logits, act = _forward(model, p, torch.tensor([10], device=d),
                                   torch.tensor([r['obs']], device=d))
            position = int(_menu_logits(act, torch.tensor([r['menu']], device=d)).argmax(-1).item())
            a = r['menu'][position]
            direct, _ = _forward(model, p)
            pred = int((logits if a else direct).argmax(-1).item())
        correct_action += a == r['action']
        correct_terminal += int(logits.argmax(-1).item()) == r['truth']
        utility += float(pred == r['truth']) - a * r['cost']
        responses.append(dict(pair_id=r['pair_id'], public=r['public'], menu=r['menu'], action_position=position, gold_action_position=r['action_position'], cost=r['cost'], truth=r['truth'], observation=r['obs'], action=a, gold_action=r['action'], terminal_prediction=pred, post_tool_prediction=int(logits.argmax(-1).item())))
    invariant = True
    with torch.no_grad():
        for w in worlds(split):
            p = torch.tensor([w['public']], device=d)
            acts = [int(_forward(model, p, torch.tensor([10], device=d), torch.tensor([o], device=d))[1].argmax(-1).item()) for o in (0, 1)]
            invariant = invariant and acts[0] == acts[1]
    n = len(items)
    oracle = sum(solve(r['cost'], {'informative': .75, 'noninformative': .5, 'perfect': 1.0}[arm])['tool'] if r['action'] else .5 for r in items) / n
    return dict(action_accuracy=correct_action/n, action_correct=correct_action, post_tool_accuracy=correct_terminal/n, post_tool_correct=correct_terminal,
                realized_utility=utility/n, total_utility=utility, oracle_expected_utility=oracle, regret=oracle-utility/n, action_invariant_to_hidden_observation=invariant, samples=n, responses=responses)


def _eval_e(model, split):
    sums = {k: 0 for k in ('relevant', 'none', 'irrelevant', 'incorrect', 'erased', 'edited')}
    changes = {k: 0 for k in ('write', 'erase', 'edit')}
    n = 0
    responses = []
    with torch.no_grad():
        eval_rows = [x for x in e_items('relevant', split) if x['feedback'] == x['truth']]
        # e_items duplicates relevant examples for balanced arm exposure.
        seen = set()
        for row in eval_rows:
            p, law, truth = row['public'], row['law'], row['truth']
            key = (p, truth)
            if key in seen: continue
            seen.add(key)
            gold = law ^ truth
            session = MemorySession(model)
            baseline = session.read(p)
            sums['none'] += baseline == gold
            # Shared-law feedback has no discriminatory value; don't write it
            # as though it were the diagnostic observation.
            session.observe(2, diagnostic=False); session.write(); session.clear_context()
            irrelevant = session.read(p)
            sums['irrelevant'] += irrelevant == gold
            session.erase()
            session.observe(truth); session.write(); session.clear_context()
            assert not session.context
            relevant = session.read(p)
            sums['relevant'] += relevant == gold
            changes['write'] += relevant != baseline
            session.erase(); erased = session.read(p)
            sums['erased'] += erased == gold
            changes['erase'] += erased != relevant
            session.edit(1-truth); edited = session.read(p)
            sums['incorrect'] += edited == gold
            sums['edited'] += edited == (law ^ (1-truth))
            changes['edit'] += edited != relevant
            responses.append(dict(public_tokens=p, public_laws={'diagnostic': [0, 1], 'query': [law, 1-law]}, pair_id=f'{split}-law{law}', truth=truth, gold=gold, no_feedback=baseline, irrelevant=irrelevant, relevant=relevant, erased=erased, edited=edited))
            n += 1
    return {'responses': responses, 'accuracy': {k: v/n for k, v in sums.items()},
            'behavior_change': {k: v/n for k, v in changes.items()}, 'samples': n}


def run(job, ctx):
    random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = P6LearnedMemoryPolicy(width=32, seed=job['seed']).to(ctx.device)
    items = d_items(job['arm']) if job['stage'] == 'D' else e_items(job['arm'])
    # Explicitly preserve equal exposure across experimental arms.
    updates = int(job['updates']) if not ctx.canary else min(2, int(job['updates']))
    rng = random.Random(job['seed'] + 810)
    batches = [[rng.randrange(len(items)) for _ in range(min(16, len(items)))] for _ in range(updates)]
    loss = _loss_d if job['stage'] == 'D' else _loss_e
    training = ctx.train(model, items, batches, loss, lr=0.003)
    metrics = (_eval_d(model, job['arm'], ctx.split) if job['stage'] == 'D' else _eval_e(model, ctx.split))
    familiar = (_eval_d(model, job['arm'], 'train') if job['stage'] == 'D' else _eval_e(model, 'train'))
    perfect = _eval_d(model, 'perfect', ctx.split) if job['stage'] == 'D' else None
    return dict(id=job['id'], stage=job['stage'], arm=job['arm'], seed=job['seed'],
                split=ctx.split, engineering_canary=bool(ctx.canary), updates_requested=updates, examples_per_epoch=len(items),
                training=training, metrics=metrics, familiar_metrics=familiar, perfect_channel_evaluation=perfect,
                rendering_note='Held-out combinations reuse trained primitive embeddings; known public law and menu are applied as output permutations. This does not test learning or understanding those mappings.')


def preflight():
    paired = worlds()
    def check_pairs(ws):
        return all({x['truth'] for x in ws if x['public'] == p} == {0, 1}
                   and len({x['cost'] for x in ws if x['public'] == p}) == 1
                   for p in {x['public'] for x in ws})
    def sham_independent(rows):
        return all(sum(x['truth'] == t and x['feedback'] == f for x in rows if x['public'] == p) == 1
                   for p in {x['public'] for x in rows} for t, f in product(range(2), repeat=2))
    sham = e_items('sham')
    corrupted = [dict(x, feedback=x['truth']) for x in sham]
    leaked = [dict(x, public=(*x['public'], x['truth'])) for x in paired]
    tests = {'paired_public': check_pairs(paired),
             'public_leak_detected': not check_pairs(leaked),
             'sham_balanced': sham_independent(sham),
             'sham_leak_detected': not sham_independent(corrupted),
             'menu_orders_balanced': all(sum(w['menu'].index(1) == position for w in paired) == len(paired)//2 for position in (0, 1)),
             'disjoint_renderings': not (set(w['public'] for w in worlds('dev')) & set(w['public'] for w in worlds('confirm'))),
             'solver_cost': solve(.05,.75)['action'] == 1 and solve(.35,.75)['action'] == 0,
             'perfect_channel_resolves_truth': all(x['obs'] == x['truth'] for x in d_items('perfect')),
             'noninformative_never_useful': solve(.05,.5)['action'] == 0,
             'matched_terminal_supervision': len(d_items('informative')) == len(d_items('noninformative')),
             'matched_exposure': len(d_items('informative')) == len(d_items('noninformative')) and len(e_items('relevant')) == len(sham)}
    return {'passed': all(tests.values()), 'checks': tests}
