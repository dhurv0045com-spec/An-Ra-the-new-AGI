"""Bounded, offline audit of the exact first owner V8 data-control archive.

No neural inference, training, unpickling or archive extraction. Raw owner data
remains external; only curated JSON is emitted. A new run requires a new identity.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256, file_digest
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from zipfile import ZipFile

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8.teacher import PublicFact, teacher_decide
from anra_v5.elendril.v8.verifier import independent_expected_outcome, verify_episode_trace
from anra_v5.elendril.v8.data import build_d1_training_dataset
from anra_v5.elendril.v8.holdout import canonical_graph_signature

SOURCE = 'cf043942511f0b14c343445e0c3fe7570bd7c63b'
ARCHIVE_SHA = 'f5d28b8759d0252e9c5e487e97cf1a817765def5a0c1523f7e2caeec097e7ef5'
MODES = ('raw', 'grammar', 'availability-mask', 'direct-answer-only')
COHORTS = ('R09', 'R10', 'E13', 'E14', 'E15', 'A19')
STAGES = ('initial', 'epoch-1', 'epoch-2', 'epoch-4')
ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    require(isinstance(a, (int, float)) and math.isfinite(a) and math.isclose(a, b, abs_tol=1e-10),
            f'measurement differs: {a!r} != {b!r}')


def audit_episode(row):
    """Resolve truth and visible teacher actions; replay tool/fact transitions."""
    world = {k: row[k] for k in ('goal_subject', 'hops', 'initial_facts', 'allowed_public_tools')}
    world['hidden'] = {'expected': row['expected_answer'], 'query_payload': row['query_payload'],
                       'memory_payload': row['memory_payload']}
    require(independent_expected_outcome(world) == row['expected_answer'], 'incorrect stored semantic answer')
    facts = list(dict.fromkeys(PublicFact(**f) for f in row['initial_facts']))
    qa = row['allowed_public_tools']['query_available']
    ma = row['allowed_public_tools']['memory_available']
    actions, observations, first_div = [], [], None
    contract_issues = Counter()
    for idx, step in enumerate(row['steps']):
        text = step['public_text_utf8']
        require(sha256(text.encode()).hexdigest() == step['public_text_sha256'], 'public text hash differs')
        flags = re.search(r'Query=([01]) Memory=([01])', text)
        require(flags is not None, 'missing public availability')
        shown_qa, shown_ma = (bool(int(x)) for x in flags.groups())
        if (shown_qa, shown_ma) != (qa, ma):
            contract_issues['stale_availability_steps'] += 1
        # This exact archive uses tuple renderer2, with canonical or dev- aliases.
        parsed = [PublicFact(subject=s.removeprefix('dev-'), source=src, tick=int(t),
                             value=None if v == 'null' else v.removeprefix('dev-'))
                  for src,t,s,v in re.findall(r'\(([^,;()]+),(\d+),([^,;()]+),([^,;()]+)\)', text)]
        require(Counter(parsed) == Counter(facts), 'public facts differ from observed world state')
        contract_ideal = teacher_decide(tuple(facts), row['goal_subject'], row['hops'], qa, ma)
        ideal = teacher_decide(tuple(facts), row['goal_subject'], row['hops'], shown_qa, shown_ma)
        if ideal != contract_ideal:
            contract_issues['teacher_target_changed_by_stale_availability'] += 1
        require(ideal == (step['ideal_operation'], step['ideal_argument']), 'recorded teacher action differs')
        proposal, raw = step['executed_proposal'], step['raw_proposal']
        action = (proposal['operation'], proposal['argument'])
        semantic_action = (action[0], action[1] if action[0] == 'ANSWER' else None)
        raw_action = (raw['operation'], raw['argument'])
        if row['mode'] == 'raw':
            require(action == raw_action, 'raw action was transformed')
        if first_div is None and (semantic_action != ideal or action != raw_action):
            first_div = idx
        payload = []
        if action[0] == 'QUERY' and shown_qa:
            if not qa: contract_issues['repeated_consumed_tool_calls'] += 1
            payload = row['query_payload']; qa = False
        elif action[0] == 'RETRIEVE' and shown_ma:
            if not ma: contract_issues['repeated_consumed_tool_calls'] += 1
            payload = row['memory_payload']; ma = False
        require(step['tool_responses'] == payload, 'tool response differs')
        for item in payload:
            fact = PublicFact(**item)
            if fact not in facts:
                facts.append(fact)
        actions.append(action)
        observations.append({'ideal': ideal, 'contract_ideal':contract_ideal,'action': semantic_action})
    outcome = verify_episode_trace(world, actions)
    require(outcome.correct == row['correct'] == row['exact_match'], 'episode score differs')
    require(first_div == row['first_divergence_step'], 'first divergence differs')
    return observations, dict(contract_issues)


def rate(rows):
    return {'correct': sum(r['correct'] for r in rows), 'n': len(rows),
            'accuracy': sum(r['correct'] for r in rows) / len(rows) if rows else None}


def compact(row):
    out = {k: row[k] for k in ('world_id', 'base_world_id', 'cohort', 'form', 'correct',
                               'predicted_answer', 'expected_answer', 'first_divergence_step')}
    last = row['steps'][-1]['executed_proposal']
    out['final_action'] = (last['operation'], last['argument'])
    return out


def analyze(path):
    with path.open('rb') as f:
        actual_sha = file_digest(f, 'sha256').hexdigest()
    require(actual_sha == ARCHIVE_SHA, 'archive is not this registered owner run')
    # Imported semantic code must be exactly the immutable runtime implementation.
    for rel in ('anra_v5/elendril/v8/teacher.py', 'anra_v5/elendril/v8/verifier.py',
                'anra_v5/elendril/v8/data.py', 'anra_v5/elendril/v8/holdout.py'):
        pinned = subprocess.check_output(['git', 'show', f'{SOURCE}:{rel}'], cwd=ROOT)
        require((ROOT / rel).read_text(encoding='utf-8') == pinned.decode().replace('\r\n', '\n'),
                f'semantic replay source drift: {rel}')
    result = {'archive_sha256': actual_sha, 'source_commit': SOURCE, 'phase': 'data-control',
              'cause': 'UNRESOLVED', 'runs': [], 'limitations': [
                  'Analysis-only custody: no FULL archive or neural checkpoint tensors verified.',
                  'Recorded-action replay does not regenerate probabilities or prove neural checkpoint recovery.',
                  'D0 and D1 change corpus, sampler, rendering and exposure; this is a recipe comparison.',
                  'All six models are B0/S20; J, F, P, F-cap, retention and transfer were not run.',
                  'Milestone training diagnostics are recorded aggregates, not independently regenerated inference.']}
    with ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'duplicate ZIP members')
        require(sum(i.file_size for i in z.infolist()) < 512*1024*1024, 'archive size exceeds audit bound')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name, 'unsafe archive path')
        manifest = json.loads(z.read('ARCHIVE_MANIFEST.json'))
        require(manifest['source_commit'] == SOURCE, 'wrong manifest source')
        require(digest(manifest['files']) == manifest['manifest_sha256'], 'manifest digest differs')
        require({f['path'] for f in manifest['files']} | {'ARCHIVE_MANIFEST.json'} == set(names), 'manifest inventory differs')
        for entry in manifest['files']:
            with z.open(entry['path']) as f:
                h, size = sha256(), 0
                while block := f.read(1024*1024):
                    h.update(block); size += len(block)
            require(size == entry['bytes'] and h.hexdigest() == entry['sha256'], 'member checksum differs')
        status = json.loads(z.read('FINAL_STATUS.json'))
        require(status['source_commit'] == SOURCE and status['phase'] == 'data-control', 'wrong campaign')
        expected_jobs = {f'V8-B0-{d}-i{s}-o{s}' for d in ('D0', 'D1') for s in (7, 19, 41)}
        require(set(status['completed_jobs']) == set(status['expected_jobs']) == expected_jobs, 'job registry differs')
        result['reported_status'] = status['status']
        result['manifest_sha256'] = manifest['manifest_sha256']
        result['verified_archive_files'] = len(names)
        result['split_audit'] = json.loads(z.read('splits/SPLIT_AUDIT.json'))
        split = result['split_audit']
        require(digest({k:v for k,v in split.items() if k!='split_audit_sha256'}) == split['split_audit_sha256'], 'split receipt digest differs')
        train = build_d1_training_dataset()
        require(train.identity == split['d1_identity'], 'regenerated training identity differs')
        world_identities = {}
        diagnostics = json.loads(z.read('DIAGNOSTICS.json'))
        result['canaries'] = diagnostics['canaries']
        result['profiles'] = diagnostics['profiles']
        result['forecast_not_measured_session_cost'] = diagnostics['forecast']
        final_by_job, common_keys, all_episode_count = {}, None, 0
        all_contract_issues = Counter()
        for job_id in sorted(expected_jobs):
            job = json.loads(z.read(f'jobs/{job_id}.json'))
            expected_updates = 2368 if job['dataset'] == 'D0' else 1472
            require(job['completed_updates'] == job['target_updates'] == expected_updates, 'update count differs')
            records, first_actions, confusions, divergences = defaultdict(list), Counter(), Counter(), Counter()
            stage_keys, finals = set(), {}
            after_query = Counter(); answer_hist = Counter(); trace_examples = []
            raw_contract_issues = Counter()
            first_by_cohort = defaultdict(Counter)
            with z.open(f'episodes/{job_id}.jsonl') as f:
                for line in f:
                    row = json.loads(line)
                    world = {k:row[k] for k in ('goal_subject','hops','initial_facts','allowed_public_tools')}
                    world['hidden'] = {'expected':row['expected_answer'],'query_payload':row['query_payload'],
                                       'memory_payload':row['memory_payload']}
                    wid = row['world_id']; identity = digest(world)
                    if wid not in world_identities:
                        signature = canonical_graph_signature(world)
                        require(signature['structure_hash'] == row['structure_hash'], 'stored structure hash differs')
                        require(signature['is_isolated_one_edge_probe'] or signature['structure_hash'] not in train.structural_hashes,
                                'development overlaps regenerated training structure')
                        world_identities[wid] = identity
                    require(world_identities[wid] == identity, 'same evaluation case changed between records')
                    observations, contract_issues = audit_episode(row)
                    all_contract_issues.update(contract_issues)
                    if contract_issues: all_contract_issues['affected_episode_records'] += 1
                    key = (row['milestone'], row['mode'], row['world_id'])
                    require(key not in stage_keys, 'duplicate episode key'); stage_keys.add(key)
                    require(row['job_id'] == job_id, 'wrong episode job')
                    milestone = job['milestones'][row['milestone']]
                    require(row['checkpoint_sha256'] == milestone['checkpoint_sha256'], 'episode checkpoint differs')
                    require(row['milestone_global_update'] == milestone['global_update'], 'episode update differs')
                    records[(row['milestone'], row['mode'])].append(compact(row))
                    if row['milestone'] == 'epoch-4' and row['mode'] == 'raw':
                        first = observations[0]
                        first_stats = first_by_cohort[row['cohort']]
                        first_stats['n'] += 1
                        first_stats['teacher_action_match'] += int(first['action'] == first['contract_ideal'])
                        first_stats[f"{first['contract_ideal'][0]}->{first['action'][0]}"] += 1
                        if first['contract_ideal'][0] == first['action'][0] == 'ANSWER':
                            first_stats['wrong_answer_value'] += int(first['contract_ideal'][1] != first['action'][1])
                        raw_contract_issues.update(contract_issues)
                        if contract_issues: raw_contract_issues['affected_episode_records'] += 1
                        finals[row['world_id']] = compact(row)
                        first_actions[observations[0]['action'][0]] += 1
                        for o in observations:
                            confusions[f"{o['ideal'][0]}->{o['action'][0]}"] += 1
                        divergences[str(row['first_divergence_step'])] += 1
                        if row['predicted_answer'] is not None: answer_hist[row['predicted_answer']] += 1
                        for idx,o in enumerate(observations[:-1]):
                            if o['action'][0] == 'QUERY' and row['steps'][idx]['tool_responses']:
                                after_query['observed_query_transitions'] += 1
                                after_query['transitions_in_eventually_correct_episodes'] += int(row['correct'])
                                after_query['next_action_matches_contract_teacher'] += int(observations[idx+1]['action'] == observations[idx+1]['contract_ideal'])
                        if len(trace_examples) < 2 and not row['correct'] and row['cohort'] in ('R09','E14'):
                            trace_examples.append({'world_id':row['world_id'], 'cohort':row['cohort'], 'form':row['form'],
                                                   'expected':row['expected_answer'], 'observations':observations})
            expected_counts = {('initial','raw'):3840, ('epoch-1','raw'):960, ('epoch-2','raw'):960,
                               **{('epoch-4', mode):3840 for mode in MODES}}
            require({k:len(v) for k,v in records.items()} == expected_counts, 'milestone/mode coverage differs')
            if common_keys is None: common_keys = stage_keys
            require(stage_keys == common_keys, 'case identities differ between jobs')
            all_episode_count += len(stage_keys)
            modes = {}
            for mode in MODES:
                rows = records[('epoch-4',mode)]
                require(Counter(r['cohort'] for r in rows) == Counter({c:640 for c in COHORTS}), 'cohort counts differ')
                modes[mode] = rate(rows)
                close(job['summary_by_mode'][mode]['exact_episode_accuracy'], modes[mode]['accuracy'])
            milestones = {}
            final_subset = {r['world_id'] for r in records[('epoch-1','raw')]}
            for stage in STAGES:
                m = job['milestones'][stage]; rows = records[(stage,'raw')]
                close(m['suite_summary']['exact_episode_accuracy'],rate(rows)['accuracy'])
                p = m['probe']; close(p['exact_action_accuracy'],p['exact_action_count']/p['total_steps'])
                milestones[stage] = {'update':m['global_update'], 'teacher_state_accuracy':p['exact_action_accuracy'],
                                     'teacher_states':p['total_steps'], 'teacher_loss':p['teacher_loss'],
                                     'raw_development':rate(rows), 'same_960_record_subset':rate([r for r in rows if r['world_id'] in final_subset])}
            step_rows = [json.loads(line) for line in z.read(f'steps/{job_id}.jsonl').splitlines()]
            require([r['global_update'] for r in step_rows] == list(range(1,expected_updates+1)), 'step history incomplete')
            require(all(math.isfinite(r[k]) for r in step_rows for k in ('loss','lr','grad_norm')), 'nonfinite training log')
            final_by_job[job_id] = finals
            raw_rows = records[('epoch-4','raw')]
            pairs = defaultdict(dict)
            for row in raw_rows: pairs[row['base_world_id']][row['form']] = row
            require(len(pairs) == 768 and all(len(p)==5 for p in pairs.values()), 'linked variant coverage differs')
            rel_both = sum(p['canonical']['correct'] and p['relevant_cf']['correct'] for p in pairs.values())
            irr_stable = sum(p['canonical']['final_action'] == p['irrelevant_cf']['final_action']
                             and p['canonical']['correct'] == p['irrelevant_cf']['correct'] for p in pairs.values())
            close(job['relevant_cf_both_correct'],rel_both/len(pairs))
            close(job['irrelevant_cf_stability'],irr_stable/len(pairs))
            run = {k:job[k] for k in ('job_id','dataset','init_seed','order_seed','checkpoint_sha256','completed_updates','wall_seconds')}
            run.update({'modes':modes,'raw_by_cohort':{c:rate([r for r in raw_rows if r['cohort']==c]) for c in COHORTS},
                        'raw_by_form':{c:rate([r for r in raw_rows if r['form']==c]) for c in sorted({r['form'] for r in raw_rows})},
                        'first_actions':dict(first_actions),'step_operation_confusions':dict(confusions),
                        'first_divergence':dict(divergences),'answer_histogram':dict(answer_hist),
                        'after_query_observational':dict(after_query),'trace_examples':trace_examples,
                        'raw_final_environment_contract_violations':dict(raw_contract_issues),
                        'first_decision_by_cohort':{k:dict(v) for k,v in first_by_cohort.items()},
                        'paired_counterfactual_counts':{'base_worlds':len(pairs),'relevant_both_correct':rel_both,'irrelevant_same_final_action_and_outcome':irr_stable},
                        'milestones':milestones,'final_probe_recorded':job['final_probe'],
                        'step_history':{'records':len(step_rows),'first_utc':step_rows[0]['timestamp_utc'],'last_utc':step_rows[-1]['timestamp_utc'],
                                        'loss_first':step_rows[0]['loss'],'loss_last':step_rows[-1]['loss']},
                        'recorded_relevant_cf_both_correct':job['relevant_cf_both_correct'],
                        'recorded_irrelevant_cf_stability':job['irrelevant_cf_stability']})
            result['runs'].append(run)
        result['independently_replayed_episodes'] = all_episode_count
        result['regenerated_split_check'] = {'training_identity':train.identity,'training_worlds':len(train.worlds),
                                            'distinct_development_records':len(world_identities),'prohibited_structural_overlaps':0}
        result['environment_contract_violations_all_modes_and_milestones'] = dict(all_contract_issues)
        result['paired_seed_effects'] = []
        for seed in (7,19,41):
            a,b = (final_by_job[f'V8-B0-{d}-i{seed}-o{seed}'] for d in ('D1','D0'))
            counts=Counter(('both' if a[k]['correct'] and b[k]['correct'] else 'd1_only' if a[k]['correct'] else 'd0_only' if b[k]['correct'] else 'neither') for k in a)
            result['paired_seed_effects'].append({'seed':seed,'n':len(a),'outcomes':dict(counts),
                                                 'd1_minus_d0':(counts['d1_only']-counts['d0_only'])/len(a)})
        result['dataset_means'] = {}
        for d in ('D0','D1'):
            runs = [r for r in result['runs'] if r['dataset']==d]
            result['dataset_means'][d] = {'raw_accuracy':sum(r['modes']['raw']['accuracy'] for r in runs)/3,
                'teacher_state_accuracy':sum(r['final_probe_recorded']['exact_action_accuracy'] for r in runs)/3,
                'one_hop_accuracy':sum(r['raw_by_cohort']['R09']['accuracy'] for r in runs)/3,
                'reported_job_wall_seconds_sum':sum(r['wall_seconds'] for r in runs),
                'update_count_sum':sum(r['completed_updates'] for r in runs)}
    result['audit_status'] = 'PASS_RECORDED_EVIDENCE'
    result['scientific_instrument_status'] = 'DEFECT_CONFIRMED_REEVALUATION_REQUIRED' if all_contract_issues else 'NO_CONTRACT_VIOLATIONS_FOUND'
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    metrics=analyze(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'audit_status':metrics['audit_status'],'episodes':metrics['independently_replayed_episodes'],
                      'dataset_means':metrics['dataset_means']},indent=2))
