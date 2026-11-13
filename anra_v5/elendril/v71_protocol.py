"""Preregistered V7.1 phases: no outcome-driven scaling or silent matrix reduction."""
from .codec import digest

SEEDS=(7,19,41)
PHASES=('core','scale','continual','frozen')


def jobs(phase):
    if phase=='frozen':
        return [{'job_id':f'{m}-s{s}','profile':'S3','method':m,'seed':s,'epochs':0,'mechanism':'ce'}
                for s in SEEDS for m in ('uniform','blocked','interleaved')]
    conditions=({'core':[('S3','uniform','ce'),('S3','blocked','ce'),('S3','blocked','intermediate')],
                 'scale':[(p,m,'ce') for p in ('S8','S20') for m in ('uniform','blocked')],
                 'continual':[('S3',m,'ce') for m in ('b-only','replay','fresh-b')]})
    if phase not in conditions: raise ValueError('unknown V7.1 phase')
    return [{'job_id':f'{p}-{m}-{mechanism}-s{s}', 'profile':p,
             'method':m if phase!='continual' else 'blocked', 'seed':s,
             'epochs':4 if phase!='continual' else 2,'mechanism':mechanism,
             **({'stage':m} if phase=='continual' else {})}
            for s in SEEDS for p,m,mechanism in conditions[phase]]


def trainer_job(job): return {k:job[k] for k in ('job_id','profile','method','seed','epochs','mechanism')}


def protocol():
    return {'schema':'elendril-v71-protocol-v1','seeds':list(SEEDS),
        'jobs':{p:jobs(p) for p in PHASES},'suite_seed':17071,'train_seed':606,'train_bytes':2200000,
        'new_skill_seed':27171,'new_skill_worlds':512,'new_skill_hops':3,
        'maximum_session_minutes':330,'acquisition_minimum':.8,
        'scale_comparison':'S3/S8/S20 x uniform/blocked x three seeds: 18 runs over core and scale phases.',
        'mechanism':'Three additional S3 blocked intermediate-supervision runs; extra targets and work explicitly charged.',
        'continual':'A=missing+memory, then novel three-hop B. B-only/replay/fresh-B share B worlds; replay adds A exposure, not matched total compute.',
        'decoding_control':'Every final model compared with gold-free legal-action grammar decoding at frozen weights; work measured, not asserted equal.',
        'frozen_source':'413a3eccf4344ef2a39051b870e92feb6271b769; all nine audited V7 final generations required, weights-only reevaluation.',
        'evaluation':'Base at epochs 1/2; complete stress suite at final. Reserved confirmation included only at final, no hyperparameter selection.',
        'costs':'Fixed-policy repricing at 0,0.1,0.5,1 query cost. No learned cost conditioning claim.',
        'admission':'Production-shaped probes on both T4s. A whole declared phase fits, or it is deferred; no silent seed/epoch pruning.',
        'deferred':['4x data factorial pending compute qualification; inherited update bounds preserved',
                    'parameter/compute-matched neural pointer architecture not implemented',
                    'genuine forward-transfer promotion requires novel-C learning curve and matched baselines',
                    'independent authorship/external sealed evaluation not available'],
        'scientific_status':'CAUSE UNRESOLVED; no automatic mechanism or capability promotion.'}


PROTOCOL_SHA=digest(protocol())


def validate(job,phase):
    if job not in jobs(phase): raise ValueError('job outside preregistered phase')
