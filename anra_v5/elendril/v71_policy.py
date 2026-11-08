"""Gold-free grammar control over the same frozen neural logits."""
from .v6_evaluation import NeuralPolicy
from .v6_data import render_public
from .representation import UTF8,SpecialToken
from .cognition import OperationCandidate
import math


class GrammarPolicy(NeuralPolicy):
    def for_case(self,case,*,memory_available=None):
        return GrammarPolicy(self.backend,case.surface,
            memory_available=case.memory is not None if memory_available is None else memory_available)

    def propose(self,state,legal,channels,positions):
        prompt=render_public(state,channels,self.surface,self.memory_available and 'RETRIEVE' in legal)
        self.prompts.append(prompt)
        ids=[UTF8.bos_id,*UTF8.encode(prompt),UTF8.special(SpecialToken.ANSWER)]
        choices={}
        if 'ANSWER' in legal: choices.update({x:('ANSWER',x) for x in 'ABCDEFGHIJKLMNOP'})
        if 'ABSTAIN' in legal: choices['!']=('ABSTAIN',None)
        if 'QUERY' in legal and len(channels)==1: choices['Q']=('QUERY',channels[0])
        if 'RETRIEVE' in legal and self.memory_available: choices['R']=('RETRIEVE',None)
        if 'DELIBERATE' in legal: choices['~']=('DELIBERATE',None)
        if not choices: return None
        tokens=[]; probabilities=[]; used=0; op=arg=None
        for stage in range(3):
            if len(ids)>self.backend.binding.context_length or used+len(ids)>positions: return None
            distribution=self.backend.next_distribution(ids); used+=len(ids)
            allowed=([UTF8.encode(x)[0] for x in choices] if stage==0 else
                     [UTF8.encode(x)[0] for x in 'ABCDEFGHIJKLMNOP'] if op=='DELIBERATE' and stage==1 else [UTF8.eos_id])
            token=max(allowed,key=lambda x:(float(distribution[x]),-x))
            tokens.append(token); probabilities.append(float(distribution[token])); ids.append(token)
            if stage==0: op,arg=choices[UTF8.decode([token])]
            elif op=='DELIBERATE' and stage==1: arg=UTF8.decode([token])
            else: break
        return OperationCandidate(op,arg,math.prod(probabilities),used,len(tokens),tuple(tokens))
