"""Plot measured results only; requires matplotlib, no training dependencies."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'METRICS.json').read_bytes())
groups=[('uniform','ce','Uniform CE','#7b8794'),('blocked','ce','Blocked CE','#2463a6'),('blocked','intermediate','Intermediate','#db7c26')]
fig,axs=plt.subplots(2,2,figsize=(12,8.4),layout='constrained')
fig.suptitle('ELENDRIL V7.1 core: confirmation and failure probes',fontsize=16)

def selected(method,mechanism,cohort):
    return [v for v in r['rows'] if v['method']==method and v['mechanism']==mechanism and v['cohort']==cohort]

def rate(rows): return 100*sum(v['raw']['correct'] for v in rows)/sum(v['raw']['episodes'] for v in rows)

ax=axs[0,0]
for x,(method,mech,label,color) in enumerate(groups):
    rows=selected(method,mech,'confirmation')
    for i,row in enumerate(sorted(rows,key=lambda v:v['seed'])):
        y=100*row['raw']['correct']/112
        px=x+(i-1)*.18
        ax.scatter(px,y,color=color,s=42)
        ax.annotate(str(row['seed']),(px,y),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    mean=rate(rows);ax.plot([x-.18,x+.18],[mean,mean],color='black',lw=2)
    ax.text(x,mean-10,f'{mean:.1f}%',ha='center',fontsize=10)
ax.set(xticks=range(3),xticklabels=[g[2] for g in groups],ylim=(0,105),ylabel='Correct (%)',title='A. Confirmation, seeds 7 / 19 / 41')
ax.text(.02,.04,'112 shared worlds per seed; bars mark means',transform=ax.transAxes,fontsize=8)

ax=axs[0,1]
for method,mech,label,color in groups:
    values=[rate(selected(method,mech,f'hop-{h}')) for h in (1,2,3,4)]
    ax.plot((1,2,3,4),values,'o-',label=label,color=color)
ax.axhline(100,color='black',ls=':',label='Last-row copy control')
ax.set(xticks=(1,2,3,4),ylim=(0,112),xlabel='Requested hops',ylabel='Correct (%)',title='B. Sparse-chain tests have a copying shortcut')
ax.legend(fontsize=8,loc='center left')
ax.text(.02,.8,'Different worlds across hop lengths; domain factors confounded',transform=ax.transAxes,fontsize=8)

ax=axs[1,0]
cohorts=('base','robustness-template','robustness-alias','robustness-distractor')
for i,(method,mech,label,color) in enumerate(groups):
    ax.bar([x+(i-1)*.24 for x in range(4)],[rate(selected(method,mech,c)) for c in cohorts],width=.23,color=color,label=label)
ax.set(xticks=range(4),xticklabels=['Base','Format','Alias','Distractor'],ylim=(0,105),ylabel='Correct (%)',title='C. Surface changes break familiar behaviors')
ax.text(.02,.94,'112 worlds × 3 seeds per condition',transform=ax.transAxes,fontsize=8)

ax=axs[1,1]
for i,(method,mech,label,color) in enumerate(groups[1:]):
    rows=selected(method,mech,'memory-manipulations')
    counts=[sum(v['panels'][p]['correct'] for v in rows) for p in ('memory-original','memory-stale-corrected')]
    ax.bar([x+(i-.5)*.3 for x in range(2)],[100*n/48 for n in counts],width=.29,color=color,label=label)
    for x,n in enumerate(counts):ax.text(x+(i-.5)*.3,100*n/48+2,f'{n}/48',ha='center',fontsize=9)
ax.set(xticks=(0,1),xticklabels=['Original memory','Stale + later correction'],ylim=(0,114),ylabel='Correct (%)',title='D. Intermediate training regresses on correction')
ax.legend(fontsize=8)
for ax in axs.flat:
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
fig.savefig(ROOT/'RESULTS.png',dpi=170)
print(ROOT/'RESULTS.png')
