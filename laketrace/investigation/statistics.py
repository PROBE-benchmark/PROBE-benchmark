"""Grouped uncertainty summaries; repetitions never become independent events."""
from collections import defaultdict
import random


def clustered_summary(rows):
    # Background is the coarsest dependence unit in this release. Data lineage
    # must be registered so shared backgrounds retain the same identity.
    groups = defaultdict(list)
    for row in rows:
        groups[row['background']].append(row)
    output = {'unit':'shared background', 'clusters':len(groups), 'bootstrap_replicates':2000,
              'scope':'Descriptive cluster resampling, not an unseen-generalization certificate.', 'metrics':{}}
    rng = random.Random(1700)
    values = list(groups.values())
    for key in ('L1_pass','L2_pass','L3_pass','L4_pass','L1_L4_joint_pass'):
        eligible = lambda r: key=='L1_pass' or (r['detectable'] and (key=='L1_L4_joint_pass' or r.get(key.replace('_pass','_applicable'),True)))
        selected=[r for r in rows if eligible(r)]
        point=sum(r.get(key) is True for r in selected)/len(selected) if selected else None
        draws=[]
        if len(groups)>=3 and selected:
            for _ in range(2000):
                sample=[r for _ in values for r in rng.choice(values) if eligible(r)]
                if sample:
                    draws.append(sum(r.get(key) is True for r in sample)/len(sample))
        draws.sort()
        output['metrics'][key]={'estimate':point,'interval_95': [draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]] if draws else None,
                                'warning':'few independent clusters; interval exploratory' if len(groups)<20 else None}
    return output
