"""Linear-Gaussian distributed-source inversion and conditional information design.

Policy input is public navigation and acquired observations ONLY. No simulator,
generator, DataStore, files, task IDs, or truth are accessible here.
The ADE Green function assumes homogeneous measured flow and unbounded transport.
This is an explicit approximate physical model, not the data-generating solver.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

CONFIG = {
    'diffusivities_m2_s': [0.02, 0.08, 0.32, 1.28],
    'vertical_horizontal_diffusivity_ratio': 0.1,
    'release_durations_s': [150., 300., 600.],
    'quadrature_nodes': 12,
    'source_prior_sd': 0.15,
    'background_prior_sd': [0.10, 0.10, 0.10, 0.10, 0.10],
    'model_discrepancy_sd': 0.02,
    'initial_edges': 12,
    'planning_edges': 8,
    'fit_limit': 32,
    'seeds': [0, 1, 2],
}


def green(xyz, velocity, diffusivity, duration):
    """Average unit-mass Gaussian pulse over a constant release interval.

    Finite source-cell width regularizes the point singularity. One GLOBAL
    normalization keeps source strengths comparable across candidate nodes.
    """
    xyz=np.asarray(xyz,float)
    spacing=np.array([min(np.diff(np.unique(xyz[:,i]))) for i in range(3)])
    t,w=np.polynomial.legendre.leggauss(CONFIG['quadrature_nodes'])
    t=(t+1)*duration/2; w=w/2
    result=np.zeros((len(xyz),len(xyz)))
    d=diffusivity*np.array([1.,1.,CONFIG['vertical_horizontal_diffusivity_ratio']])
    for age,weight in zip(t,w):
        variance=2*d*age+spacing**2/12
        delta=xyz[:,None,:]-xyz[None,:,:]-np.asarray(velocity)*age
        result+=weight*np.exp(-.5*np.sum(delta**2/variance,axis=2))/np.sqrt((2*np.pi)**3*np.prod(variance))
    return result/max(result.max(),1e-20)


def prior(xyz):
    xyz=np.asarray(xyz,float)
    spacing=np.array([min(np.diff(np.unique(xyz[:,i]))) for i in range(3)])
    delta=(xyz[:,None,:]-xyz[None,:,:])/spacing
    # A Gaussian net-source field permits distributed releases and sinks.
    # Only POSITIVE inferred loading may be reported as a source.
    q=CONFIG['source_prior_sd']**2*(np.exp(-.5*np.sum(delta**2,axis=2))+np.eye(len(xyz))*1e-6)
    p=(xyz-xyz.mean(axis=0))/np.maximum(np.ptp(xyz,axis=0),1.)
    b=np.column_stack([np.ones(len(xyz)),p,p[:,2]**2])
    b*=np.asarray(CONFIG['background_prior_sd'])
    return q,b


def infer(navigation, samples, calibration):
    """Exact Gaussian conditioning within each fixed transport component.

    Diffusivity/duration have a uniform discrete prior; model evidence produces
    mixture weights. Flow is estimated from unique measured-node averages.
    Source covariance is not a calibrated posterior under physical-model error.
    """
    xyz=np.asarray([[r[k] for k in 'xyz'] for r in navigation])
    ids={int(r['node_id']):i for i,r in enumerate(navigation)}
    grouped={}
    for s in samples:grouped.setdefault(int(s['node_id']),[]).append(s)
    observed=sorted(grouped)
    ix=np.array([ids[i] for i in observed])
    y=[]; noise=[]; flow=[]; refs={}
    for c in calibration:
        if c['channel']=='fdom':refs[c['layer']]=(float(c['background_median']),float(c['lod']))
    # Unknown bathymetry is NOT used to extrapolate unmeasured layer references.
    # Background basis fits all offsets; the same constant reference is used
    # everywhere, based solely on public calibration.
    reference=np.mean([v[0] for v in refs.values()])
    for node in observed:
        ss=grouped[node]
        y.append(np.mean([r['fdom'] for r in ss])-reference)
        ratio=ss[-1]['z']/ss[-1]['bed_depth']
        layer='surface' if ratio<1/3 else 'middle' if ratio<2/3 else 'bottom'
        noise.append(refs[layer][1]**2/len(ss))
        flow.append([np.mean([r[k] for r in ss]) for k in ['u','v','w']])
    y=np.asarray(y); noise=np.asarray(noise); velocity=np.mean(flow,axis=0)
    q,b=prior(xyz); bb=b@b.T+np.eye(len(xyz))*CONFIG['model_discrepancy_sd']**2
    # Background conditional on the entire latent source; nuisance uncertainty
    # is excluded from the information reward about the source itself.
    cf=cho_factor(bb[np.ix_(ix,ix)]+np.diag(noise),lower=True)
    nuisance=bb-bb[:,ix]@cho_solve(cf,bb[ix,:])
    components=[]; evidence=[]
    for diffusivity in CONFIG['diffusivities_m2_s']:
        for duration in CONFIG['release_durations_s']:
            h=green(xyz,velocity,diffusivity,duration)
            sourcecov=h@q@h.T
            full=sourcecov+bb
            c=full[np.ix_(ix,ix)]+np.diag(noise)
            factor=cho_factor(c,lower=True,check_finite=False)
            alpha=cho_solve(factor,y,check_finite=False)
            cross=full[:,ix]
            covariance=full-cross@cho_solve(factor,cross.T,check_finite=False)
            covariance=(covariance+covariance.T)/2
            source_mean=q@h[ix].T@alpha
            # Expected source contribution, separated from nuisance background.
            contribution=h@source_mean
            components.append(dict(source=source_mean,contribution=contribution,
                                   covariance=covariance,mean=cross@alpha))
            evidence.append(-.5*(y@alpha+2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi)))
    weights=np.exp(np.asarray(evidence)-logsumexp(evidence))
    source=sum(w*c['source'] for w,c in zip(weights,components))
    contribution=sum(w*c['contribution'] for w,c in zip(weights,components))
    mean=sum(w*c['mean'] for w,c in zip(weights,components))
    covariance=sum(w*(c['covariance']+np.outer(c['mean']-mean,c['mean']-mean)) for w,c in zip(weights,components))
    peak=int(ix[np.argmax(contribution[ix])])
    location=int(np.argmax(source)) if source.max()>0 else None
    return dict(source=source,contribution=contribution,covariance=covariance,
                nuisance=nuisance,noise=float(np.mean([v[1]**2 for v in refs.values()])),
                weights=weights,velocity=velocity,peak_node=int(navigation[peak]['node_id']),
                source_position=None if location is None else dict(zip('xyz',xyz[location].tolist())),
                source_node=None if location is None else int(navigation[location]['node_id']),
                measured_nodes=len(ix))


def information_route(result, current, neighbors, steps):
    """Greedy receding-horizon source information design on real adjacent edges.

    0.5*log predictive variance / variance conditional on source. The mixture
    covariance uses moment matching; conditional Gaussian components are exact.
    Covariance is updated after each planned arrival so no sample is free.
    """
    covariance=result['covariance'].copy(); nuisance=result['nuisance'].copy()
    noise=result['noise']; route=[]; gains=[]
    for _ in range(steps):
        candidates=neighbors[current]
        gains_here={i:max(0.,.5*np.log(max(covariance[i,i]+noise,noise)/max(nuisance[i,i]+noise,noise))) for i in candidates}
        chosen=max(candidates,key=lambda i:(gains_here[i],-i))
        route.append(chosen);gains.append(gains_here[chosen])
        for matrix in [covariance,nuisance]:
            col=matrix[:,chosen].copy()
            matrix-=np.outer(col,col)/(matrix[chosen,chosen]+noise)
            matrix[:]=(matrix+matrix.T)/2
        current=chosen
    return route,gains
