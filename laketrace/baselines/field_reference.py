"""Bounded-memory approximate ADE source diagnostic using observations only.

A nonnegative continuous point input is fitted jointly with a smooth nuisance
background. It is a misspecified reference for distributed/bed releases, not an
identifiability proof or a calibrated posterior. No generator templates or truth.
"""
import numpy as np
from scipy.spatial import cKDTree

CONFIG={'coarse_candidates':128,'refine_neighbors':24,'refine_modes':4,
        'diffusivities':[.05,.2,.8],'duration_fractions':[.25,1.],
        'quadrature_nodes':6,'batch':32}

def spatial_candidates(xyz,count):
    scaled=(xyz-xyz.min(0))/np.maximum(np.ptp(xyz,axis=0),1.)
    chosen=[0];distance=np.sum((scaled-scaled[0])**2,axis=1)
    for _ in range(min(count,len(xyz))-1):
        i=int(np.argmax(distance));chosen.append(i)
        distance=np.minimum(distance,np.sum((scaled-scaled[i])**2,axis=1))
    return np.array(chosen)

def kernel(observed,sources,velocity,diffusion,duration,spacing):
    t,w=np.polynomial.legendre.leggauss(CONFIG['quadrature_nodes']);t=(t+1)*duration/2;w=w/2
    result=np.zeros((len(observed),len(sources)))
    for age,weight in zip(t,w):
        variance=2*diffusion*np.array([1.,1.,.1])*age+spacing**2/12
        delta=observed[:,None,:]-sources[None,:,:]-velocity*age
        result+=weight*np.exp(-.5*np.sum(delta**2/variance,axis=2))/np.sqrt(np.prod(2*np.pi*variance))
    return result

def infer(navigation,observations,horizon_s):
    """nav=Nx3; observations=Nx7: x,y,z,value,u,v,w. No private background."""
    nav=np.asarray(navigation,float);data=np.asarray(observations,float)
    if len(data)<16:raise ValueError('at least 16 distinct observations required')
    if not np.isfinite(data).all():raise ValueError('nonfinite observations')
    data=data[np.lexsort((data[:,2],data[:,1],data[:,0]))]
    xyz=data[:,:3];y=data[:,3];scaled=(xyz-nav.mean(0))/np.maximum(np.ptp(nav,axis=0),1.)
    x,yy,z=scaled.T;B=np.column_stack([np.ones(len(y)),x,yy,z,x*x,yy*yy,z*z,x*yy])
    # Spatially ordered split; reused across diagnostics, not independent testing.
    train=np.arange(len(y))%5!=0;test=~train
    X=B[train];pinv=np.linalg.pinv(X);beta=pinv@y[train];residual=y[train]-X@beta
    null=float(residual@residual);velocity=np.mean(data[:,4:7],axis=0)
    spacing=np.array([np.min(np.diff(np.unique(nav[:,i]))) if len(np.unique(nav[:,i]))>1 else 1. for i in range(3)])
    found=[]
    def fit(ids):
        for diffusion in CONFIG['diffusivities']:
            for fraction in CONFIG['duration_fractions']:
                duration=max(1.,horizon_s*fraction)
                for offset in range(0,len(ids),CONFIG['batch']):
                    ix=ids[offset:offset+CONFIG['batch']]
                    K=kernel(xyz[train],nav[ix],velocity,diffusion,duration,spacing)
                    projected=pinv@K;H=K-X@projected
                    h2=np.sum(H*H,axis=0);dot=residual@H
                    masses=np.maximum(0.,np.divide(dot,h2,out=np.zeros_like(dot),where=h2>1e-30))
                    rss=np.maximum(0.,null-2*masses*dot+masses*masses*h2)
                    for j,i in enumerate(ix):found.append((float(rss[j]),int(i),diffusion,duration,float(masses[j]),(beta-projected[:,j]*masses[j]).tolist()))
    coarse=spatial_candidates(nav,CONFIG['coarse_candidates']);fit(coarse)
    ordered=sorted(found,key=lambda r:r[:4]);modes=[]
    for r in ordered:
        if all(np.linalg.norm((nav[r[1]]-nav[i])/spacing)>2 for i in modes):modes.append(r[1])
        if len(modes)==CONFIG['refine_modes']:break
    tree=cKDTree(nav/spacing);near=tree.query(nav[modes]/spacing,k=min(CONFIG['refine_neighbors'],len(nav)))[1]
    refined=np.setdiff1d(np.asarray(near).ravel(),coarse);fit(refined)
    best=min(found,key=lambda r:r[:4]);rss,i,duration_k,duration,mass,beta_best=best
    pred=B[test]@np.asarray(beta_best)+kernel(xyz[test],nav[[i]],velocity,duration_k,duration,spacing)[:,0]*mass
    null_mse=float(np.mean((y[test]-B[test]@beta)**2));held_mse=float(np.mean((y[test]-pred)**2))
    alternatives=[];seen=set()
    for r in sorted(found,key=lambda r:r[:4]):
        if r[1] in seen:continue
        seen.add(r[1]);alternatives.append({'position':nav[r[1]].tolist(),'train_rss':r[0],'relative_to_best':r[0]/max(rss,1e-20)})
        if len(alternatives)==12:break
    return {'source_position':dict(zip('xyz',nav[i].tolist())), 'source_node_index':i,
            'null_train_rss':null,'source_train_rss':rss,'null_held_mse':null_mse,'source_held_mse':held_mse,
            'held_relative_improvement':1-held_mse/max(null_mse,1e-20),'candidate_count':len(coarse)+len(refined),
            'diffusivity':duration_k,'duration_s':duration,'amplitude':mass,'velocity':velocity.tolist(),
            'alternatives':alternatives,'observations':len(data),
            'scope':'Approximate continuous point ADE plus fitted quadratic background; reused held spatial subset is diagnostic, not independent validation. Distributed sources and nonuniform flow are outside this reference model.'}
