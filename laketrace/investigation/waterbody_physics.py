"""Explicit computational bed completion and physical release-face geometry.

Input rows are retained exactly. Added nodes carry constant deepest-row background
extrapolation, labelled as model completion. This is not new measured evidence.
"""
import numpy as np
import pandas as pd
from laketrace.env.graph import NodeGraph


def complete_water_columns(frame):
    frame=frame.copy()
    frame['record_origin']='source_archive'
    graph=NodeGraph(frame[['x_m_east','y_m_north','depth_m']].itertuples(index=False,name=None))
    dz=graph.spacing[2]
    if dz is None or dz<=0:
        raise ValueError('vertical spacing required')
    added=[]
    for _,column in frame.groupby(['x_m_east','y_m_north']):
        ordered=column.sort_values('depth_m')
        if np.ptp(ordered.bed_depth_m.to_numpy())>1e-6:
            raise ValueError('inconsistent bed elevation in column')
        row=ordered.iloc[-1].copy();bed=float(row.bed_depth_m)
        if float(row.depth_m)>bed:
            raise ValueError('input node below bed')
        z=float(row.depth_m)+dz
        while z < bed-1e-8:
            extra=row.copy();extra['depth_m']=round(z,8);extra['record_origin']='model_bed_completion'
            added.append(extra);z+=dz
    result=pd.concat([frame,pd.DataFrame(added)],ignore_index=True) if added else frame
    result=result.sort_values(['x_m_east','y_m_north','depth_m']).reset_index(drop=True)
    return result


def release_faces(frame, source_indices, kind):
    graph=NodeGraph(frame[['x_m_east','y_m_north','depth_m']].itertuples(index=False,name=None))
    dx,dy,dz=graph.spacing
    faces=[]
    for i in source_indices:
        row=frame.iloc[int(i)];x,y,z=float(row.x_m_east),float(row.y_m_north),float(row.depth_m)
        column=frame[(frame.x_m_east==x)&(frame.y_m_north==y)].depth_m.to_numpy()
        before=column[column<z];after=column[column>z]
        lower=0. if not len(before) else (before.max()+z)/2
        upper=float(row.bed_depth_m) if not len(after) else (after.min()+z)/2
        bounds={'x':[x-dx/2,x+dx/2],'y':[y-dy/2,y+dy/2],'z':[lower,upper]}
        if kind in {'internal_release','physical_resuspension'}:
            if len(after):raise ValueError('bed source receiver is not deepest model cell')
            bed=float(row.bed_depth_m)
            if bed-z>dz+1e-6:raise ValueError('unresolved water column beneath bed source')
            bounds['z']=[bed,bed];boundary='bed'
        else:
            # A prescribed mass discharge on the computational-window boundary.
            # This does not claim a mapped real pipe or a shoreline inlet.
            directions=[d for d in ['west','east','south','north'] if d not in graph.neighbor_map[int(i)]]
            if not directions:raise ValueError('external source is not a boundary cell')
            direction=directions[0];axis='x' if direction in {'west','east'} else 'y'
            edge=bounds[axis][0 if direction in {'west','south'} else 1]
            bounds[axis]=[edge,edge];boundary='open_window_release'
        bounds={a:[float(v) for v in limits] for a,limits in bounds.items()}
        widths=[b[1]-b[0] for b in bounds.values()]
        area=float(np.prod([w for w in widths if w>0]))
        faces.append({'face_id':f'face-{int(i)}','bounds':bounds,'boundary_type':boundary,
                      'area_m2':area,'receiving_node':{'x':x,'y':y,'z':z}})
    return {'type':'axis_aligned_faces','faces':faces}


def component_transport(graph, volumes, dissolved_operator, particle_operator, weights,
                        mass, schedule, duration, organic_fraction=.6,
                        decay_s=1/86400, settling_m_s=1e-4, step_scale=.1):
    """Positive dissolved/particle transport with degradation and deposition.

    Column 0: fluorescent dissolved tracer; 1: particles; 2: degraded dissolved
    material (an advected oxygen-demand proxy). No reaction products disappear
    from the mass ledger. Bed particle export is recorded as deposition.
    All components use consistent abstract tracer mass units.
    """
    from scipy import sparse
    n=len(volumes);dx,dy,_=graph.spacing
    if not 0<=organic_fraction<=1 or decay_s<0 or settling_m_s<0:
        raise ValueError('invalid component parameters')
    area=dx*dy;rates=settling_m_s*area/volumes
    rows=[];cols=[];vals=[];bed_rates=np.zeros(n)
    for i,es in enumerate(graph.neighbor_map):
        rows.append(i);cols.append(i);vals.append(-rates[i])
        if 'down' in es:
            rows.append(es['down'].target);cols.append(i);vals.append(rates[i])
        else:bed_rates[i]=rates[i]
    settling=sparse.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr()
    Ap=particle_operator+settling
    od=np.maximum(-np.asarray(dissolved_operator.sum(0)).ravel(),0)
    op=np.maximum(-np.asarray(particle_operator.sum(0)).ravel(),0)
    maximum=max(float((-dissolved_operator.diagonal()).max()+decay_s),float((-Ap.diagonal()).max()))
    dt=step_scale/max(maximum,1e-12)
    x=np.zeros((n,3));exported=deposited=emitted=last=0.
    for when,fraction in [*schedule,(duration,0.)]:
        if when<last or when>duration:raise ValueError('unordered/out-of-window injection')
        steps=max(1,int(np.ceil((when-last)/dt)));h=(when-last)/steps
        for _ in range(steps):
            exported+=h*float(od@(x[:,0]+x[:,2])+op@x[:,1])
            deposited+=h*float(bed_rates@x[:,1])
            change=dissolved_operator@x[:,[0,2]]
            reaction=decay_s*x[:,0]
            dx0=change[:,0]-reaction;dx2=change[:,1]+reaction;dx1=Ap@x[:,1]
            x+=h*np.column_stack([dx0,dx1,dx2])
        x[:,0]+=weights*mass*fraction*organic_fraction
        x[:,1]+=weights*mass*fraction*(1-organic_fraction)
        emitted+=mass*fraction;last=when
    return x,{'injected':float(emitted),'remaining':float(x.sum()),'open_export':exported,
              'bed_deposition':deposited,'mass_error':abs(x.sum()+exported+deposited-emitted),
              'dt_s':dt,'organic_fraction':organic_fraction,'decay_s':decay_s,'settling_m_s':settling_m_s}


def component_channels(background, channels, components, volumes):
    """Dimensioned response proxies in each channel's native source units.

    fDOM follows dissolved fluorescent tracer; turbidity follows particles;
    conductivity/TDS follow total dissolved tracer; DO responds to accumulated
    degradation. Temperature, pH and chlorophyll do not acquire invented source
    signatures. These are registered proxies, not calibrated stoichiometry.
    """
    bg=np.maximum(background,0.)
    fd=channels.index('fdom');scale=max(np.quantile(bg[:,fd],.9)-np.quantile(bg[:,fd],.1),.1)
    dissolved,particles,product=(components/volumes[:,None]).T
    out=bg.copy()
    out[:,fd]+=dissolved
    for c,component in [('turb',particles),('cond',dissolved+product),('tds',dissolved+product)]:
        if c in channels:
            j=channels.index(c);native_scale=max(np.quantile(bg[:,j],.9)-np.quantile(bg[:,j],.1),0.)
            out[:,j]+=component/scale*native_scale
    if 'do' in channels:
        j=channels.index('do')
        out[:,j]=bg[:,j]*np.exp(-product/scale)
    return out
