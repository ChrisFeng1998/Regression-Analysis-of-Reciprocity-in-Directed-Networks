"""MCMC-based CMLE estimator and diagnostics.
"""


import math
import time
import numpy as np


def _kernel(input_graph,V1,V2,theta,draws,burnin,interval,seed):
    np.random.seed(seed)
    A = input_graph.copy()
    n = A.shape[0]
    trace = np.zeros((draws,3))
    masks = np.zeros(draws,np.int64)
    counts = np.zeros(7,np.int64)
    tail = np.zeros(6,np.int64)
    head = np.zeros(6,np.int64)
    uu = np.zeros(6,np.int64)
    ww = np.zeros(6,np.int64)
    t0,t1,t2 = 0.,0.,0.
    for i in range(n):
        for j in range(i+1,n):
            if A[i,j] and A[j,i]:
                t0 += 1.; t1 += V1[i,j]; t2 += V2[i,j]
    row = 0
    for step in range(1,burnin+draws*interval+1):
        counts[0] += 1
        if np.random.random()<.05:
            counts[1] += 1
        else:
            switch = np.random.random()<.8
            valid = False
            nt = 0
            if switch and n>=4:
                a = int(np.random.random()*n)
                c = int(np.random.random()*n)
                while c==a:
                    c = int(np.random.random()*n)
                b,d = -1,-1
                for search in range(16):
                    candidate = int(np.random.random()*n)
                    if candidate!=a and candidate!=c and A[a,candidate]==1 and A[c,candidate]==0:
                        b=candidate;break
                if b>=0:
                    for search in range(16):
                        candidate = int(np.random.random()*n)
                        if candidate!=a and candidate!=c and A[a,candidate]==0 and A[c,candidate]==1:
                            d=candidate;break
                if b>=0 and d>=0:
                    valid = A[a,b]==A[c,d] and A[a,d]==A[c,b] and A[a,b]!=A[a,d]
                    if valid:
                        nt=4
                        tail[0],head[0]=a,b;tail[1],head[1]=c,d
                        tail[2],head[2]=a,d;tail[3],head[3]=c,b
            elif not switch and n>=3:
                a=int(np.random.random()*n)
                b=int(np.random.random()*n)
                while b==a:b=int(np.random.random()*n)
                c=int(np.random.random()*n)
                while c==a or c==b:c=int(np.random.random()*n)
                valid=(A[a,b]==A[b,c] and A[b,c]==A[c,a] and
                    A[b,a]==A[c,b] and A[c,b]==A[a,c] and A[a,b]!=A[b,a])
                if valid:
                    nt=6
                    tail[0],head[0]=a,b;tail[1],head[1]=b,c;tail[2],head[2]=c,a
                    tail[3],head[3]=b,a;tail[4],head[4]=c,b;tail[5],head[5]=a,c
            if not valid:
                counts[2]+=1
            else:
                counts[3]+=1;counts[5 if switch else 6]+=1
                pairs=0
                for k in range(nt):
                    u,w=min(tail[k],head[k]),max(tail[k],head[k])
                    seen=False
                    for h in range(pairs):
                        if uu[h]==u and ww[h]==w:seen=True
                    if not seen:
                        uu[pairs],ww[pairs]=u,w;pairs+=1
                d0,d1,d2=0.,0.,0.
                for k in range(pairs):
                    u,w=uu[k],ww[k]
                    if A[u,w] and A[w,u]:d0-=1.;d1-=V1[u,w];d2-=V2[u,w]
                for k in range(nt):A[tail[k],head[k]]=1-A[tail[k],head[k]]
                for k in range(pairs):
                    u,w=uu[k],ww[k]
                    if A[u,w] and A[w,u]:d0+=1.;d1+=V1[u,w];d2+=V2[u,w]
                log_ratio=theta[0]*d0+theta[1]*d1+theta[2]*d2
                if math.log(max(np.random.random(),1e-300))<min(0.,log_ratio):
                    counts[4]+=1;t0+=d0;t1+=d1;t2+=d2
                else:
                    for k in range(nt):A[tail[k],head[k]]=1-A[tail[k],head[k]]
        if step>burnin and (step-burnin)%interval==0:
            if row%1000==0:
                full0,full1,full2=0.,0.,0.
                for i in range(n):
                    for j in range(i+1,n):
                        if A[i,j] and A[j,i]:full0+=1.;full1+=V1[i,j];full2+=V2[i,j]
                if max(abs(full0-t0),abs(full1-t1),abs(full2-t2))>1e-7:
                    raise ValueError('Incremental statistic drift')
            trace[row,0],trace[row,1],trace[row,2]=t0,t1,t2
            if n<=5:
                mask,bit=0,0
                for i in range(n):
                    for j in range(n):
                        if i!=j:
                            if A[i,j]:mask|=1<<bit
                            bit+=1
                masks[row]=mask
            row+=1
    return trace,A,counts,masks


_compiled = None


def get_kernel(backend):
    global _compiled
    if backend=='python':return _kernel
    if backend!='numba':raise ValueError(backend)
    if _compiled is None:
        from numba import njit
        _compiled=njit(cache=True)(_kernel)
    return _compiled


def statistics(A,V):
    i,j=np.triu_indices(len(A),1)
    mutual=A[i,j]*A[j,i]
    return np.array([mutual.sum(),mutual@V[0][i,j],mutual@V[1][i,j]],float)


def run_chain(A,V,theta,draws,burnin,interval,seed,backend):
    trace,final,counts,masks=get_kernel(backend)(np.asarray(A,np.int8),V[0],V[1],
        np.asarray(theta,float),int(draws),int(burnin),int(interval),int(seed))
    if not (np.array_equal(A.sum(0),final.sum(0)) and np.array_equal(A.sum(1),final.sum(1))
            and not np.diag(final).any()):
        raise RuntimeError('Degree or loop constraint violated')
    if not np.allclose(trace[-1],statistics(final,V),rtol=0,atol=1e-7):
        raise RuntimeError('Final sufficient statistics disagree')
    return dict(trace=trace,final_graph=final,counts=counts,masks=masks)


def warmup():
    A=np.zeros((4,4),np.int8)
    A[0,1]=A[1,2]=A[2,3]=A[3,0]=1
    V=[np.zeros((4,4)),np.zeros((4,4))]
    start=time.perf_counter()
    run_chain(A,V,np.zeros(3),2,2,1,17,'numba')
    return time.perf_counter()-start

import math
import numpy as np
from scipy.fft import rfft,irfft,next_fast_len
from scipy.special import ndtri
from scipy.stats import rankdata


def _rhat_basic(x):
    n=len(x);within=np.var(x,axis=0,ddof=1).mean()
    if within<1e-20 or not np.isfinite(within):return float('inf')
    return float(np.sqrt(((n-1)/n*within+np.var(x.mean(0),ddof=1))/within))


def rank_rhat(x):
    h=len(x)//2
    split=np.column_stack([x[:h],x[-h:]])
    def normal(v):return ndtri((rankdata(v.ravel())-.375)/(v.size+.25)).reshape(v.shape)
    return max(_rhat_basic(normal(split)),_rhat_basic(normal(abs(split-np.median(split)))))


def spectral_ar(series):
    x=np.asarray(series,float);n=len(x);x=x-x.mean()
    if np.var(x)<1e-20:return 0.,0
    ordermax=min(n-1,int(math.floor(10*math.log10(n))))
    size=next_fast_len(2*n-1)
    transform=rfft(x,size)
    covariance=irfft(transform*transform.conjugate(),size)[:ordermax+1]/n
    variance=covariance[0]
    best_aic=n*math.log(variance)
    best_var,best_coeff,best_order=variance,np.empty(0),0
    coeff=np.empty(0)
    for order in range(1,ordermax+1):
        reflection=(covariance[order]-np.dot(coeff,covariance[1:order][::-1]))/variance
        coeff=np.r_[coeff-reflection*coeff[::-1],reflection]
        variance=variance*(1-reflection**2)
        if variance<=0:break
        aic=n*math.log(variance)+2*order
        if aic<=best_aic:
            best_aic,best_var,best_coeff,best_order=aic,variance,coeff.copy(),order
    prediction_variance=best_var*n/(n-(best_order+1))
    spectrum=prediction_variance/(1-best_coeff.sum())**2
    return float(spectrum),best_order


def diagnose(chains):
    arrays=np.asarray(chains,float)
    m,n,d=arrays.shape
    if n<100 or m<2:raise ValueError('Too few chains/draws for diagnostic calculation')
    batch_size=max(10,math.floor(math.sqrt(n)))
    nb=n//batch_size
    batch_covariances=[]
    spectral=np.empty((m,d));ess=np.empty((m,d))
    for i,z in enumerate(arrays):
        means=z[:nb*batch_size].reshape(nb,batch_size,d).mean(1)
        batch_covariances.append(batch_size*np.cov(means,rowvar=False,ddof=1))
        for j in range(d):
            spectral[i,j],_=spectral_ar(z[:,j])
            ess[i,j]=n*np.var(z[:,j],ddof=1)/spectral[i,j] if spectral[i,j]>0 else 0.
    batch_variance=np.diag(np.sum(batch_covariances,axis=0))/(m*m*n)
    ar_variance=spectral.sum(0)/(m*m*n)
    return dict(mean=arrays.mean((0,1)),mcse=np.sqrt(np.maximum(batch_variance,ar_variance)),
        ess_ar=ess.sum(0),rhat=np.array([rank_rhat(arrays[:,:,j].T) for j in range(d)]))


import time
import numpy as np
from scipy.optimize import minimize

FORMAL=dict(burnin=500000,interval=128,baseline_draws=50000,
    baseline_cap=400000,extended_cap=800000,rounds=6,
    init_draws=2000,init_burnin=8192,init_interval=128,init_rounds=20,
    graph_start_burnin=50000)


def homogeneous_start(A,V):
    i,j=np.triu_indices(len(A),1);v1,v2=V[0][i,j],V[1][i,j]
    y=A[i,j]+A[j,i];mut=A[i,j]*A[j,i]
    observed=np.array([y.sum(),mut.sum(),mut@v1,mut@v2])
    empty=(y==0).sum()+.5;single=(y==1).sum()/2+.5;both=(y==2).sum()+.5
    def objective(z):
        eta11=2*z[0]+z[1]+z[2]*v1+z[3]*v2
        mx=np.maximum(0,np.maximum(z[0],eta11))
        denominator=np.exp(-mx)+2*np.exp(z[0]-mx)+np.exp(eta11-mx)
        p10=np.exp(z[0]-mx)/denominator;p11=np.exp(eta11-mx)/denominator
        value=np.sum(mx+np.log(denominator))-z@observed
        gradient=np.array([np.sum(2*p10+2*p11),p11.sum(),v1@p11,v2@p11])-observed
        return value,gradient
    initial=np.array([np.log(single/empty),np.log(both*empty/single**2),0.,0.])
    result=minimize(objective,initial,jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':300})
    if not np.isfinite(result.x).all() or np.max(abs(result.jac))>1e-4:
        raise RuntimeError('Homogeneous initializer did not converge')
    return result.x[1:]


def make_batch(A,V,theta,graphs,draws,burnin,interval,seed,stage,round_index,backend):
    outputs=[run_chain(graph,V,theta,draws,burnin,interval,
        seed+stage*10000+round_index*100+ch,backend) for ch,graph in enumerate(graphs)]
    chains=np.array([x['trace'] for x in outputs])
    information=np.cov(chains.reshape(-1,3),rowvar=False,ddof=1)
    if np.linalg.eigvalsh(information).min()<=1e-10:
        raise RuntimeError('Conditional information is singular')
    inverse=np.linalg.inv(information);se=np.sqrt(np.diag(inverse))
    raw=diagnose(chains)
    projected=diagnose(chains@inverse)
    correction=inverse@(statistics(A,V)-raw['mean'])
    rhat=float(max(raw['rhat'].max(),projected['rhat'].max()))
    passed=bool(np.isfinite(rhat) and rhat<1.01 and np.max(projected['mcse']/se)<=.05)
    return dict(chains=chains,finals=np.array([x['final_graph'] for x in outputs]),
        counts=np.array([x['counts'] for x in outputs]),theta=np.array(theta),draws=int(draws),
        information=information,se=se,raw=raw,projected=projected,correction=correction,
        diagnostics_pass=passed,max_rhat=rhat)


def refine(A,V,anchor,N,cap,graphs,seed,stage,backend,cfg,progress=None):
    sample=make_batch(A,V,anchor,graphs,N,cfg['burnin'],cfg['interval'],seed,stage,0,backend)
    reference=anchor+np.clip(sample['correction'],-.5,.5)
    construction=sample
    history=[]
    for round_index in range(1,cfg['rounds']+1):
        verification=make_batch(A,V,reference,sample['finals'],N,cfg['burnin'],cfg['interval'],
            seed,stage,round_index,backend)
        se=verification['se']
        quality=bool(verification['diagnostics_pass'] and construction['diagnostics_pass'] and
            construction['max_rhat']<1.01 and
            np.max(construction['projected']['mcse']/se)<=.05 and
            np.max(abs(verification['correction']/se))<=.1)
        item=dict(round=round_index,draws=N,max_rhat=verification['max_rhat'],
            max_mcse_se=float(np.max(verification['projected']['mcse']/se)),
            max_construction_mcse_se=float(np.max(construction['projected']['mcse']/se)),
            max_residual_se=float(np.max(abs(verification['correction']/se))),passed=quality)
        history.append(item)
        if progress:progress(f'CMLE_STAGE_{stage} {item}')
        if quality:break
        sample=verification
        if not verification['diagnostics_pass']:
            N=min(cap,2*N)
        else:
            reference=reference+np.clip(verification['correction'],-.5,.5)
            construction=verification
            if np.max(construction['projected']['mcse']/se)>.05:N=min(cap,2*N)
    return dict(theta=verification['theta'],pass_diagnostics=quality,
        construction=construction,verification=verification,history=history,
        final_draws=verification['draws'])


def preliminary_fit(A,V,seed,backend,cfg,progress=None):
    theta=homogeneous_start(A,V)
    graphs=np.array([A.copy() for _ in range(4)])
    observed=statistics(A,V);history=[]
    # This supplies a starting point only. It is not reported as a certified fit.
    for iteration in range(cfg['init_rounds']):
        draws=[]
        for ch in range(4):
            out=run_chain(graphs[ch],V,theta,cfg['init_draws'],cfg['init_burnin'],
                cfg['init_interval'],seed+100*iteration+ch,backend)
            graphs[ch]=out['final_graph'];draws.append(out['trace'])
        raw=np.concatenate(draws)
        information=np.cov(raw,rowvar=False,ddof=1)
        if np.linalg.eigvalsh(information).min()<=1e-10:
            raise RuntimeError('Preliminary conditional information is singular')
        inverse=np.linalg.inv(information);se=np.sqrt(np.diag(inverse))
        correction=inverse@(observed-raw.mean(0))
        size=float(np.max(abs(correction/se)))
        theta=theta+np.clip(correction,-.5,.5)
        history.append(dict(iteration=iteration+1,correction_se=size,theta=theta.tolist()))
        if progress:progress(f'CMLE_INITIALIZER iteration={iteration+1} correction_SE={size:.4g}')
        if iteration>=1 and size<.35:break
    return dict(theta=theta,history=history)


def fit_cmle(A,V,seed,backend='numba',config=None,progress=None):
    cfg=dict(FORMAL if config is None else config)
    phases={}
    def phase(name,fn):
        begin=time.perf_counter();value=fn();phases[name]=time.perf_counter()-begin
        return value
    initial=phase('initialization',lambda:preliminary_fit(A,V,seed,backend,cfg,progress))
    def starts():
        graphs=[A.copy()]
        for ch in range(1,4):
            tilt=np.zeros(3) if ch==1 else initial['theta']+np.array([1.5 if ch==2 else -1.5,0,0])
            graphs.append(run_chain(A,V,tilt,1,cfg['graph_start_burnin'],1,
                seed+40000+ch,backend)['final_graph'])
        return np.array(graphs)
    graphs=phase('dispersed_graph_initialization',starts)
    baseline=phase('baseline',lambda:refine(A,V,initial['theta'],cfg['baseline_draws'],
        cfg['baseline_cap'],graphs,seed,1,backend,cfg,progress))
    extended=None
    if baseline['pass_diagnostics']:
        extended=phase('extended',lambda:refine(A,V,baseline['theta'],
            min(cfg['extended_cap'],2*baseline['final_draws']),cfg['extended_cap'],
            baseline['verification']['finals'],seed,2,backend,cfg,progress))
    def certify():
        for stage in [baseline,extended]:
            if stage is None:continue
            sample=stage['verification'];chains=sample['chains'];info=np.cov(chains.reshape(-1,3),rowvar=False,ddof=1)
            inverse=np.linalg.inv(info);correction=inverse@(statistics(A,V)-chains.mean((0,1)))
            if not np.allclose(correction,sample['correction'],rtol=0,atol=1e-8):
                raise RuntimeError('Saved residual correction disagrees')
            if not np.array_equal(sample['theta'],stage['theta']):
                raise RuntimeError('Diagnostic trace is not at the reported fixed parameter')
            for graph in sample['finals']:
                if not (np.array_equal(graph.sum(0),A.sum(0)) and np.array_equal(graph.sum(1),A.sum(1))):
                    raise RuntimeError('Degree constraint changed')
        return True
    phase('final_certification',certify)
    passed=bool(baseline['pass_diagnostics'] and extended is not None and extended['pass_diagnostics'])
    shift=None if extended is None else float(np.max(abs((extended['theta']-baseline['theta'])/extended['verification']['se'])))
    return dict(theta=None if extended is None else extended['theta'],pass_diagnostics=passed,
        wall_seconds=sum(phases.values()),phase_seconds=phases,backend=backend,config=cfg,
        initializer=initial,baseline=baseline,extended=extended,max_extension_shift_se=shift,
        extension_stable=bool(shift is not None and shift<=.15))

import numpy as np
from scipy.optimize import minimize

def setup(A,V):
    i,j=np.triu_indices(len(A),1);v1,v2=V[0][i,j],V[1][i,j]
    y=A[i,j]+A[j,i];mut=A[i,j]*A[j,i]
    observed=np.array([y.sum(),mut.sum(),mut@v1,mut@v2])
    empty=(y==0).sum()+.5;single=(y==1).sum()/2+.5;both=(y==2).sum()+.5
    start=np.array([np.log(single/empty),np.log(both*empty/single**2),0.,0.])
    def probability(z):
        eta=2*z[0]+z[1]+z[2]*v1+z[3]*v2
        mx=np.maximum(0,np.maximum(z[0],eta))
        den=np.exp(-mx)+2*np.exp(z[0]-mx)+np.exp(eta-mx)
        return np.exp(z[0]-mx)/den,np.exp(eta-mx)/den,mx+np.log(den)
    def original(z):
        p10,p11,logz=probability(z)
        score=np.array([np.sum(2*p10+2*p11),p11.sum(),v1@p11,v2@p11])-observed
        return np.sum(logz)-z@observed,score
    def centered(reference):
        p10,p11,_=probability(reference)
        def objective(delta):
            eta=2*delta[0]+delta[1]+delta[2]*v1+delta[3]*v2
            change=2*p10*np.expm1(delta[0])+p11*np.expm1(eta)
            den=1+change
            new10=p10*np.exp(delta[0])/den;new11=p11*np.exp(eta)/den
            value=np.sum(np.log1p(change))-delta@observed
            score=np.array([np.sum(2*new10+2*new11),new11.sum(),v1@new11,v2@new11])-observed
            return value,score
        return objective
    return start,original,centered

def recover_homogeneous_start(A,V):
    start,original,centered=setup(A,V)
    reference_fit=minimize(original,start,jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':300})
    reference=reference_fit.x.copy()
    if not np.isfinite(reference).all():raise RuntimeError('Initializer recovery has a nonfinite reference')
    objective=centered(reference)
    fit=minimize(objective,np.zeros(4),jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':300})
    z=reference+fit.x
    value,gradient=original(z)
    if not np.isfinite(z).all() or np.max(abs(gradient))>1e-4:
        raise RuntimeError(f'Centered identical-likelihood initializer did not converge: score={np.max(abs(gradient))}')
    details=dict(repair='same likelihood, centered log1p/expm1 evaluation',
        optimizer='BFGS',gtol=1e-6,maxiter=300,required_absolute_score=1e-4,
        original_reference_score_max=float(np.max(abs(reference_fit.jac))),
        recovered_absolute_score_max=float(np.max(abs(gradient))),
        centered_score_max=float(np.max(abs(fit.jac))),
        recovered_parameters=z.tolist(),original_objective_at_solution=float(value),
        reference_solver_message=str(reference_fit.message),recovered_solver_message=str(fit.message))
    return z[1:],details
