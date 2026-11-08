
import sys, time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
core = sys.modules[__name__]  
def design(n, seed, a=.2, b=.1):
    rng = np.random.default_rng(seed)
    beta, alpha = rng.uniform(size=n), rng.uniform(size=n)
    alpha -= alpha.mean()
    beta -= beta.mean()
    i, j = np.triu_indices(n, 1)
    V = np.zeros((n, n, 2))
    V[i, j, 0] = rng.uniform(-1, 1, len(i))
    V[i, j, 1] = rng.normal(size=len(i))
    V[j, i] = V[i, j]
    return dict(alpha=alpha, beta=beta, V=V, mu=-a*np.log(n)+.5,
                theta=np.array([b*np.log(n)+.5, 1., 1.]), n=n, seed=seed)

def independent_draw(d, seed, mu_shift=0.):
    n = d["n"]
    i, j = np.triu_indices(n, 1)
    eta = d["mu"] + mu_shift + d["alpha"][:, None] + d["beta"][None, :]
    r = d["theta"][0] + d["V"][i, j] @ d["theta"][1:]
    logits = np.column_stack([np.zeros(len(i)), eta[i, j], eta[j, i],
                              eta[i, j]+eta[j, i]+r])
    p = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    u = np.random.default_rng(seed).uniform(size=len(i))
    states = np.sum(u[:, None] > p.cumsum(axis=1), axis=1)
    A = np.zeros((n, n), np.int8)
    A[i, j] = (states == 1) | (states == 3)
    A[j, i] = (states == 2) | (states == 3)
    density = np.mean((p[:, 1]+p[:, 2]+2*p[:, 3])/2)
    return A, float(density)

@njit(cache=True)
def raw_triples(A):
    n, total = len(A), 0
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                for k in range(n):
                    if k != i and k != j:
                        total += int(A[j, k])*int(A[i, k])
    return total

@njit(cache=True)
def change_matrix(A):
    n = len(A)
    C = np.zeros((n, n), np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    C[i, j] += (int(A[i, k])*int(A[k, j])
                                + int(A[i, k])*int(A[j, k])
                                + int(A[k, i])*int(A[k, j]))
    return C

@njit(cache=True)
def toggle(A, C, i, j, value):
    change = int(value) - int(A[i, j])
    triple_change = change*C[i, j]
    if change:
        for k in range(len(A)):
            if k != i and k != j:
                C[i, k] += change*(int(A[j, k])+int(A[k, j]))
                C[k, j] += change*(int(A[k, i])+int(A[i, k]))
                C[k, i] += change*int(A[k, j])
                C[j, k] += change*int(A[i, k])
        A[i, j] = value
    return triple_change

@njit(cache=True)
def network_stats(A):
    n = len(A)
    edges, mutual, paths = 0, 0, 0
    for i in range(n):
        out, inc = 0, 0
        for j in range(n):
            out += int(A[i, j])
            inc += int(A[j, i])
            if j > i:
                mutual += int(A[i, j])*int(A[j, i])
        edges += out
        paths += out*inc
    paths -= 2*mutual
    triples = raw_triples(A)
    return np.array([edges/(n*(n-1)), mutual/(n*(n-1)/2),
                     triples/(n-2), triples/max(1, paths)], np.float64)

def sample(d, delta, mu_shift, seed, draws=1, burn=128, interval=1, initial=None):
    n = d["n"]
    if initial is None:
        initial = independent_draw(d, seed+1)[0]
    eta = d["mu"]+mu_shift+d["alpha"][:, None]+d["beta"][None, :]
    reciprocity = d["theta"][0] + d["V"] @ d["theta"][1:]
    return gibbs(initial.astype(np.int8), eta, reciprocity, float(delta),
                 int(draws), int(burn), int(interval), int(seed))

@njit(cache=True)
def objective(theta, x1, x2, y, w):
    m, k = x1.shape
    total_weight = w.sum()
    val, grad, H = 0., np.zeros(k), np.zeros((k, k))
    for r in range(m):
        e1, e2 = 0., 0.
        for v in range(k):
            e1 += x1[r, v]*theta[v]
            e2 += x2[r, v]*theta[v]
        mx = max(0., e1, e2)
        den = np.exp(-mx)+np.exp(e1-mx)+np.exp(e2-mx)
        p1, p2 = np.exp(e1-mx)/den, np.exp(e2-mx)/den
        val += w[r]*(mx+np.log(den)-(e1 if y[r] == 1 else (e2 if y[r] == 2 else 0.)))
        for v in range(k):
            grad[v] += w[r]*((p1-(y[r] == 1))*x1[r, v]+(p2-(y[r] == 2))*x2[r, v])
            for z in range(k):
                H[v, z] += w[r]*(p1*(1-p1)*x1[r, v]*x1[r, z]+p2*(1-p2)*x2[r, v]*x2[r, z]
                                             -p1*p2*(x1[r, v]*x2[r, z]+x2[r, v]*x1[r, z]))
    return val/total_weight, grad/total_weight, H/total_weight

@njit(cache=True)
def dyad_projection(theta, x1, x2, y, w, nodes, n):
    k = len(theta)
    Q = np.zeros((n*(n-1)//2, k))
    for r in range(len(y)):
        e1, e2 = 0., 0.
        for v in range(k):
            e1 += x1[r, v]*theta[v]
            e2 += x2[r, v]*theta[v]
        mx = max(0., e1, e2)
        den = np.exp(-mx)+np.exp(e1-mx)+np.exp(e2-mx)
        p1, p2 = np.exp(e1-mx)/den, np.exp(e2-mx)/den
        for a in range(3):
            i = nodes[r, a]
            for b in range(a+1, 4):
                j = nodes[r, b]
                d = i*(2*n-i-1)//2+j-i-1
                for v in range(k):
                    Q[d, v] += w[r]*((p1-(y[r] == 1))*x1[r, v]+(p2-(y[r] == 2))*x2[r, v])
    return Q

def _ess(chains):
    m, n = chains.shape
    W = np.var(chains, axis=1, ddof=1).mean()
    between = n*np.var(chains.mean(axis=1), ddof=1)
    varplus = (n-1)/n*W+between/n
    if varplus <= 0:
        return 0.
    centered = chains-chains.mean(axis=1, keepdims=True)
    size = 1 << (2*n-1).bit_length()
    fourier = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(fourier*np.conj(fourier), n=size, axis=1)[:, :n]/n
    rho = 1-(W-acov.mean(axis=0))/varplus
    rho[0] = 1.
    pairs = []
    for j in range(0, n-1, 2):
        pair = rho[j]+rho[j+1]
        if pair < 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    tau = max(1., -1+2*sum(pairs))
    return float(min(m*n, m*n/tau))

def diagnostics(trace, names=None):
    from scipy.stats import rankdata
    m, draws, q = trace.shape
    if names is None:
        names = ["density", "mutual_fraction", "T_normalized", "closure_rate"]
    assert len(names) == q
    half = draws//2
    split = np.concatenate([trace[:, :half], trace[:, -half:]], axis=0)
    def rhat(x):
        W = np.var(x, axis=1, ddof=1).mean()
        B = half*np.var(x.mean(axis=1), ddof=1)
        return float(np.sqrt(((half-1)/half*W+B/half)/W)) if W > 0 else None
    rows = []
    for k in range(q):
        x = split[:, :, k]
        def normal(v):
            ranks = rankdata(v.ravel()).reshape(v.shape)
            return ndtri((ranks-.375)/(v.size+.25))
        z, folded = normal(x), normal(np.abs(x-np.median(x)))
        rh = [rhat(z), rhat(folded)]
        ess = _ess(z)
        lo, hi = np.quantile(x, [.05, .95])
        tail = min(_ess((x <= lo).astype(float)), _ess((x >= hi).astype(float)))
        raw_ess = _ess(x)
        rows.append(dict(statistic=names[k],
                         rhat=max(rh) if all(v is not None for v in rh) else None,
                         bulk_ess=ess, tail_ess=tail,
                         mean=float(x.mean()), sd=float(x.std(ddof=1)),
                         mean_mcse=float(x.std(ddof=1)/np.sqrt(raw_ess)) if raw_ess > 0 else None))
    return rows

def energy(A, d, delta, mu_shift=0.):
    eta = d["mu"]+mu_shift+d["alpha"][:, None]+d["beta"][None, :]
    i, j = np.triu_indices(len(A), 1)
    return float(np.sum(A*eta)+np.sum(A[i, j]*A[j, i]*(d["theta"][0]+d["V"][i, j]@d["theta"][1:]))
                 +delta*raw_triples(A)/(len(A)-2))
@njit(cache=True)
def toggle_dyad(A,C,i,j,a,b):
    d1,d2=a-int(A[i,j]),b-int(A[j,i])
    if d1==0:
        return toggle(A,C,j,i,b) if d2 else 0
    if d2==0:return toggle(A,C,i,j,a)
    change=d1*C[i,j]+d2*C[j,i]
    for k in range(len(A)):
        if k!=i and k!=j:
            aj,ja,ai,ia=int(A[j,k]),int(A[k,j]),int(A[i,k]),int(A[k,i])
            C[i,k]+=d1*(aj+ja)+d2*aj
            C[j,k]+=d2*(ai+ia)+d1*ai
            C[k,i]+=d1*ja+d2*(ja+aj)
            C[k,j]+=d2*ia+d1*(ia+ai)
    A[i,j],A[j,i]=a,b
    return change

@njit(cache=True)
def gibbs(A_input,eta,reciprocity,delta,draws,burn_sweeps,interval_sweeps,seed):
    np.random.seed(seed)
    A=A_input.copy();n=len(A)
    C=change_matrix(A);triples=raw_triples(A)
    edges,mutual=int(A.sum()),0
    for i in range(n):
        for j in range(i+1,n):mutual+=int(A[i,j])*int(A[j,i])
    trace=np.zeros((draws,4));saved=0
    for sweep in range(burn_sweeps+draws*interval_sweeps):
        for block in range(n*(n-1)//2):
            i=np.random.randint(n);j=np.random.randint(n-1)
            if j>=i:j+=1
            e1=eta[i,j]+delta*C[i,j]/(n-2)
            e2=eta[j,i]+delta*C[j,i]/(n-2)
            e3=e1+e2+reciprocity[i,j]
            mx=max(0.,e1,e2,e3)
            w0,w1,w2,w3=np.exp(-mx),np.exp(e1-mx),np.exp(e2-mx),np.exp(e3-mx)
            u=np.random.random()*(w0+w1+w2+w3)
            s=0 if u<w0 else (1 if u<w0+w1 else (2 if u<w0+w1+w2 else 3))
            a,b=int(s==1 or s==3),int(s==2 or s==3)
            olda,oldb=int(A[i,j]),int(A[j,i])
            edges+=a+b-olda-oldb;mutual+=a*b-olda*oldb
            triples+=toggle_dyad(A,C,i,j,a,b)
        if sweep>=burn_sweeps and (sweep-burn_sweeps+1)%interval_sweeps==0:
            paths=0
            for i in range(n):
                out,inc=0,0
                for j in range(n):out+=int(A[i,j]);inc+=int(A[j,i])
                paths+=out*inc
            paths-=2*mutual
            trace[saved]=np.array([edges/(n*(n-1)),mutual/(n*(n-1)/2),triples/(n-2),triples/max(1,paths)])
            saved+=1
    assert triples==raw_triples(A)
    assert np.all(C==change_matrix(A))
    return A,trace
@njit(cache=True)
def enumerate_rows(A,V,fill,x1,x2,y,w,nodes):
    n,q,count=len(A),V.shape[2],0
    for i in range(n-3):
        for j in range(i+1,n):
            s1=int(A[i,j])+2*int(A[j,i])
            required=(3 if s1==0 else (0 if s1==3 else s1))
            for k in range(i+1,n):
                if k==j or int(A[j,k])+2*int(A[k,j])!=required:
                    continue
                for l in range(j+1,n):
                    if l==k:
                        continue
                    if int(A[k,l])+2*int(A[l,k])!=s1:
                        continue
                    if int(A[l,i])+2*int(A[i,l])!=required:
                        continue
                    if fill:
                        x1[count,0]=2.
                        x2[count,0]=2.
                        for h in range(q):
                            x1[count,h+1]=V[i,j,h]+V[k,l,h]
                            x2[count,h+1]=V[i,l,h]+V[j,k,h]
                        y[count]=2 if s1==0 else (1 if s1==3 else 0)
                        w[count]=8. if s1==0 or s1==3 else 4.
                        a,b=min(j,k,l),max(j,k,l)
                        nodes[count,0]=i
                        nodes[count,1]=a
                        nodes[count,2]=j+k+l-a-b
                        nodes[count,3]=b
                    count+=1
    return count

def tetrad_design(A,V):
    dummy=np.empty((0,V.shape[2]+1))
    m=enumerate_rows(A,V,False,dummy,dummy,np.empty(0,np.int8),np.empty(0),np.empty((0,4),np.int32))
    if m==0:raise ValueError("No identifying tetrad contributions")
    x1,x2=np.empty((m,V.shape[2]+1)),np.empty((m,V.shape[2]+1))
    y,w,nodes=np.empty(m,np.int8),np.empty(m),np.empty((m,4),np.int32)
    assert enumerate_rows(A,V,True,x1,x2,y,w,nodes)==m
    return x1,x2,y,w,nodes

@njit(cache=True)
def objective3(theta,x1,x2,y,w):
    value=0.
    gradient=np.zeros(3)
    H=np.zeros((3,3))
    for r in range(len(y)):
        a,c=x1[r,1],x1[r,2]
        b,d=x2[r,1],x2[r,2]
        e1=2*theta[0]+a*theta[1]+c*theta[2]
        e2=2*theta[0]+b*theta[1]+d*theta[2]
        mx=max(0.,e1,e2)
        z0,z1,z2=np.exp(-mx),np.exp(e1-mx),np.exp(e2-mx)
        den=z0+z1+z2
        p0,p1,p2=z0/den,z1/den,z2/den
        weight=w[r]
        value+=weight*(mx+np.log(den)-(e1 if y[r]==1 else (e2 if y[r]==2 else 0.)))
        gradient[0]+=weight*2*(p1+p2-(y[r]!=0))
        gradient[1]+=weight*(p1*a+p2*b-(a if y[r]==1 else (b if y[r]==2 else 0.)))
        gradient[2]+=weight*(p1*c+p2*d-(c if y[r]==1 else (d if y[r]==2 else 0.)))
        H[0,0]+=weight*4*p0*(p1+p2)
        H[0,1]+=weight*2*p0*(p1*a+p2*b)
        H[0,2]+=weight*2*p0*(p1*c+p2*d)
        H[1,1]+=weight*(p0*(p1*a*a+p2*b*b)+p1*p2*(a-b)**2)
        H[1,2]+=weight*(p0*(p1*a*c+p2*b*d)+p1*p2*(a-b)*(c-d))
        H[2,2]+=weight*(p0*(p1*c*c+p2*d*d)+p1*p2*(c-d)**2)
    H[1,0],H[2,0],H[2,1]=H[0,1],H[0,2],H[1,2]
    scale=w.sum()
    return value/scale,gradient/scale,H/scale

@njit(cache=True)
def projection3(theta,x1,x2,y,w,nodes,n):
    Q=np.zeros((n*(n-1)//2,3))
    for r in range(len(y)):
        a,c=x1[r,1],x1[r,2]
        b,d=x2[r,1],x2[r,2]
        e1=2*theta[0]+a*theta[1]+c*theta[2]
        e2=2*theta[0]+b*theta[1]+d*theta[2]
        mx=max(0.,e1,e2)
        z0,z1,z2=np.exp(-mx),np.exp(e1-mx),np.exp(e2-mx)
        den=z0+z1+z2
        p1,p2=z1/den,z2/den
        s0=w[r]*2*(p1+p2-(y[r]!=0))
        s1=w[r]*(p1*a+p2*b-(a if y[r]==1 else (b if y[r]==2 else 0.)))
        s2=w[r]*(p1*c+p2*d-(c if y[r]==1 else (d if y[r]==2 else 0.)))
        for h in range(3):
            i=nodes[r,h]
            for z in range(h+1,4):
                j=nodes[r,z]
                k=i*(2*n-i-1)//2+j-i-1
                Q[k,0]+=s0
                Q[k,1]+=s1
                Q[k,2]+=s2
    return Q

def fit_tetrad(A,V,ci=True):
    start=time.monotonic()
    x1,x2,y,w,nodes=tetrad_design(A,V)
    objective=objective3 if V.shape[2]==2 else core.objective
    cache=[None,None]
    def evaluate(t):
        if cache[0] is None or not np.array_equal(cache[0],t):
            cache[:]=[t.copy(),objective(t,x1,x2,y,w)]
        return cache[1]
    result=minimize(lambda t:evaluate(t)[0],np.zeros(V.shape[2]+1),
                    jac=lambda t:evaluate(t)[1],hess=lambda t:evaluate(t)[2],
                    method="trust-exact",options={"gtol":1e-8,"maxiter":100})
    value,gradient,H=evaluate(result.x)
    eig=np.linalg.eigvalsh(H).min()
    if np.max(np.abs(gradient))>1e-6 or eig<1e-10 or not np.isfinite(result.x).all():
        raise RuntimeError(f"Unstable tetrad fit: {result.message}; score={gradient}; curvature={eig}")
    n=len(A)
    keys=((nodes[:,0].astype(np.int64)*n+nodes[:,1])*n+nodes[:,2])*n+nodes[:,3]
    answer=dict(theta=result.x,gradient_max=float(np.max(np.abs(gradient))),
                hessian_min_eigenvalue=float(eig),objective=float(value),
                identifying_tetrads=int(len(np.unique(keys))),perimeter_rows=len(y),
                original_contributions=int(w.sum()),configurations=np.bincount(y,weights=w,minlength=3),
                iterations=int(result.nit))
    if ci:
        project=projection3 if V.shape[2]==2 else core.dyad_projection
        Q=project(result.x,x1,x2,y,w,nodes,n)
        Q-=Q.mean(axis=0)
        inverse=np.linalg.inv(H*w.sum())
        N,k=len(Q),len(result.x)
        cov=N/(N-k)*inverse@(Q.T@Q)@inverse
        se=np.sqrt(np.diag(cov))
        if not np.isfinite(se).all() or np.any(se<=0):raise RuntimeError("Invalid standard errors")
        answer.update(se=se,covariance=cov,lower=result.x-1.96*se,upper=result.x+1.96*se)
    answer["seconds"]=time.monotonic()-start
    return answer
@njit(cache=True)
def reciprocity_stats(A, V):
    s = np.zeros(V.shape[2]+1)
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i, j] and A[j, i]:
                s[0] += 1
                for k in range(V.shape[2]):
                    s[k+1] += V[i, j, k]
    return s

@njit(cache=True)
def fiber_chain(input_A, V, theta, draws, burn, interval, seed, save_graphs=True, mutual_proposal=False, graph_stride=1):
    np.random.seed(seed)
    A = input_A.copy()
    n, q = len(A), len(theta)
    initial_out, initial_in = A.sum(axis=1), A.sum(axis=0)
    s = reciprocity_stats(A, V)
    trace = np.zeros((draws, q+2))
    graphs = np.zeros(((draws+graph_stride-1)//graph_stride if save_graphs else 0, n, n), np.int8)
    counts = np.zeros(8, np.int64)
    tail, head = np.zeros(8, np.int64), np.zeros(8, np.int64)
    pairs_u, pairs_v = np.zeros(8, np.int64), np.zeros(8, np.int64)
    at = 0
    for step in range(burn+draws*interval):
        counts[0] += 1
        if np.random.random() < .05:
            counts[1] += 1
        else:
            choice = np.random.random()
            switch = choice < (.6 if mutual_proposal else .8)
            mutual_switch = mutual_proposal and .6 <= choice < .85
            valid = False
            nt = 0
            if switch and n >= 4:
                a = np.random.randint(n)
                c = np.random.randint(n-1)
                if c >= a:
                    c += 1
                b, d = -1, -1
                for attempt in range(16):
                    candidate = np.random.randint(n)
                    if candidate != a and candidate != c and A[a, candidate] == 1 and A[c, candidate] == 0:
                        b = candidate
                        break
                if b >= 0:
                    for attempt in range(16):
                        candidate = np.random.randint(n)
                        if candidate != a and candidate != c and A[a, candidate] == 0 and A[c, candidate] == 1:
                            d = candidate
                            break
                if b >= 0 and d >= 0:
                    valid, nt = True, 4
                    tail[0], head[0] = a, b
                    tail[1], head[1] = c, d
                    tail[2], head[2] = a, d
                    tail[3], head[3] = c, b
            elif mutual_switch and n >= 4:
                a = np.random.randint(n)
                c = np.random.randint(n-1)
                if c >= a:
                    c += 1
                b, d = -1, -1
                for attempt in range(16):
                    candidate = np.random.randint(n)
                    if candidate != a and candidate != c and A[a, candidate] == 1 and A[candidate, a] == 1 and A[c, candidate] == 0 and A[candidate, c] == 0:
                        b = candidate
                        break
                if b >= 0:
                    for attempt in range(16):
                        candidate = np.random.randint(n)
                        if candidate != a and candidate != c and A[c, candidate] == 1 and A[candidate, c] == 1 and A[a, candidate] == 0 and A[candidate, a] == 0:
                            d = candidate
                            break
                if b >= 0 and d >= 0:
                    valid, nt = True, 8
                    tail[0], head[0] = a, b
                    tail[1], head[1] = b, a
                    tail[2], head[2] = c, d
                    tail[3], head[3] = d, c
                    tail[4], head[4] = a, d
                    tail[5], head[5] = d, a
                    tail[6], head[6] = c, b
                    tail[7], head[7] = b, c
            elif not switch and not mutual_switch:
                a, b = np.random.randint(n), np.random.randint(n)
                while b == a:
                    b = np.random.randint(n)
                c = np.random.randint(n)
                while c == a or c == b:
                    c = np.random.randint(n)
                valid = (A[a, b] == A[b, c] and A[b, c] == A[c, a] and
                         A[b, a] == A[c, b] and A[c, b] == A[a, c] and A[a, b] != A[b, a])
                if valid:
                    nt = 6
                    tail[0], head[0] = a, b
                    tail[1], head[1] = b, c
                    tail[2], head[2] = c, a
                    tail[3], head[3] = b, a
                    tail[4], head[4] = c, b
                    tail[5], head[5] = a, c
            if not valid:
                counts[2] += 1
            else:
                counts[3] += 1
                counts[5 if switch else (7 if mutual_switch else 6)] += 1
                npairs = 0
                for z in range(nt):
                    u, v = min(tail[z], head[z]), max(tail[z], head[z])
                    seen = False
                    for h in range(npairs):
                        if pairs_u[h] == u and pairs_v[h] == v:
                            seen = True
                    if not seen:
                        pairs_u[npairs], pairs_v[npairs] = u, v
                        npairs += 1
                diff = np.zeros(q)
                for z in range(npairs):
                    u, v = pairs_u[z], pairs_v[z]
                    if A[u, v] and A[v, u]:
                        diff[0] -= 1
                        for h in range(q-1):
                            diff[h+1] -= V[u, v, h]
                for z in range(nt):
                    A[tail[z], head[z]] = 1-A[tail[z], head[z]]
                for z in range(npairs):
                    u, v = pairs_u[z], pairs_v[z]
                    if A[u, v] and A[v, u]:
                        diff[0] += 1
                        for h in range(q-1):
                            diff[h+1] += V[u, v, h]
                ratio = 0.
                for h in range(q):
                    ratio += theta[h]*diff[h]
                if np.log(np.random.random()) < min(0., ratio):
                    s += diff
                    counts[4] += 1
                else:
                    for z in range(nt):
                        A[tail[z], head[z]] = 1-A[tail[z], head[z]]
        if step >= burn and (step-burn+1) % interval == 0:
            assert np.max(np.abs(s-reciprocity_stats(A, V))) < 1e-7
            trace[at, :q] = s
            tt = raw_triples(A)
            paths = 0
            for i in range(n):
                paths += initial_out[i]*initial_in[i]
            paths -= 2*s[0]
            trace[at, q] = tt/(n-2)
            trace[at, q+1] = tt/max(1., paths)
            if save_graphs and at % graph_stride == 0:
                graphs[at//graph_stride] = A
            at += 1
    assert np.all(A.sum(axis=1) == initial_out)
    assert np.all(A.sum(axis=0) == initial_in)
    return A, trace, graphs, counts
