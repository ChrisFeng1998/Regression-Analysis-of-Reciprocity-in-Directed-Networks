"""Read the published application files and construct arrays in memory.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def _file(raw_dir, folder, filename):
    root=Path(raw_dir)
    for path in [root/folder/filename,root/filename]:
        if path.is_file():return path
    raise FileNotFoundError(f'Place {filename} in {root/folder} (or directly in {root}). See README data links.')


def _standardize(x):
    x=np.asarray(x,float)
    sd=x.std(ddof=1)
    if not np.isfinite(sd) or sd==0:raise ValueError('Constant or nonfinite continuous covariate')
    return (x-x.mean())/sd


def _arrays(A,V,X,node_ids):
    n=len(A)
    if A.shape!=(n,n) or not np.isin(A,[0,1]).all() or np.any(np.diag(A)):
        raise ValueError('Expected a binary directed adjacency matrix without self-links')
    if V.shape[:2]!=(n,n) or X.shape[0]!=n or not np.isfinite(V).all() or not np.isfinite(X).all():
        raise ValueError('Invalid node or dyad covariate dimensions/values')
    if not np.allclose(V,V.transpose(1,0,2),atol=1e-12,rtol=0):
        raise ValueError('Reciprocity covariates must be symmetric across directions')
    i,j=np.triu_indices(n,1)
    return dict(A=A.astype(np.int8),V=V,X=X,i=i,j=j,node_ids=np.asarray(node_ids))


def load_lawyer(raw_dir):
    folder='LazegaLawyers'
    attr=pd.read_csv(_file(raw_dir,folder,'ELattr.dat'),sep=r'\s+',header=None)
    attr.columns=['seniority','status','gender','office','tenure','age','practice','law_school']
    A=pd.read_csv(_file(raw_dir,folder,'ELadv.dat'),sep=r'\s+',header=None).to_numpy()
    n=len(attr)
    if A.shape==(n,n+1) and np.array_equal(A[:,0],np.arange(1,n+1)):
        A=A[:,1:]  # Also accept a text file with explicit row labels.
    if A.shape!=(n,n):raise ValueError('Advice adjacency and lawyer attributes have different node counts')
    cats=['status','gender','office','practice','law_school']
    V=np.stack([(attr[c].to_numpy()[:,None]==attr[c].to_numpy()[None,:]).astype(float) for c in cats],axis=2)
    design=pd.get_dummies(attr[cats].astype(str),drop_first=True,dtype=float)
    design['age']=_standardize(attr['age'])
    design['tenure']=_standardize(attr['tenure'])
    X=design.to_numpy(float);X-=X.mean(axis=0)
    data=_arrays(A,V,X,np.arange(1,n+1))
    data.update(status=attr['status'].to_numpy(int),
                v_names=['same_'+c for c in cats],x_names=design.columns.tolist())
    return data


def load_trade(raw_dir):
    folder='trade_network'
    gravity=pd.read_excel(_file(raw_dir,folder,'Log of Gravity.xls'))
    codes=pd.read_excel(_file(raw_dir,folder,'countrycodes.xls')).iloc[:,0].tolist()
    if len(set(codes))!=len(codes):raise ValueError('Duplicate country codes')
    if gravity.duplicated(['s1_im','s2_ex']).any():raise ValueError('Duplicate directed trade pairs')
    n=len(codes)
    # A row exporter -> column importer; threshold is 1% of the exporter's total.
    trade=gravity.pivot(index='s2_ex',columns='s1_im',values='trade').reindex(index=codes,columns=codes)
    off_diagonal=~np.eye(n,dtype=bool)
    if trade.to_numpy()[off_diagonal].size and trade.isna().to_numpy()[off_diagonal].any():
        raise ValueError('Missing trade values for potential off-diagonal country pairs')
    if (trade.to_numpy()[off_diagonal]<0).any():raise ValueError('Negative trade value')
    total_exports=gravity.groupby('s2_ex')['trade'].transform('sum')
    link=(gravity['trade']>=.01*total_exports).astype(int)
    A=gravity.assign(link=link).pivot(index='s2_ex',columns='s1_im',values='link').reindex(index=codes,columns=codes).fillna(0).to_numpy()
    np.fill_diagonal(A,0)
    labels=['ldist','border','comlang','colony','comfrt']
    V=np.stack([gravity.pivot(index='s2_ex',columns='s1_im',values=c).reindex(index=codes,columns=codes).to_numpy(float) for c in labels],axis=2)
    for k in range(len(labels)):np.fill_diagonal(V[:,:,k],0)
    i,j=np.triu_indices(n,1)
    mean,sd=V[i,j,0].mean(),V[i,j,0].std(ddof=1)
    if not np.isfinite(sd) or sd==0:raise ValueError('Invalid log-distance scale')
    V[:,:,0]=(V[:,:,0]-mean)/sd;np.fill_diagonal(V[:,:,0],0)
    nodal=gravity.groupby('s2_ex')[['lypex','landl_ex']].mean().reindex(codes)
    X=np.column_stack([_standardize(nodal['lypex']),nodal['landl_ex']-nodal['landl_ex'].mean()])
    data=_arrays(A,V,X,codes)
    data.update(v_names=labels,x_names=['log_gdp','landlocked'])
    return data


def load_application(name,raw_dir):
    if name=='lawyer':return load_lawyer(raw_dir)
    if name=='trade':return load_trade(raw_dir)
    raise ValueError(f'Unknown application: {name}')
