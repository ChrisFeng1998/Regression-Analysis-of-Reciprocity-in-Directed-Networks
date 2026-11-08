"""Fit the original empirical models directly from downloaded raw files."""
import argparse
from pathlib import Path
from estimators import CLL
from data_analysis.raw_data import load_application

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=ROOT,help='Directory containing LazegaLawyers/ and trade_network/')
    parser.add_argument('--dataset',choices=['lawyer','trade'],help='Default: both networks')
    args=parser.parse_args()
    for name in ([args.dataset] if args.dataset else ['lawyer','trade']):
        data=load_application(name,args.raw_dir)
        A,V=data['A'],data['V']
        print(f'{name}: {len(A)} nodes, {int(A.sum())} directed edges',flush=True)
        print('Constant reciprocity:',flush=True)
        CLL.tetrad_con_logit(A,[])
        print('Covariate-dependent reciprocity:',flush=True)
        CLL.tetrad_con_logit(A,[V[:,:,k] for k in range(V.shape[2])])


if __name__=='__main__':main()
