
import os, tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"jasa_visualization_mpl"))
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=["0.2_0.1","0.2_-0.1","0.3_0.5"]
SIZES=[50,100,150,200]
PARAMETERS=["rho","gamma1","gamma2"]


def draw_comparison(errors_method1,errors_method2,n_values,legend_labels,xlabel,
                    ylim,path,legend_location="upper right",bbox=False):

    positions1 = np.arange(len(n_values)) * 2.0         
    positions2 = positions1 + 0.6                       

    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot for method 1
    bp1 = ax.boxplot(errors_method1, positions=positions1, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='darkblue'))

    # Boxplot for method 2 (MLE)
    bp2 = ax.boxplot(errors_method2, positions=positions2, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', color='green'),
                     medianprops=dict(color='darkgreen'))

    ax.set_xticks(positions1 + 0.3)
    ax.set_xticklabels(n_values, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel('Absolute error', fontsize=22)
    ax.tick_params(axis='y', labelsize=20)

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], legend_labels, fontsize=15, loc=legend_location)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(path,format="pdf",**({"bbox_inches":"tight"} if bbox else {}))
    plt.close(fig)


def read_result(path, **kwargs):
    if not path.exists():path=path.with_suffix("")  
    return pd.read_csv(path, **kwargs)


def plot_all(out, input_dir):
    DATA=input_dir
    matplotlib.rcdefaults()
    out.mkdir(parents=True,exist_ok=True)
    mle_limits={"0.2_0.1":[.55,.55,.45],"0.2_-0.1":[.72,.52,.52],"0.3_0.5":[1.15,.65,.45]}
    p15_limits={"0.2_0.1":[.92,.65,.55],"0.2_-0.1":[1.05,.75,.75],"0.3_0.5":[.85,.65,.55]}
    mis_limits={"0.2_0.1":[.52,.52,.52],"0.2_-0.1":[.62,.52,.52],"0.3_0.5":[.82,.52,.52]}
    summary=[]
    for setting in SETTINGS:
        b=float(setting.split("_")[1])
        cl=[read_result(DATA/f"CL/{setting}/{n}.csv.gz") for n in SIZES]
        mle=[read_result(DATA/f"MLE/{setting}/{n}.csv.gz",header=None) for n in SIZES]
        mom=[read_result(DATA/f"p15_comparison/CL/{setting}/{n}.csv.gz") for n in SIZES]
        p15=[read_result(DATA/f"p15_comparison/p15/{setting}/{n}.csv.gz") for n in SIZES]
        for j,p in enumerate(PARAMETERS):
            ecl=[np.abs(f.iloc[:,j]-(.5+b*np.log(n) if j==0 else 1.)) for n,f in zip(SIZES,cl)]
            emle=[np.abs(f.iloc[:,j]-f.iloc[:,3]) if j==0 else np.abs(f.iloc[:,j]-1.) for f in mle]
            draw_comparison(ecl,emle,SIZES,["Conditional likelihood","Maximum likelihood"],
                "Network size",(-.02,mle_limits[setting][j]),out/f"comparison_{setting}_error_{p}.pdf",
                bbox=(setting=="0.2_-0.1" and j==2))
            # These files contain saved absolute errors, not raw coefficients.
            ecl_mis=[f.iloc[:,j] for f in mom]
            col=[2,4,5][j]
            # Use all available simulation results for every setting and parameter.
            ep15=[f.iloc[:,col] for f in p15]
            draw_comparison(ecl_mis,ep15,SIZES,
                ["Conditional likelihood in proposed model" if setting=="0.2_-0.1" and j==0 else "Conditional likelihood",
                 r"Maximum likelihood in $p_{1.5}$ model"],
                "Network size",(-.02,p15_limits[setting][j]),out/f"comparison_p1.5_{setting}_error_{p}.pdf")
            for family,methods in [("MLE",[("CL",ecl),("MLE",emle)]),("p15",[("CL",ecl_mis),("p15",ep15)])]:
                for method,errors in methods:
                    for n,values in zip(SIZES,errors):
                        assert np.isfinite(values).all() and np.all(values>=0)
                        summary.append(dict(family=family,setting=setting,parameter=p,method=method,
                                            n=n,N=len(values),MAE=np.mean(values)))
        rates=["0","0.25","0.5","0.75","1"]
        cl_mis=[read_result(DATA/f"misspecification/CL/{setting}/{r}.csv.gz") for r in rates]
        p15_mis=[read_result(DATA/f"misspecification/p15/{setting}/{r}.csv.gz") for r in rates]
        for j,p in enumerate(PARAMETERS):
            # CL: first four files contain estimates + errors; rate=1 has errors only.
            ecl=[f.iloc[:,j+3 if index<4 else j] for index,f in enumerate(cl_mis)]
            # In this experiment rho error is column 1, unlike the size comparison above.
            ep15=[f.iloc[:,[1,4,5][j]] for f in p15_mis]
            draw_comparison(ecl,ep15,[0,.25,.5,.75,1],
                ["Conditional likelihood",r"Maximum likelihood in $p_{1.5}$ model"],
                "Misspecification rate",(-.02,mis_limits[setting][j]),out/f"mis_{setting}_{p}.pdf",
                legend_location="upper left",bbox=True)
            for method,errors in [("CL",ecl),("p15",ep15)]:
                for rate,values in zip(rates,errors):
                    assert np.isfinite(values).all() and np.all(values>=0)
                    summary.append(dict(family="misspecification",setting=setting,parameter=p,method=method,
                                        n=100,rate=float(rate),N=len(values),MAE=np.mean(values)))
    pd.DataFrame(summary).to_csv(out/"simulation_plot_summary.csv",index=False)
    print(f"Generated 27 simulation/misspecification PDF panels in {out}; no models fitted.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True,help="Your simulation-result directory; see README")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/visualization/simulations")
    args=parser.parse_args()
    plot_all(args.output_dir,args.input_dir)
