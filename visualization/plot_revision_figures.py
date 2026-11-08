"""Replot KS summary (S21), lawyer network (Figure 8), and trade (Figure 9).
"""
from pathlib import Path
import argparse, json, os, tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"jasa_visualization_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from visualization import generate_qq_plots as qq
from data_analysis.raw_data import load_application
from estimators.transitivity_core import fit_tetrad

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir",type=Path,default=ROOT,help="Directory containing the downloaded raw datasets")
    parser.add_argument("--qq-input-dir",type=Path,help="Optional simulation CL CSV directory for the KS summary")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/visualization/applications")
    args=parser.parse_args()
    OUT=args.output_dir;TABLES=OUT/"tables"
    OUT.mkdir(parents=True,exist_ok=True);TABLES.mkdir(parents=True,exist_ok=True)
    # Source plotting commands below retain their original styles.
    plt.rcParams.update({'font.family':'DejaVu Sans', 'mathtext.fontset':'dejavusans',
        'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10,
        'xtick.labelsize':9, 'ytick.labelsize':9, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.7, 'pdf.fonttype':42,
        'savefig.facecolor':'white'})
    COLORS = ['#0072B2','#D55E00','#009E73']
    MARKERS = ['o','s','^']
    LABELS = [r'$\rho_n$',r'$\gamma_1$',r'$\gamma_2$']

    def save(fig, name):
        fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(OUT/f'{name}.png', bbox_inches='tight', dpi=220)
        plt.close(fig)


    if args.qq_input_dir is not None:
        panels, qq_manifest, qq_checks = qq.load_panels(args.qq_input_dir)
        records = []
        for setting in qq.SETTINGS:
            for n in qq.SIZES:
                for parameter, label in qq.PARAMETERS:
                    panel = panels[(setting['directory'],n,parameter)]
                    result = stats.kstest(panel['z'], 'norm')
                    z = panel['ordered']; B = len(z); F = stats.norm.cdf(z)
                    independent_D = max(np.max(np.arange(1,B+1)/B-F), np.max(F-np.arange(B)/B))
                    assert abs(result.statistic-independent_D) < 1e-14
                    records.append({'figure':setting['figure'],'a':setting['a'],'b':setting['b'],
                        'n':n,'parameter':parameter,'B':B,'KS_distance':float(independent_D),
                        'mean_studentized':float(np.mean(z)), 'sd_studentized':float(np.std(z,ddof=1))})
        ks = pd.DataFrame(records)
        ks.to_csv(TABLES/'Figure4_all_settings_KS.csv',index=False)
        upper = float(ks.KS_distance.max()*1.2)

        def ks_axis(ax, frame, mode='line', legend=True):
            for j, (parameter,label) in enumerate(qq.PARAMETERS):
                f = frame[frame.parameter==parameter].sort_values('n')
                if mode=='line':
                    ax.plot(f.n,f.KS_distance,color=COLORS[j],marker=MARKERS[j],label=label,
                            linewidth=1.4,markersize=5)
                else:
                    ax.bar(np.arange(4)+(j-1)*.23,f.KS_distance,width=.22,color=COLORS[j],label=label)
            if mode=='line': ax.set_xticks(qq.SIZES)
            else: ax.set_xticks(np.arange(4),qq.SIZES)
            ax.set(xlabel=r'Network size $n$',ylabel=r'KS distance to $N(0,1)$',ylim=(0,upper))
            ax.grid(axis='y',color='#e8e8e8',linewidth=.6)
            ax.set_axisbelow(True)
            if legend: ax.legend(frameon=False,ncol=3,loc='upper right')


        fig, axes = plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
        for ax, setting in zip(axes,qq.SETTINGS):
            ks_axis(ax,ks[ks.figure==setting['figure']],legend=False)
            ax.set_title(rf"$a={setting['a']},\ b={setting['b']}$")
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
        save(fig,'Figure4_KS_all_settings')


    law = load_application('lawyer',args.raw_dir)
    A = law['A']; n = len(A)
    status = law['status']
    assert len(status)==n and dict(zip(*np.unique(status,return_counts=True)))=={1:36,2:35}
    mutual = [(i,j) for i in range(n) for j in range(i+1,n) if A[i,j] and A[j,i]]
    oneway = [(i,j) for i,j in zip(*np.where(A)) if not A[j,i]]
    assert len(mutual)==175 and len(oneway)==542 and A.sum()==892
    G = nx.from_numpy_array(A,create_using=nx.DiGraph)
    pos = nx.spring_layout(G,seed=42,iterations=100)
    pd.DataFrame([{'node_id':i+1,'status':int(status[i]),'x':pos[i][0],'y':pos[i][1]} for i in range(n)]).to_csv(TABLES/'Figure8_node_layout.csv',index=False)
    pd.DataFrame([{'source':i+1,'target':j+1,'type':'one_way'} for i,j in oneway]+[{'source':i+1,'target':j+1,'type':'mutual_pair'} for i,j in mutual]).to_csv(TABLES/'Figure8_edges.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,6.8),layout='constrained')
    nx.draw_networkx_edges(G,pos,ax=ax,edgelist=oneway,edge_color='#9da4ab',alpha=.30,
        width=.55,arrows=True,arrowsize=6,node_size=65,connectionstyle='arc3,rad=0.035')
    nx.draw_networkx_edges(G,pos,ax=ax,edgelist=mutual,edge_color='#303846',alpha=.75,
        width=1.15,arrows=False,node_size=65)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_size=65,node_color=[COLORS[0] if s==1 else COLORS[1] for s in status],edgecolors='white',linewidths=.65)
    ax.set_axis_off()
    ax.set_title('Advice network: mutual ties highlighted',pad=7)
    legend=[Line2D([],[],marker='o',color='none',markerfacecolor=COLORS[0],markeredgecolor='white',markersize=8,label='Partner (36)'),
            Line2D([],[],marker='o',color='none',markerfacecolor=COLORS[1],markeredgecolor='white',markersize=8,label='Associate (35)'),
            Line2D([],[],color='#303846',linewidth=1.8,label='Mutual dyad (175 pairs)'),
            Line2D([],[],color='#9da4ab',linewidth=.8,marker='>',markersize=4,label='One-way tie (542 arcs)')]
    ax.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=2,frameon=False,fontsize=9)
    ax.margins(.08)
    save(fig,'Figure8_mutual_highlight')


    trade=load_application('trade',args.raw_dir)
    ii,jj=trade['i'],trade['j'];V=trade['V'];A=trade['A']
    # Both are unpenalized full-network fits, computed from the raw data here.
    theta=fit_tetrad(A,V,ci=False)['theta']
    constant=float(fit_tetrad(A,V[:,:,:0],ci=False)['theta'][0])
    rho=theta[0]+V[ii,jj]@theta[1:]
    state=A[ii,jj]+2*A[jj,ii]
    assert len(rho)==9180 and A.sum()==1880 and np.count_nonzero(state==3)==234
    trade_frame=pd.DataFrame(dict(i=ii,j=jj,state=state,standardized_log_distance=V[ii,jj,0],
                                PTA=V[ii,jj,4].astype(int),estimated_reciprocity=rho))
    trade_frame.to_csv(TABLES/'Figure9_all_9180_dyads.csv',index=False)
    matplotlib.rcdefaults()
   
    mutual_log=V[ii,jj,0][state==3].tolist()
    one_log=V[ii,jj,0][(state==1)|(state==2)].tolist()


    data = [mutual_log, one_log]
    labels = ['Mutual link', 'One-way link']

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red'))
    #plt.xlabel(fontsize=22)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Standardized log distance', fontsize=22)
    #plt.title('Comparison of Variable by Group')
    plt.ylim(-5.2, 2.2)
    #plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(OUT/"log_distance.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    plt.rcParams.update({'font.family':'DejaVu Sans', 'mathtext.fontset':'dejavusans',
        'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10,
        'xtick.labelsize':9, 'ytick.labelsize':9, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.7, 'pdf.fonttype':42,
        'savefig.facecolor':'white'})
    COLORS = ['#0072B2','#D55E00','#009E73']
    MARKERS = ['o','s','^']
    LABELS = [r'$\rho_n$',r'$\gamma_1$',r'$\gamma_2$']

    def trade_scatter(ax):
        for value,color in [(0,COLORS[0]),(1,COLORS[1])]:
            use=trade_frame.PTA.to_numpy()==value
            ax.scatter(V[ii,jj,0][use],rho[use],s=11 if value else 7,alpha=.70 if value else .38,color=color,edgecolors='none',
                rasterized=True,label=f"{'PTA' if value else 'No PTA'} ({use.sum():,} dyads)")
        ax.axhline(constant,color='#555555',linestyle='--',linewidth=1.15,
            label=rf'Constant $p_1$: {constant:.2f}')
        ax.set(xlabel='Standardized log distance',ylabel= r'Estimated reciprocity $\hat\rho_{ij}$')
        ax.grid(color='#eeeeee',linewidth=.6); ax.set_axisbelow(True)
        ax.legend(loc='upper right',frameon=True,facecolor='white',edgecolor='none',fontsize=8)

    fig,ax=plt.subplots(figsize=(6.5,4.5),layout='constrained')
    trade_scatter(ax);ax.set_title('Fitted reciprocity across all 9,180 country pairs')
    save(fig,'Figure9_right_distance_PTA')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[.85,1.3]},layout='constrained')
    groups=[V[ii,jj,0][state==3],V[ii,jj,0][(state==1)|(state==2)],V[ii,jj,0][state==0]]
    box=axes[0].boxplot(groups,patch_artist=True,widths=.48,
        medianprops={'color':'#303846','linewidth':1.3},
        flierprops={'marker':'.','markersize':2,'markeredgecolor':'#777777','alpha':.4})
    for patch,col in zip(box['boxes'],['#a8cddd','#c8dfe8','#e2ebef']):patch.set_facecolor(col)
    axes[0].set_xticks([1,2,3],[f'Mutual\n({len(groups[0]):,})',f'One-way\n({len(groups[1]):,})',f'No link\n({len(groups[2]):,})'])
    axes[0].set_ylabel('Standardized log distance')
    axes[0].set_title('(a) Observed dyad types')
    axes[0].grid(axis='y',color='#eeeeee',linewidth=.6);axes[0].set_axisbelow(True)
    trade_scatter(axes[1]); axes[1].set_title('(b) Fitted reciprocity by distance')
    save(fig,'Figure9_distance_and_reciprocity')

    print(f"Generated lawyer network and trade figures in {OUT}."+(" KS summary included." if args.qq_input_dir is not None else ""))


if __name__=="__main__":main()
