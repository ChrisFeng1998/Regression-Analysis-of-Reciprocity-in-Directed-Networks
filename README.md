# Regression Analysis of Reciprocity in Directed Networks 

`estimators/CLL.py` implements the main function of the project. It enumerates all the tetrads in the directed network, 
constructs the conditional likelihood function, and provides point estimates together with their asymptotic variances.

For data analysis, please download the data from the [Lazega's lawyer data](https://www.stats.ox.ac.uk/~snijders/siena/Lazega_lawyers_data.htm) 
and [World trade network](http://personal.lse.ac.uk/tenreyro/LGW.html).


## Scope

This repository contains source code and dependency specifications. 

## Dependencies

Install the Python packages with:

```bash
python -m pip install -r requirements.txt
```


## Code structure

```text
estimators/        # Estimators and reusable numerical functions
    CLL.py
    assessment_models.py
    mcmc.py
    transitivity_core.py
simulations/       # Simulation experiments and MCMC input generator
    simulation.py
    p1_MLE.py
    p1_5_misspecification.py
    subsampling.py
    mcmc_simulation.py
    mcmc_inputs.R
    transitivity.py
data_analysis/     # Real-data analysis and held-out assessment
    raw_data.py
    empirical.py
    model_assessment.py
visualization/     # Simulation, QQ, KS and application plots
    plot_simulations.py
    generate_qq_plots.py
    plot_revision_figures.py
README.md
requirements.txt
```

## Usage

Each added entry point supports `--help`, e.g. `python -m simulations.subsampling --help`.

Examples of new experiments (these perform estimation):

```bash
python -m simulations.subsampling simulate --n 200 --regime 1 --replications 1 --out outputs/subsampling
python -m simulations.transitivity simulate --n 100 --delta 0.5 --replications 1 --out outputs/transitivity
python -m simulations.mcmc_simulation --regime 3 --replications 1 --output-dir outputs/mcmc
```

## Raw application data

Place the real datasets under a local directory as follows:

```text
/path/to/raw/
    LazegaLawyers/
        ELattr.dat
        ELadv.dat
    trade_network/
        Log of Gravity.xls
        countrycodes.xls
```


```bash
python -m data_analysis.empirical --raw-dir /path/to/raw
python -m data_analysis.model_assessment --raw-dir /path/to/raw --dataset lawyer --output-dir outputs/assessment
python -m simulations.transitivity application --raw-dir /path/to/raw --dataset lawyer --out outputs/closure
```

The empirical script retains the original CL estimates and variance calculation.
Model assessment constructs both complete ten-fold dyad partitions in memory,
including 00 dyads, and tunes parameters within training data. Closure checks
first estimate full-network reciprocity from the raw data, then simulate conditional
on the observed degree sequences. 

## Plotting your outputs

```bash
python -m visualization.plot_simulations --input-dir /path/to/simulation/results
python -m visualization.generate_qq_plots --input-dir /path/to/full_CL/results
python -m visualization.plot_revision_figures --raw-dir /path/to/raw
```

