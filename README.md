# LCPexperiments

You can either install LCPexperiments as a package or source the files in under the R directory to run numerical examples in [1].

## Illustrating example in Intro
See example_intro.Rmd

## Experiments with simulated synthetic data
X = 1D_setA,  1D_setB, 1D_setC, 1D_setD represents one of the four different data generative settings.

Run example1: Rscript synthetic_example1.R X
Run example2: Rscript synthetic_example1.R X

The intermediate training model is saved under synthetic_results/example1/* and synthetic_results/example1/* , and the final results are saved under  synthetic_results/* .

## Experiments with real data from UCI databases
X = CASP, Concrete, facebook, facebook2 represents one of the four different data sets.

Run X: Rscript simulation_X.R

The intermediate training model is saved under UCI_results/X/*  , and the final results are saved under  UCI_results/* .


