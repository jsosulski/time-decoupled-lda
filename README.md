# Time-decoupling in covariance matrices for LDA for ERP classification

**Disclaimer:** consider using [ToeplitzLDA](https://github.com/jsosulski/toeplitzlda), which reduces time and memory complexity, and in addition, is more stable and tends to perform even better.

### If you are looking for exact reproducibility of our paper, please refer to the [Neuroinformatics submission](https://github.com/jsosulski/time-decoupled-lda/tree/neuroinformatics_submission) tag

This repository contains the time-decoupling method as described in [1]. For details, please refer to the paper available [here](
https://rdcu.be/ccd0g). The implementation can be used as a drop-in replacement for an sklearn classifier. Additionally, the benchmark results for the publicly available datasets can be reproduced in this repository.

The results in [1] were produced using python 3.6.9 on an Ubuntu 18.04 installation.

---
[1] Sosulski, J., Kemmer, JP. & Tangermann, M. Improving Covariance Matrices Derived from Tiny Training Datasets for the Classification of Event-Related Potentials with Linear Discriminant Analysis. Neuroinform (2020). https://doi.org/10.1007/s12021-020-09501-8

## Installation (Ubuntu)

### 0. Clone this repository to your local machine.

### 1. Setting up the virtual environment.

a) If you have `make` installed, use:

```
make venv
```

b) Alternatively you should be able to set up the environment using pip. Open a terminal and enter the directory where you cloned this repository.

```
python3 -m venv tdlda_venv
. tdlda_venv/bin/activate
pip install --upgrade pip==19.3.1
pip install -r requirements.txt
pip install -e .
```

## (1b. Experimental Installation (Windows - Anaconda) )

Note: we recommend using Ubuntu 18.04 (if reproducibility is desired) or 20.04 or any other Linux flavor.

Start the anaconda prompt and go into the directory where you cloned this repository.

```
conda env create -f environment_mne.yml
conda env update -n tdlda -f environment.yml
```

Warning: we cannot guarantee that this works on your machine / reproduces our results.
If there is demand (open an issue) we could provide a singularity container that reproduces our results on all machines. As this uses virtualization when not used with a Linux kernel, performance will be noticeably worse than when used on a native Linux machine.

### 2. Setting up the configs

Copy the `scripts/local_config.yaml.example` to `scripts/local_config.yaml` and enter the `results_root` path where you want to store the benchmark results on your local machine. You do not need to change `benchmark_meta_name`. The default configuration is to evaluate only the optimal hyperparameters (see [1]). If you want to instead evaluate all hyperparameters of the parameter grid we used, you need to change `best_only` to `full`.

The `analysis_config.yaml` file defines the EEG preprocessing, as well as the used time intervals for the different paradigms and dimension settings. In the default state you have the exact same preprocessing as used in [1]. Note that changing anything could result in different results from the paper.

## Usage

As this was developed on Ubuntu 18.04 with python 3.6.9, I will hereinafter give instructions how to create the results in this environment.

### Reproduce the paper benchmark results

#### Obtain results

Open the terminal and navigate to the folder with all the scripts:

```
cd scripts/batch_submission
```

Run the benchmark using:

```
./batch_main_moabb_pipeline.sh submission_default.txt
```

**Warning: running the full benchmark when using all hyperparameters will take a while. For reference, on a i7-4710MQ running the benchmark took 24 hours, and on a i7-8700K it took 8 hours. The first time each dataset is analyzed, the relevant data is automatically downloaded. In total 22.1gb of data will be downloaded when running the benchmark. Evaluating only the best hyperparameters took around 1 hour on the i7-4710MQ.**

*Note:* if you have an unstable internet connection it could make sense to first download all datasets before starting the benchmark to not have the benchmark abort due to connection issues. To do this uncomment the line "#pipelines = dict(test=pipelines['jm\_few\_lda\_p\_cov'])" in `scripts/main_moabb_pipeline.py` and start the benchmark as described above. Once it is done you can comment out this line again and start the actual benchmark.

Results will be stored depending on the settings in your `local_config.yaml`. After starting the benchmark, some information and the output folder will be displayed, for example:

```
Results will be stored in:
    /home/jan/bci_data/results/covariance_benchmark/2020-05-08/6b41520c-fd0f-4b42-bed7-bd3a72e7b0d2
```

#### Analyzing the benchmark results

After the benchmark is done, you need to run the the scripts generating the figures. These scripts need to know where the results of the benchmark run are stored. If you simply want to analyse the results of the last benchmark run, this is stored in `last_benchmark_results_short_path.txt` and will be automatically passed to the analysis scripts if no arguments are given. Therefore:

Explicit way (given the example output from above):

```
./batch_main_moabb_analysis.sh 2020-05-08/6b41520c-fd0f-4b42-bed7-bd3a72e7b0d2
```

Implicit way (analysing results of last run benchmark):

```
./batch_main_moabb_analysis.sh
```

On our machine this analysis took ~5min. It will create a lot of figures (around 200) for various different subjects, datasets, and methods that are compared against one another.

#### Matching your results with the paper

In the results folder (in our example: `/home/jan/bci_data/results/covariance_benchmark/2020-05-08/6b41520c-fd0f-4b42-bed7-bd3a72e7b0d2`) there is a `plots` subfolder. All analyses are stored in here.

- `highscores_all_datasets.csv` stores the grand average AUC for all methods with all hyperparameter combinations. Note that this is different from the results obtained in [1] as we used additional datasets that are not publicly available as the obtained subject's consent did not permit publication of raw EEG data.
- Assuming we are currently in the `plots` subfolder:
    + Figure 5 in [1] corresponds to the file `results_across_all_datasets.pdf`. Again, note that in the paper more datasets are represented.
    + Figure 6 corresponds to the files:
        * (a) `across_datasets/REF_ceat_rg_xdawncomps_5_xdawnclasses_Target_VS_jm_numerous_lda_imp_p_cov.pdf`
        * (b) `across_datasets/REF_jm_numerous_lda_p_cov_VS_jm_numerous_lda_imp_p_cov.pdf`
    + Figure 7 corresponds to the files:
        * (a) `Spot_single_trial__jm_numerous.pdf`
        * (c) `Brain_invaders_singlesession__jm_numerous.pdf`

### Using the time-decoupled LDA classifier in your own python program

As the classifier is implemented using an sklearn like interface, you can simple consider it a drop-in replacement for classifiers that use the sklearn classifier interface.

Common errors:

+ The feature vector is stacked in the wrong way, i.e. x(channel1,time1), x(channel1, time2) instead of x(channel1, time1), x(channel2, time1), ...
+ Forgot to specify number of channels / number of time intervals. This information is needed to determine which blocks in the matrix are to be replaced.
+ When using this classifier, baseline correction of the epochs before classification may impede performance considerably (see [1]). If possible, skip baseline correction and use a high-pass band that makes it mostly superfluous, e.g. 0.5 Hz.

## Troubleshooting

Sometimes you need to install additional system packages on your Ubuntu 18.04 installation.

Required:

- `apt install python3-venv`
- `apt install python3-tk`

If you want to use the makefile:
- `apt install make`
