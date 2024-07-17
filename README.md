![python version](https://img.shields.io/badge/python-3.10_|_3.11-white)
![license](https://img.shields.io/badge/license-MIT-orange)
[![Static Badge](https://img.shields.io/badge/ChemRxiv-10.26434/chemrxiv--2024-tm7n6)](https://doi.org/10.26434/chemrxiv-2024-tm7n6)
[![Static Badge](https://img.shields.io/badge/Zenodo-_10.5281/12713950-blue)](https://doi.org/10.5281/zenodo.12713950)

# fragSMILES analysis and evaluation

Workflow employed for fragSMILES analysis and evaluation for de novo drug design provided by [relative work](https://doi.org/10.26434/chemrxiv-2024-tm7n6)\
Here, notebook files are explained about their aim sorted by steps of workflow (enumeration order).

## Requirements

The entire workflow is based on Python language. Neural network training was made possible by Pytorch package. However, requirements are provided in [requirements.txt](requirements.txt). Essential Python repositories include [chemicalgof](https://github.com/f48r1/chemicalgof) for molecular decomposition of each dataset and [MOSES](https://github.com/molecularsets/moses) for RNN model architecture and metric evaluations. **Ensure these repositories are installed before cloning this repository**:

```shell
git clone https://github.com/f48r1/fragsmiles.git
```

After cloning, navigate to the folder and install the required packages for the pipeline (pip is suggested):

```shell
cd fragsmiles/
pip install -r requirements.txt
```

The [experiments/scripts](experiments/scripts) folder has been adapted from the MOSES benchmarking framework to accommodate various chemical notations used in our study for training, sampling and evaluation phases. Given that fragSMILES is a *chemical-word-level notation*, modifications were made to the char-RNN model architecture originally provided by [MOSES](https://github.com/molecularsets/moses), resulting in a tailored word-RNN model architecture, found in ([experiments/scripts/word_rnn](experiments/scripts/word_rnn)).

## Training phase

RNN models were trained for each notation (SMILES, SELFIES, fragSMILES) and each hyperparameter setting. In the experiments/ folder, you will find the `train_single_model.sh`, `sample_single_model.sh`, and `eval_single_model.sh` bash script files. These can be used to initiate the training, sampling, or evaluation phases for a single model. The files provide key arguments to specify the chemical notation, hyperparameters, and other parameters. The script files must be run from the same directory ([experiments/](experiments/)). Additionally, if you are using a **conda or pure Python environment, it needs to be activated within the script**.

Ex: to train an RNN model on the [Grisoni](https://github.com/ETHmodlab/BIMODAL) dataset, using the first fold, represented in fragSMILES notation with 2 hidden layers, 512 hidden units, a batch size of 512, learning rate of 0.001, and an embedding size of 300. The device can be set as either CPU or GPU:

```bash
cd experiments/
bash train_single_model.sh --dataset grisoni --fold 0 --notation fragsmiles --hl 2 --hu 512 --bs 512 --lr 0.001 --es 300 --device cuda:0 --epochs 16
```

Saved files concerning the weights of the models are not included in this repository. However, training log files, generated set of molecules and their evaluation metrics are stored in the respective settings within the [experiments/](experiments/) folder.

---

# Notebooks

## 0_smileIssues

This notebook highlights the differences between the notations SMILES, SELFIES, and fragSMILES for the molecules depicted in Figure 1 and Figure 2. Specifically, it examines three known drugs composed of indole fragments and molecules with chiral stereocenters. The images have been saved in the figures/ path.

## 1_zincCleavage

Decomposition of Zinc250K dataset were made starting from [zinc.csv](data/zinc.csv) file stored in [data](data/) folder. Original file was provided by [Cheng et al](https://github.com/aspuru-guzik-group/group-selfies) work (Group SELFIES).\
Dataset was cured employing functions imported from [processer.py](processer.py) file. Then, each molecule was decomposed by standard cleavage rule of [GoF](https://github.com/f48r1/chemicalgof) framework (for fragSMILES notation) and by rotatable bonds rule. Resulting decomposed molecules as textual representation are stored in [results/01_zincToks.csv](results/01_zincToks.csv) and [results/01_zincRotatableToks.csv](results/01_zincRotatableToks.csv).

## 2_fragCounts

Decomposed molecules of Zinc250K represented as fragSMILES were compared with Group SELFIES, SELFIES and SMILES by number of tokens (encoding lenght). Benchmarking file (stored in [data/compactnessBenchmarking.csv](data/compactnessBenchmarking.csv) and provided by [Cheng et al](https://github.com/aspuru-guzik-group/group-selfies) work) were employed for the purpose and plotted in figure saved in [figures/encodingCount.pdf](figures/encodingCount.pdf).

## 3_vocabCounts

Counts of unique tokens provided by files [results/01_zincToks.csv](results/01_zincToks.csv) and [results/01_zincRotatableToks.csv](results/01_zincRotatableToks.csv) are compared. The comparison is summarized in the table saved in [results/03_cleavageBenchmarking.csv](results/03_cleavageBenchmarking.csv).

## 4_fragMoses

Whole molecules dataset provided by [MOSES](https://github.com/molecularsets/moses) benchmarking framework was decomposed into fragSMILES notation. Functions employed for this purpose were imported from [processer.py](processer.py) (multiprocessing was adopted to accelerate decomposition). The molecule dataset represented as SMILES, SELFIES and fragSMILES was split into train and test sets using a 5-fold cross-validation scheme. These datasets were then stored in the [experiments/data/](experiments/data/) following a schematic naming convention such as `moses_train_*.tar.xz` and `moses_test_*.tar.xz` where `*` indicates the fold number.

## 5_fragGrisoni

Whole molecules dataset provided by [Grisoni et al](https://github.com/ETHmodlab/BIMODAL) work was first cured and then decomposed into fragSMILES notation. Functions employed for this purpose were imported from [processer.py](processer.py) (multiprocessing was adopted to accelerate decomposition). Molecules represented by a number of tokens in range 10-32 were retained. Then, resulting molecules represented as SMILES, SELFIES and fragSMILES were augmented till 5 representations. Finally, 2 resulting dataset were split into train and test sets using a 5-fold cross-validation scheme. These datasets were then stored in [experiments/data/](experiments/data/) following a schematic naming convention such as `grisoni_train_*.tar.xz`, `grisoni_test_*.tar.xz` and `grisoni_trainAug5_*.tar.xz` where `*` indicates the fold number.

## 6_metricMoses

Sampled molecules generated by RNN models trained on the MOSES dataset, for each hyperparameter setting, were evaluated using functions imported from the [evaluation.py](evaluation.py) file. Notably, [experiments/](experiments/) folder does not contain saved weight models, but it does include log files, generated sets and novel generated set. Results are stored in [results/](results/) folder.

## 7_metricGrisoni

See above, same of metricMoses notebook but on [Grisoni et al](https://github.com/ETHmodlab/BIMODAL) dataset. Evaluation on chiral molecules is also provided. Results are stored in [results/](results/) folder.

## 8_scaffoldGrisoni

A scaffold analysis was made on generated set of novel molecules by RNN models trained on Grisoni Dataset. Functions for this purpose were imported from [evaluation.py](evaluation.py) file.
