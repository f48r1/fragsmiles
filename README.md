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

MOSES scripts have been extended from benchmarking framework to accommodate various chemical notations used in our study for training, sampling and evaluation phases. Given that fragSMILES is a *chemical-word-level notation*, modifications were made to the char-RNN model architecture originally provided by [MOSES](https://github.com/molecularsets/moses), resulting in a tailored word-RNN model architecture, found in ([src/word_rnn](src/word_rnn)).

## Training phase

RNN models were trained for each notation (SMILES, SELFIES, t-SMILES and fragSMILES) and each hyperparameter setting. Root folder provides `train.py`, `sample.py` and `eval.py` python scripts. These files can be used to initiate the training, sampling, or evaluation phases for a single model. The files provide key arguments to specify the chemical notation, hyperparameters, and other parameters.

Ex: to train an RNN model on the [ChEMBL subset](https://github.com/ETHmodlab/BIMODAL), using the first fold, represented in fragSMILES notation with 2 hidden layers, 512 hidden units, a batch size of 512, learning rate of 0.001, and an embedding size of 300. The device can be set as either CPU or GPU:

```shell
python train.py word_rnn --dataset chembl --fold 0 --notation fragsmiles --hl 2 --hu 512 --bs 512 --lr 0.001 --embedding_size 300 --device cuda:0 --train_epochs 16
```

Saved files concerning the weights of the models are not included in this repository. However, training log files, generated set of molecules and their evaluation metrics are stored in the respective settings within the default experiment storage folder [experiments/](experiments/).

## Dataset

[rawdata/](rawdata/) folder contains original data file collected from different works in literature. Specifically:

- `chembl_smiles.tar.xz` by [Grisoni et al.](https://github.com/ETHmodlab/BIMODAL) work providing a subset of bioactive molecules from ChEMBL represented as SMILES notation;
- `zinc.csv` by [Cheng et al](https://github.com/aspuru-guzik-group/group-selfies) work (Group SELFIES) providing molecules from Zinc250K represented as SMILES notation. Moreover 'compactnessBenchmarking.csv' provides the number of tokens to represent each molecule of the latter dataset as SMILES, SELFIES and Group SELFIES notations.
- `moses.txt` by [MOSES](https://github.com/molecularsets/moses) work contains over 1M molecules represented as SMILES notation, with no predefined train and validation splits.

[data/](data/) folder is for storage of elaborated molecular rawdata. In particular each tabular file is composed of columns dedicated to each chemical notation (SMILES, SELFIES, t-SMILES and fragSMILES). In addition, files adopted for model training, are composed of columns reporting the belonging split for molecules (named as 'fold{n}') whose value is a string, i.e., 'train' for molecules adopted for training and 'valid' for molecules adopted for validation.

---

# Notebooks

## 0_smileIssues

This notebook highlights the differences between the notations SMILES, SELFIES, and fragSMILES for the molecules depicted in Figure 1 and Figure 2. Specifically, it examines three known drugs composed of indole fragments and molecules with chiral stereocenters.

## 1_zincCleavage

Decomposition of Zinc250K dataset were made starting from [zinc.csv](rawdata/zinc.csv) file stored in [rawdata](rawdata/) folder. Original file was provided by [Cheng et al](https://github.com/aspuru-guzik-group/group-selfies) work (Group SELFIES).\
Dataset was cured employing functions imported from [processer.py](src/processer.py) file. Then, each molecule was decomposed by standard cleavage rule of [GoF](https://github.com/f48r1/chemicalgof) framework (for fragSMILES notation) and by rotatable bonds rule. Resulting decomposed molecules as textual representation are stored in [data/01_zincToks.csv](results/01_zincToks.csv) and [data/01_zincRotatableToks.csv](results/01_zincRotatableToks.csv).

## 2_fragCounts

Decomposed molecules of Zinc250K represented as fragSMILES were compared with Group SELFIES, SELFIES and SMILES by number of tokens (encoding lenght). Benchmarking file (stored in [rawdata/compactnessBenchmarking.csv](rawdata/compactnessBenchmarking.csv) and provided by [Cheng et al](https://github.com/aspuru-guzik-group/group-selfies) work) were employed for the purpose and plotted in figure saved in [figures/02_encodingCount.pdf](figures/02_encodingCount.pdf). Finally, the word level tokenization for the t-SMILES notation was compared with the fragSMILES tokenization obtained from different fragmentation rules.

## 3_vocabCounts

Counts of unique tokens provided by files [data/01_zincToks.csv](data/01_zincToks.csv) and [data/01_zincRotatableToks.csv](results/01_zincRotatableToks.csv) are compared. The comparison is summarized in the table saved in [results/03_cleavageBenchmarking.csv](results/03_cleavageBenchmarking.csv).

## 4_fragMoses

Whole molecules dataset provided by [MOSES](https://github.com/molecularsets/moses) benchmarking framework was decomposed into fragSMILES notation. Functions employed for this purpose were imported from [processer.py](src/processer.py) (multiprocessing was adopted to accelerate decomposition). The molecule dataset represented as SMILES, SELFIES, t-SMILES and fragSMILES was split into train and test sets using a 5-fold cross-validation scheme. These datasets were then stored in the [data/](data/).

## 5_fragGrisoni

Whole molecules dataset provided by [Grisoni et al](https://github.com/ETHmodlab/BIMODAL) work was first cured and then decomposed into fragSMILES notation. Functions employed for this purpose were imported from [processer.py](src.processer.py) (multiprocessing was adopted to accelerate decomposition). Molecules represented by a number of tokens in range 10-32 were retained. Then, resulting molecules represented as SMILES, SELFIES, t-SMILES and fragSMILES were augmented till 5 representations. Finally, 2 resulting dataset were split into train and test sets using a 5-fold cross-validation scheme. These datasets were then stored in [data/](data/).

## 6_metricMoses

Sampled molecules generated by RNN models trained on the MOSES dataset, for each hyperparameter setting, were evaluated using functions imported from the [evaluation.py](src.evaluation.py) file. Notably, [experiments/](experiments/) folder does not contain saved weight models, but it does include log files, generated sets and novel generated set. Results are stored in [results/](results/) folder.

## 7_metricChembl

See above, same of metricMoses notebook but on [Grisoni et al](https://github.com/ETHmodlab/BIMODAL) dataset. Evaluation on chiral molecules is also provided. Results are stored in [results/](results/) folder.

## 8_scaffoldChembl

A scaffold analysis was made on generated set of novel molecules by RNN models trained on Grisoni Dataset. Functions for this purpose were imported from [evaluation.py](src.evaluation.py) file.
