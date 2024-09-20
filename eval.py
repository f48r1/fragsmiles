import argparse
import numpy as np, pandas as pd
import rdkit
import os

from moses.metrics.metrics import get_all_metrics
from moses.models_storage import ModelsStorage

from src.word_rnn import WordRNN, WordRNNTrainer, word_rnn_parser

import os
from src.utils import (add_common_params, 
                       add_sample_params, 
                       root_path_from_config, 
                       setup_name_from_config, 
                       data_path_from_config,
                       gen_path_from_config)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()
MODELS.add_model("word_rnn", WordRNN, WordRNNTrainer, word_rnn_parser)

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models eval script', description='available models', dest='model')
    for model in MODELS.get_model_names():
        modelPars = subparsers.add_parser(model)
        add_common_params(modelPars)
        add_sample_params(modelPars)

    return parser


def main(config, print_metrics=True):
    test = None
    test_scaffolds = None
    ptest = None
    ptest_scaffolds = None
    train = None


    ROOT = root_path_from_config(config)
    SETUP = setup_name_from_config(config)


    data_strformat = data_path_from_config(config)

    test = pd.read_csv(data_strformat.format('test'), usecols = ["smiles"], compression="xz", 
                        ).squeeze().values

    train = pd.read_csv(data_strformat.format('train'), usecols = ["smiles"], compression="xz", 
                        ).squeeze().values


    gen_path = os.path.join(ROOT,SETUP, gen_path_from_config(config))

    gen = pd.read_csv(gen_path+'.csv', usecols=["smiles"]).fillna("-").squeeze().values

    
    metrics = get_all_metrics(gen=gen, k=config.ks, n_jobs=config.n_jobs,
                              device=config.device,
                              test_scaffolds=test_scaffolds,
                              ptest=ptest, ptest_scaffolds=ptest_scaffolds,
                              test=test, train=train)

    table = pd.DataFrame([metrics]).T
    table.to_csv(gen_path+'_metrics.csv', header=False)

    if print_metrics:
        for name, value in metrics.items():
            print('{},{}'.format(name, value))
    else:
        return metrics

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    model_metrics = main(config, False)