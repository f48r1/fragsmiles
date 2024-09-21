import argparse
import sys
import torch
import rdkit
from rdkit import Chem
import pandas as pd, numpy as np
import re
from tqdm.auto import tqdm
from moses.models_storage import ModelsStorage
from moses.script_utils import set_seed

import os

from src.utils import (add_common_params, 
                       add_sample_params, 
                       root_path_from_config, 
                       setup_name_from_config, 
                       suffix_files_from_config,
                       data_path_from_config,
                       gen_name_from_config)

import selfies as sf

## importing fucntions for converting fragSmiles representations
# from chemicalgof import Sequence2Smiles

## Extra model loaded to employ settings
from src.word_rnn import WordRNN, WordRNNTrainer, word_rnn_parser
from src.conversion import (GenSmiles2Smiles,
                            GenSelfies2Smiles,
                            GenFragSmiles2Smiles,
                            GentSmiles2Smiles)

from src.utils import load_data_from_path

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()
MODELS.add_model("word_rnn", WordRNN, WordRNNTrainer, word_rnn_parser)


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models sampler script', description='available models', dest='model')
    for model in MODELS.get_model_names():
        modelPars = MODELS.get_model_train_parser(model)( subparsers.add_parser(model) )

        common_arg = modelPars.add_argument_group('Common')
        add_common_params(common_arg)
        add_sample_params(common_arg)

    return parser

def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    ROOT = root_path_from_config(config)
    SETUP = setup_name_from_config(config)
    SUFFIX = suffix_files_from_config(config)

    setup_path = os.path.join(ROOT,SETUP)

    suffix_files_path= os.path.join(setup_path,SUFFIX)

    model_config = torch.load(f'{suffix_files_path}_config.pt')
    model_vocab = torch.load(f'{suffix_files_path}_vocab.pt')
    model_state = torch.load(f'{suffix_files_path}_model'+'_{0:03d}.pt'.format(config.epoch))

    model = MODELS.get_model_class(config.model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
        
    data_path= data_path_from_config(config)
    if 'Aug' in data_path:
        print('loading no augmented list of SMILES training set')
        data_path = re.sub('aug[0-9]+','',data_path, flags=re.IGNORECASE)

    smilesRef = load_data_from_path(data_path, notation='smiles', fold=config.fold, return_valid=False)

    samples = pd.DataFrame()
    smiles = pd.Series()
    n = config.n_samples
    with tqdm(total=config.n_samples, desc='Generating samples') as T:
        while n > 0:
            current_samples = model.sample(
                min(n, config.n_batch), config.max_len, temp = config.temp
            )
            current_samples = pd.DataFrame(current_samples)
            if config.notation == 'fragsmiles':
                current_smiles = current_samples.apply(GenFragSmiles2Smiles, axis=1)
            elif config.notation == 'smiles':
                current_smiles = current_samples.apply(GenSmiles2Smiles, axis=1)
            elif config.notation == 'selfies':
                current_smiles = current_samples.apply(GenSelfies2Smiles, axis=1)
            elif config.notation == 'tsmiles':
                current_smiles = current_samples.apply(GentSmiles2Smiles, axis=1)

            if config.onlyNovels:
                maskDupl = current_smiles.duplicated()
                maskNotNov = current_smiles.isin(smilesRef)
                maskNotVal = current_smiles.isna()
                current_samples = current_samples[~maskDupl &  ~maskNotNov & ~maskNotVal]
                current_smiles = current_smiles[~maskDupl &  ~maskNotNov & ~maskNotVal]
                smilesRef = pd.concat([smilesRef,current_smiles], ignore_index=True)
                
            samples = pd.concat([samples, current_samples], ignore_index=True)
            smiles = pd.concat([smiles, current_smiles], ignore_index=True)
            
            n -= len(current_samples)
            T.update(len(current_samples))

    samples.insert(0, "smiles", smiles)

    gen_path = os.path.join ( setup_path, gen_name_from_config(config) + '.csv')
    samples.to_csv(gen_path, index=False)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    main(config)
