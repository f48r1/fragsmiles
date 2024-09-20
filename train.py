import argparse
import os
import sys
import torch
import rdkit
from rdkit import Chem
import pandas as pd, numpy as np

from moses.script_utils import set_seed
from moses.models_storage import ModelsStorage

## Extra model loaded for employ settings
from src.word_rnn import WordRNN, WordRNNTrainer, word_rnn_parser
from src.utils import (root_path_from_config, 
                       data_path_from_config, 
                       setup_name_from_config, 
                       suffix_files_from_config, 
                       add_common_params)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()
MODELS.add_model("word_rnn", WordRNN, WordRNNTrainer, word_rnn_parser)

def load_data_from_path(path, config, return_train=True, return_valid=True):

    col_fold = f'fold{config.fold}'

    data = pd.read_csv(path, usecols = [config.notation,col_fold], 
                        compression="xz" if 'tar.xz' in path else 'infer')

    if config.notation == 'fragsmiles' and config.aug > 1:
        data.dropna(axis=0, inplace=True)
    
    if config.notation in ('fragsmiles','selfies'):
        data = data.str.split(' ')

    groups = data.groupby(col_fold)

    if return_train:
        train = groups.get_group('train').drop(columns=col_fold).squeeze()
    if return_valid:
        valid = groups.get_group('valid').drop(columns=col_fold).squeeze()

    if return_train and return_valid:
        return train,valid
    elif return_train:
        return train
    elif return_valid:
        return valid


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models', dest='model'
    )
    
    for model in MODELS.get_model_names():
        modelPars = MODELS.get_model_train_parser(model)( subparsers.add_parser(model) )
        
        common_arg = modelPars.add_argument_group('Common')
        add_common_params(common_arg)
        
    return parser

def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    ROOT_DIR = root_path_from_config(config) ## = experiments/{dataset}_{notation}-RNN/

    setup = setup_name_from_config(config)
    SETUP_DIR = os.path.join(ROOT_DIR,setup)

    suffix_files = suffix_files_from_config(config)
    suffix_files_path= os.path.join(SETUP_DIR,suffix_files)

    d_config = vars(config)
    d_config['config_save']=f'{suffix_files_path}_config.pt'
    d_config['model_save']=f'{suffix_files_path}_model.pt'
    d_config['log_file']=f'{suffix_files_path}_log.txt'
    d_config['vocab_save']=f'{suffix_files_path}_vocab.pt'

    torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    DATA_PATH = data_path_from_config(config)

    train_data, valid_data = load_data_from_path(DATA_PATH, config)
  
    trainer = MODELS.get_model_trainer(config.model)(config)

    vocab = trainer.get_vocabulary(train_data.tolist())

    os.makedirs(SETUP_DIR, exist_ok=True)
    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = MODELS.get_model_class(config.model)(vocab, config).to(device)
    trainer.fit(model, train_data.tolist(), valid_data.tolist())

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config, unk = parser.parse_known_args()
    if unk:
        print('unknown arguments:',*unk)
    main(config)
