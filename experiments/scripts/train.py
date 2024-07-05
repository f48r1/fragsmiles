import argparse
import os
import sys
import torch
import rdkit
from rdkit import Chem
import pandas as pd, numpy as np
import pickle
from ast import literal_eval

from moses.script_utils import add_train_args, read_smiles_csv, set_seed
from moses.models_storage import ModelsStorage
from moses.dataset import get_dataset
from moses.utils import OneHotVocab, SpecialTokens, CharVocab

## Extra model loaded for employ settings
from word_rnn import WordRNN, WordRNNTrainer, word_rnn_parser

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()
MODELS.add_model("word_rnn", WordRNN, WordRNNTrainer, word_rnn_parser)

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        modelPars = MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
                    )
        add_train_args(
            modelPars
            )
        groups = {grp.title:grp for grp in modelPars._action_groups}
        common = groups['Common']
        common.add_argument("--notation", required=False,
                                type=str, default='fragsmiles',
                                help="Molecular representation employed to feed model")
    return parser

## converting string of tuple when importing csv avoiding literal eval on nan cells
def _converting(cell):
    if cell.startswith('(') :
        return literal_eval(cell)


def main(model, config):
    set_seed(config.seed)
    print(config.device)
    device = torch.device(config.device)

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    if config.train_load is None:
        train_data = get_dataset('train')
    else:
        # train_data = read_smiles_csv(config.train_load) # Edited here to adapting fSMiles
        # print('Loading training dataset')
        train_data = pd.read_csv(config.train_load, usecols = [config.notation], 
                                compression="xz" if 'tar.xz' in config.train_load else 'infer',
                                converters={config.notation: _converting} if config.notation != 'smiles' else None ).squeeze()
        
        if config.notation == 'fragsmiles':
            train_data.dropna(inplace=True)
        
    if config.val_load == "default":
        val_data = get_dataset('test')
    elif config.val_load is not None:
        # print('Loading validation dataset')
        val_data = pd.read_csv(config.val_load, usecols = [config.notation], 
                                compression="xz" if 'tar.xz' in config.val_load else 'infer',
                                converters={config.notation: _converting} if config.notation != 'smiles' else None ).squeeze()
        
        if config.notation == 'fragsmiles':
            val_data.dropna(inplace=True)

    else:
        val_data = None
  
    trainer = MODELS.get_model_trainer(model)(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), \
            'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data.tolist())

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = MODELS.get_model_class(model)(vocab, config).to(device)
    trainer.fit(model, train_data.tolist(), val_data.tolist())

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
