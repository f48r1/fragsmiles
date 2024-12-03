import os
from moses.script_utils import add_common_arg
import pandas as pd

def add_common_params(parser):
    add_common_arg(parser)

    parser.add_argument('--save_frequency',
                        type=int, default=1,
                        help='How often to save the model')
    parser.add_argument('--dir_experiments',
                        type=str, required=False, default='experiments/',
                        help='Folder root of experiments')
    parser.add_argument('--dir_data', default='data/',
                        type=str, required=False,
                        help='Folder in which data are stored')
    parser.add_argument("--notation", required=False,
                        type=str, default='fragsmiles',
                        help="Molecular representation employed to feed model")
    parser.add_argument("--fold", required=False,
                        type=int, default=0,
                        help="Molecular representation employed to feed model")
    parser.add_argument("--dataset", required=False,
                        type=str, default='chembl',
                        help="Molecular representation employed to feed model")
    parser.add_argument("--aug", required=False,
                        type=int, default=1,
                        help="Molecular representation employed to feed model")
    
def add_sample_params(parser):
    parser.add_argument('--epoch',
                        type=int, required=True,
                        help='Epoch of stored model')
    parser.add_argument('--n_samples',
                        type=int, required=False, default=6000,
                        help='Number of samples to sample')
    parser.add_argument("--max_len",
                        type=int, default=100,
                        help="Max of length of SMILES")
    parser.add_argument("--temp",
                        type=float, default=1.0,
                        help="Sampling temperature for softmax")
    parser.add_argument('--onlyNovels',
                        default=False, action='store_true',
                        help='If generation leads to only novel molecules')
    
def add_eval_params(parser):
    parser.add_argument('--ks', '--unique_k',
                        nargs='+', default=1000,
                        type=int,
                        help='Number of molecules to calculate uniqueness at.'
                             'Multiple values are possible. Defaults to '
                             '--unique_k 1000')
    
def data_name_from_config(config):
    return config.dataset + ('Aug5' if config.aug == 5 else '')

def data_path_from_config(config):
    return os.path.join(
            config.dir_data,
            data_name_from_config(config) + '.tar.xz'
                 )

def root_path_from_config(config):
    notation_dict={
        'fragsmiles':'fragSMILES',
        'smiles':'SMILES',
        'selfies':'SELFIES',
        'tsmiles':'t-SMILES',
    }

    return os.path.join(
            config.dir_experiments,
            data_name_from_config(config) + f'_{notation_dict[config.notation]}-RNN'
    )

def setup_name_from_config(config):
    return f'{config.num_layers}hl_{config.hidden}hu_{config.n_batch}bs' + \
    (f'_{config.embedding_size}es' if config.model == 'word_rnn' else '') +\
    f'_{config.lr}lr'

def suffix_files_from_config(config):
    return f'{config.model}_{config.fold}'

def gen_name_from_config(config):
    suffix = suffix_files_from_config(config)
    return suffix + f'_generated{config.n_samples}' + ('novels' if config.onlyNovels else '') +\
          f'_{config.epoch}_T{config.temp}'

def load_data_from_path(path, notation, fold, return_train=True, return_valid=True):
    
    if (not return_train and not return_valid) or fold in [None, False]:
        return pd.read_csv(path, usecols = [notation], compression="xz" if 'tar.xz' in path else 'infer').squeeze()

    col_fold = f'fold{fold}'

    data = pd.read_csv(path, usecols = [notation,col_fold], 
                        compression="xz" if 'tar.xz' in path else 'infer')

    if notation == 'fragsmiles' and 'Aug' in path:
        data.dropna(axis=0, inplace=True)
    
    if notation in ('fragsmiles','selfies'):
        data = data.str.split(' ')

    groups = data.groupby(col_fold)

    if return_train:
        train = groups.get_group('train').drop(columns=col_fold).squeeze().reset_index(drop=True)
    if return_valid:
        valid = groups.get_group('valid').drop(columns=col_fold).squeeze().reset_index(drop=True)

    if return_train and return_valid:
        return train,valid
    elif return_train:
        return train
    elif return_valid:
        return valid