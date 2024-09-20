import os
from moses.script_utils import add_common_arg

def add_common_params(parser):
    add_common_arg(parser)

    parser.add_argument('--save_frequency',
                        type=int, default=20,
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
                        type=str, default='grisoni',
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
    parser.add_argument('--n_jobs',
                        type=int, default=1,
                        help='Number of processes to run metrics')
    
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