import argparse
import sys
import torch
import rdkit
from rdkit import Chem
import pandas as pd, numpy as np
import re
from tqdm.auto import tqdm
from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed
import selfies as sf

## importing fucntions for converting fragSmiles representations
from chemicalgof import Sequence2Smiles

## Extra model loaded to employ settings
from word_rnn import WordRNN, WordRNNTrainer, word_rnn_parser

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()
MODELS.add_model("word_rnn", WordRNN, WordRNNTrainer, word_rnn_parser)

def cleanTokens(arr):
    toDel=["<bos>","<eos>","<pad>"]
    arr=np.array(arr)
    ids=np.where(  np.logical_or.reduce([arr==i for i in toDel]) )
    return np.delete(arr, ids, axis=None)

def GenSmiles2Smiles(x):
    x = Chem.MolFromSmiles(*x)
    if not x:
        return None
    elif Chem.DetectChemistryProblems(x):
        return None

    return Chem.MolToSmiles(x)

def GenSelfies2Smiles(x):
    try:
        sm = sf.decoder(*x)
        return Chem.CanonSmiles(sm)
    except:
        return None

def GenFragSmiles2Smiles(x:pd.Series) -> str:
    lst=x.dropna()
    lst=cleanTokens(lst)
    return Sequence2Smiles(lst)


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models sampler script', description='available models')
    for model in MODELS.get_model_names():
        modelPars = subparsers.add_parser(model)
        add_sample_args(modelPars)
        if model in ("char_rnn","word_rnn"):
            groups = {grp.title:grp for grp in modelPars._action_groups}
            common = groups['Common']
            common.add_argument("--temp",
                                    type=float, default=1.0,
                                    help="Sampling temperature for softmax")
            common.add_argument('--onlyNovels',
                                   default=False, action='store_true',
                                   help='If generation leads to only novel molecules')
            common.add_argument("--notation",
                                type=str, required=True,
                                help="Molecular representation employed to feed model")
    return parser

def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)

    model_state = torch.load(config.model_load)


    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
        
    if 'aug' in model_config.train_load.lower():
        print('loading no augmented list of SMILES training set')
        smilesRef = pd.read_csv(re.sub('aug[0-9]+','',model_config.train_load, flags=re.IGNORECASE), 
                                usecols = ["smiles"], compression="xz", ).squeeze()
    else:
        smilesRef = pd.read_csv(model_config.train_load, usecols = ["smiles"], compression="xz", ).squeeze()
    
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
    # nameOut = config.gen_save[:-4]+"_T"+str(temp)+".csv"
    samples.to_csv(config.gen_save, index=False)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
