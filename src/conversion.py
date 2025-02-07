from rdkit import Chem
from chemicalgof import Sequence2Smiles, String2Tokens
import pandas as pd
import numpy as np
import selfies as sf

DEFAULT = None

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
        return GenSmiles2Smiles([sm])
    except:
        return None

def GenFragSmiles2Smiles(x:pd.Series) -> str:
    lst=x.dropna()
    lst=cleanTokens(lst)
    return Sequence2Smiles(lst)

def GenAtomFragSmiles2Smiles(x):
    try:
        T = String2Tokens(*x)
        diG = T.getGraph()
        mol = diG.getMol()
        x = Chem.MolToSmiles(mol)
        return x
    except:
        return None

from tsmiles.DataSet.STDTokens import CTokens, STDTokens_Frag_File
from tsmiles.MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from tsmiles.DataSet.Graph.CNJTMol import CNJMolUtils
_ctoken = CTokens(STDTokens_Frag_File(None), max_length = 256, invalid = True, onehot = False)
def getTsmiles(sm, dec_alg = Fragment_Alg.MMPA_DY, default=DEFAULT):
    if not sm:
        return default

    combine_sml, combine_smt = CNJMolUtils.encode_single(sm, _ctoken, dec_alg)

    return combine_sml ## for TSSA/TSDY
    return combine_smt ## for TSID

asm_alg = 'CALG_TSDY'
from tsmiles.DataSet.Graph.CNJMolAssembler import CNJMolAssembler
def GentSmiles2Smiles(x):
    re_smils, bfs_ex_smiles_sub, new_vocs_sub = CNJMolAssembler.decode_single(*x, _ctoken , asm_alg, n_samples = 1, p_mean = None)
    return GenSmiles2Smiles[re_smils]
