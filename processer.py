import pandas as pd
import numpy as np
from os import cpu_count
import selfies as sf

### Caution HERE : name of package imported can change...
from chemicalgof import Smiles2GoF, GoF2Tokens, GoF2MoreTokens, CanonicalGoF2Tokens

from multiprocesspandas import applyparallel

from rdkit import Chem
from rdkit.Chem.MolStandardize.fragment import is_organic

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            if atom.GetSymbol()=="B":
                continue
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def MolWithoutIsotopes(mol):
    atoms = [atom for atom in mol.GetAtoms() if atom.GetIsotope()]
    for atom in atoms:
    # restore original isotope values
        atom.SetIsotope(0)
    return mol

def RemoveStereoFromSmiles(s, chars=["/","\\"]):
    for c in chars:
        s=s.replace(c,"")
    return s

def OrganicChooser(m):
    ## This function try to reproduce what these rdkit objects would do  !:
    # rdMolStandardize.MetalDisconnector()
    # rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    
    organicMols = [mol for mol in Chem.GetMolFrags(m, asMols=True) if is_organic(mol)]
    if len(organicMols)==0:
        return None
    elif len(organicMols)==1:
        return organicMols[0]
    else:
        return max(organicMols, key=lambda x: x.GetNumAtoms() )

def preprocessingWorker(sm):
    mol=Chem.MolFromSmiles(sm)
    if not mol:
        return None
    mol = OrganicChooser(mol)
    if not mol:
        return None
    mol = neutralize_atoms(mol)
    mol = MolWithoutIsotopes(mol)
    sm = Chem.MolToSmiles(mol)
    sm = RemoveStereoFromSmiles(sm)
    return Chem.CanonSmiles(sm)

def preprocessSmilesList(smilesList, nCpu=cpu_count()):
    if not type(smilesList) == pd.Series:
        smilesList = pd.Series(smilesList).squeeze()

    smilesList=smilesList.apply_parallel(preprocessingWorker, num_processes=nCpu)
    smilesList.dropna(inplace=True)
    smilesList.drop_duplicates(inplace=True)
    return smilesList

def tokenizeWorker(sm, aug, fncDecompose):

    diG=fncDecompose(sm)
    if diG is None:
        out = np.nan
    elif aug>1:
        T=GoF2MoreTokens(diG, aug)
        out = [ tuple(t.getSequence()) for t in T]
        if len(out)<aug:
            out.extend([np.nan for _ in range(aug-len(out))])
    else:
        # T=GoF2Tokens(diG)
        T = CanonicalGoF2Tokens(diG)
        out= tuple(T.getSequence())

    objArr = np.empty(1, dtype=object)
    objArr[0] = out
    
    return objArr

def fragmentSmilesList(smilesList, augmentation=1,fncDecompose=Smiles2GoF, nCpu=cpu_count()):
    if not type(smilesList) == pd.Series:
        smilesList = pd.Series(smilesList).squeeze()

    smilesList = smilesList.to_frame(name="smiles")
    smilesList["tokens"] = smilesList["smiles"].apply_parallel(tokenizeWorker, aug=augmentation,fncDecompose=fncDecompose, num_processes=nCpu)
    if augmentation>1:
        smilesList = smilesList.explode("tokens")

    ## nan should be dropped after processing ... then we can track which smiles cant be fragmented or augmented ##
    # smilesList.dropna(inplace=True)
    smilesList["nToks"] = smilesList["tokens"].apply_parallel(lambda s: np.nan if pd.isna(s) else len(s), num_processes=nCpu)
    return smilesList

def getSelfiesToks(sm):
    selfies = sf.encoder(sm)
    toks = tuple(sf.split_selfies(selfies))

    objArr = np.empty(1, dtype=object)
    objArr[0] = toks
    
    return objArr

def augSmiles(smi, aug=5):
    augs = [smi]
    mol = Chem.MolFromSmiles(smi)
    attempts = 0
    while len(augs)<aug and attempts<20:
        new = Chem.MolToSmiles(mol, doRandom=True)
        if new not in augs:
            augs.append(new)

        attempts+=1
        
    objArr = np.empty(1, dtype=object)
    objArr[0] = augs
    
    return objArr