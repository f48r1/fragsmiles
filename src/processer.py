import pandas as pd
import numpy as np
from os import cpu_count
import selfies as sf

### Caution HERE : name of package imported can change...
from chemicalgof import Smiles2GoF, GoF2Tokens, GoF2MoreTokens, CanonicalGoF2Tokens

import multiprocessing

import networkx as nx
from rdkit import Chem

try:
    from rdkit.Chem.MolStandardize.fragment import is_organic
except:
    import warnings
    warnings.warn(r'function `is_organic` is not more available on latest version of rdkit. Custom function is then provided.')
    simpleOrganicAtomQuery = Chem.MolFromSmarts('[!$([#1,#5,#6,#7,#8,#9,#15,#16,#17,#35,#53])]')
    simpleOrganicBondQuery = Chem.MolFromSmarts('[#6]-,=,#,:[#6]')
    hasCHQuery = Chem.MolFromSmarts('[C!H0]')
    
    def is_organic(mol):
        return (not mol.HasSubstructMatch(simpleOrganicAtomQuery)) and mol.HasSubstructMatch(hasCHQuery) and mol.HasSubstructMatch(simpleOrganicBondQuery)

DEFAULT = None

def apply_on_list(lst, fnc):
    ret = []
    for x in lst:
        try:
            ret.append(fnc(x))
        except:
            print(x,'in',lst)
            ret.append(x)
    return ret

def applyFncPool(column, fnc, CPUs=cpu_count()):
    with multiprocessing.Pool(processes=CPUs) as pool:
        column = pool.map(fnc, column)

    return column

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

def preprocessSmiles(sm, default=DEFAULT):
    mol=Chem.MolFromSmiles(sm)
    if not mol:
        return default
    mol = OrganicChooser(mol)
    if not mol:
        return default
    mol = neutralize_atoms(mol)
    mol = MolWithoutIsotopes(mol)
    sm = Chem.MolToSmiles(mol)
    sm = RemoveStereoFromSmiles(sm)
    return Chem.CanonSmiles(sm)

def fragmentSmiles(sm, aug=1, fncDecompose=Smiles2GoF, sep=" ", default=DEFAULT):

    diG=fncDecompose(sm)
    # [x] feature added : recognize if there are cyclic pieces of graph
    if diG is None:
        out=default
    elif list(nx.simple_cycles(diG.to_undirected())):
        out=default
    elif aug>1:
        Ts=GoF2MoreTokens(diG, aug)
        out = [ sep.join( T.getSequence() ) for T in Ts ]
        if len(out)<aug:
            out.extend([default for _ in range(aug-len(out))])
    else:
        T = CanonicalGoF2Tokens(diG)
        out= sep.join( T.getSequence() )
    
    return out

def augSmiles(smi, aug=5, default=DEFAULT):
    out = [smi]
    mol = Chem.MolFromSmiles(smi)
    attempts = 0
    while len(out)<aug and attempts<20:
        new = Chem.MolToSmiles(mol, doRandom=True)
        if new not in out:
            out.append(new)

        attempts+=1
        
    if len(out)<aug:
        out.extend([default for _ in range(aug-len(out))])

    return out

def getSelfiesToks(sm, default=DEFAULT):
    if not sm:
        return default
    selfies = sf.encoder(sm)
    toks = sf.split_selfies(selfies)
    return ' '.join(toks)


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