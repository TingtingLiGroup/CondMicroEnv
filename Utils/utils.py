import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
import json, os
import requests


def get_mol(smiles_or_mol):
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol, remove_salt=True):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    if remove_salt:
        remover = SaltRemover(defnData="[F,Cl,Br,I,Na+,K+,O,H+,Fe+3,Fe+2,Ca+2,Gd+3]")
        mol = remover.StripMol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def Calc_AP(mol):
    return len(mol.GetAromaticAtoms()) / mol.GetNumHeavyAtoms()


def Calc_ARR(mol):
    mol = Chem.RemoveHs(mol)
    num_bonds = mol.GetNumBonds()
    num_aromatic_bonds = 0
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            num_aromatic_bonds += 1
    ARR = num_aromatic_bonds/num_bonds
    return ARR


# distribution of C atom counts
def get_C_atom_counts(mol):
    counts = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            counts += 1
        
    return counts


def get_C_in_ring_proportion(mol):
    total_counts, in_ring_counts = 0, 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            total_counts += 1
            if atom.IsInRing():
                in_ring_counts += 1
    if total_counts > 0:
        return in_ring_counts / total_counts
    return -1  # for molecules without C atom


# distribution of heavy atom counts
def get_heavy_atom_counts(mol):
    return mol.GetNumHeavyAtoms()


# get protein info from UniProt database
def get_protein_info(uniprot_id):
    res = requests.get(f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.json')
    data = json.loads(res.text)

    # gene name
    for dct in data['genes']:
        if 'geneName' in dct:
            gene_name = dct['geneName']['value']
            break

    # pdb ids
    pdb_ids = []
    for dct in data['uniProtKBCrossReferences']:
        if dct['database'] == 'PDB':
            # keep X-ray & NMR
            for i in dct['properties']:
                if i['key'] == 'Method' and i['value'] in ['X-ray', 'NMR']:
                    pdb_ids.append(dct['id'])
                    break
    pdb_ids = '|'.join(pdb_ids)

    # protein family
    pro_family = []
    for dct in data['comments']:
        if dct['commentType'] == 'SIMILARITY':
            for t in dct['texts']:
                if 'value' in t:
                    pro_family.append(t['value'])
                    break
    pro_family = '|'.join(pro_family)

    # protein function
    functions = []
    for dct in data['keywords']:
        if dct['category'] == 'Molecular function':
            functions.append(dct['name'])
    functions = '|'.join(functions)

    # protein sublocation
    locations = []
    for dct in data['keywords']:
        if dct['category'] == 'Cellular component':
            locations.append(dct['name'])
    locations = '|'.join(locations)

    return gene_name, pdb_ids, pro_family, functions, locations


def family_parser(info):
    if info == '' or type(info) == float:
        return None, None, None

    elif info.startswith('Belongs to the '):
        info = info.split('Belongs to the ')[1]
        # family
        if '.' in info:
            infos = info.split('. ')
        else:
            infos = [info]

        superfamily = None
        family = None
        subfamily = None
        for i in infos:
            if ' superfamily' in i.strip():
                superfamily = i.split(' superfamily')[0].strip()
            if ' family' in i.strip():
                family = i.split(' family')[0].strip()
            if ' subfamily' in i.strip():
                subfamily = i.split(' subfamily')[0].strip()
        return superfamily, family, subfamily
    
    else:  # N terminal/C terminal: do it manually
        return info, info, info


def tanimoto_similarity_smiles(smi1, smi2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(get_mol(smi1), 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(get_mol(smi2), 2, 2048)
    sim = DataStructs.FingerprintSimilarity(fp1, fp2)
    return sim


def tanimoto_similarity_fp(fp1, fp2):
    fp1 = np.array(fp1)
    fp2 = np.array(fp2)
    fp1_pos = np.arange(len(fp1))[fp1 == 1]
    fp2_pos = np.arange(len(fp2))[fp2 == 1]
    intersection = np.intersect1d(fp1_pos, fp2_pos)
    union = np.union1d(fp1_pos, fp2_pos)
    return len(intersection) / len(union)


def remove_redundance(lst, cutoff=0.8):
    """
    remove similar small molecules in a smiles list `lst` with tanimoto similarity cutoff;
    """
    lst = list(lst)
    keep_lst = [lst[0]]
    for smi in lst[1:]:
        max_sim = 0.
        for smi_ in keep_lst:
            sim = tanimoto_similarity_smiles(smi, smi_)
            if sim > max_sim:
                max_sim = sim
        
        if max_sim < cutoff:
            keep_lst.append(smi)

    return keep_lst
