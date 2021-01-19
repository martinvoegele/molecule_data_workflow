# Standard modules
from __future__ import print_function, division
import os
import pickle
import argparse
import pandas as pd
import numpy as np

# Pytorch for data set
import torch
from torch.utils.data import Dataset, DataLoader

# RDkit 
from rdkit import Chem

# Molecules dataset definition
from molecules_dataset import *



def write_dataset(ds, out_dir_name, split='random', seed=42, element_list=None):

    datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    # Create the directory for this data set
    try: os.mkdir(out_dir_name)
    except FileExistsError: pass

    # Define indices to split the data set
    if split == 'random':
        ind_te, ind_va, ind_tr = ds.split_randomly(train_split=0.8,vali_split=0.1,test_split=0.1,random_seed=seed)
        print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(ind_tr),len(ind_va),len(ind_te)))
    elif split == 'stdev':
        ind_te, ind_va, ind_tr = ds.split_by_list(train_list = 'datasets_raw/aqsoldb/split-by-stdev/training.txt',
                                                  vali_list = 'datasets_raw/aqsoldb/split-by-stdev/validation.txt',
                                                  test_list = 'datasets_raw/aqsoldb/split-by-stdev/test.txt',
                                                  random_seed = seed)

    # Save the indices for the split
    np.savetxt(out_dir_name+'/indices_test.dat', ind_te, fmt='%i')
    np.savetxt(out_dir_name+'/indices_vali.dat', ind_va, fmt='%i')
    np.savetxt(out_dir_name+'/indices_train.dat',ind_tr, fmt='%i')
    
    # Save the SMILES strings for the split
    np.savetxt(out_dir_name+'/smiles_test.txt', np.array(ds.smiles, dtype=str)[ind_te], fmt='%s')
    np.savetxt(out_dir_name+'/smiles_vali.txt', np.array(ds.smiles, dtype=str)[ind_va], fmt='%s')
    np.savetxt(out_dir_name+'/smiles_train.txt',np.array(ds.smiles, dtype=str)[ind_tr], fmt='%s')
    
    # Save the names for the split
    np.savetxt(out_dir_name+'/names_test.txt', np.array(ds.lnum, dtype=str)[ind_te], fmt='%s')
    np.savetxt(out_dir_name+'/names_vali.txt', np.array(ds.lnum, dtype=str)[ind_va], fmt='%s')
    np.savetxt(out_dir_name+'/names_train.txt',np.array(ds.lnum, dtype=str)[ind_tr], fmt='%s')
    
    # Save the data sets as compressed numpy files
    test_file_name  = out_dir_name+'/test.npz'
    vali_file_name  = out_dir_name+'/valid.npz'
    train_file_name = out_dir_name+'/train.npz'
    if len(ind_te) > 0: ds.write_compressed(test_file_name, indices=ind_te, datatypes=datatypes, element_list=element_list, write_bonds=True)
    if len(ind_va) > 0: ds.write_compressed(vali_file_name, indices=ind_va, datatypes=datatypes, element_list=element_list, write_bonds=True)
    if len(ind_tr) > 0: ds.write_compressed(train_file_name,indices=ind_tr, datatypes=datatypes, element_list=element_list, write_bonds=True)
                     
    return
    
    
############
# - MAIN - #

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_rare_elements', action='store_true', 
                        help="Excludes molecules with rare elements" )
    parser.add_argument('--rename_rare_elements', action='store_true',
                        help="Renames rare elements (sets their atomic number to 0)" )
    parser.add_argument('--max_36_at', action='store_true', 
                        help="Excludes molecules with more than 36 atoms" )
    parser.add_argument('--only_with_sd', action='store_true', 
                        help="Uses the std. dev. and excludes data points with std. dev of zero" )
    parser.add_argument('--split', type=str, choices=['random','stdev'], default='random', 
                        help="Split method." )
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed." )
    args = parser.parse_args()
    
    
    dir_name = 'datasets_processed/aqsoldb'

    elements = None
    element_list = None

    if args.only_with_sd:
        data_file = 'datasets_raw/aqsoldb/curated-solubility-dataset-with-SD.csv'
        dir_name += '_withsd'
        sd_names = ['SD']
    else: 
        data_file = 'datasets_raw/aqsoldb/curated-solubility-dataset.csv'
        sd_names = None
    
    if args.no_rare_elements:
        elements = ['C','O','N','Cl','S','F','P','Br','Na']
        dir_name += '_norareelements'

    if args.rename_rare_elements:
        element_list = ['C','H','O','N','Cl','S','F','P','Br','Na','I','K']
        dir_name += '_renamerareelements'

    if args.max_36_at:   
        maxnumat = 36
        dir_name += '_max36at'
    else:
        maxnumat = None
        
    dir_name += '_'+args.split+'-split'
    
    
    # Create the data set
    ds = MoleculesDataset(data_file, ['Solubility'], name = 'AqSolDB',
                          alt_labels = ['logS'], sd_names = sd_names, id_name = 'Name', 
                          num_conf = 1, order_atoms = False, shuffle_atoms = False, max_num_at = maxnumat, 
                          add_h = False, elements = elements, badlist = None)
    print('Created dataset with %i molecules.'%len(ds.smiles))
    
    # Create the directory for this data set
    try: os.makedirs(dir_name)
    except FileExistsError: pass
    
    # Dump the entire dataset
    ds_file = open(dir_name+'/dataset.pkl', 'wb')
    pickle.dump(ds,ds_file)
    
    # Load the dataset
    ds_file = open(dir_name+'/dataset.pkl', 'rb')
    ds = pickle.load(ds_file)
    
    # Write the dataset in NPZ format
    write_dataset(ds, dir_name, split=args.split, seed=args.seed, element_list=element_list)


