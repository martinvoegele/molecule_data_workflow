# Standard modules
from __future__ import print_function, division
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# needed to make copies of data sets and variables
import copy

# Pytorch for data set
import torch
from torch.utils.data import Dataset, DataLoader

# RDkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Molecules Dataset 
from molecules_dataset import *




def write_dataset(ds, out_dir_name, split='random', seed=42):

    datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    # Create the directory for this data set
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    # Define indices to split the data set
    if args.split=='random':
        print('Writing dataset with random split.')
        ind_te, ind_va, ind_tr = ds.split_randomly(train_split=0.8, vali_split=0.1, test_split=0.1, random_seed=seed)
        print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(ind_tr),len(ind_va),len(ind_te)))
    elif args.split=='random_small':
        print('Writing small dataset with random split.')
        ind_te, ind_va, ind_tr = ds.split_randomly(train_split=0.1, vali_split=0.1, test_split=0.1, random_seed=seed)
        print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(ind_tr),len(ind_va),len(ind_te)))
    elif  args.split=='temporal':
        print('Writing dataset with temporal split.')
        temp_split_dir = 'datasets_raw/'+ds.name+'-curated/split-temporal'
        ind_te, ind_va, ind_tr = ds.split_by_list(train_list = temp_split_dir+'/training.txt', 
                                                  vali_list  = temp_split_dir+'/validation.txt', 
                                                  test_list  = temp_split_dir+'/test.txt',
                                                  random_seed = seed)
        print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(ind_tr),len(ind_va),len(ind_te)))
    else:
        print('No valid split selected. No NPZ files were written.')
    ind_tv = np.concatenate([ind_tr, ind_va])

    # Save the indices for the split
    np.savetxt(out_dir_name+'/indices_test.dat', ind_te, fmt='%i')
    np.savetxt(out_dir_name+'/indices_vali.dat', ind_va, fmt='%i')
    np.savetxt(out_dir_name+'/indices_train.dat',ind_tr, fmt='%i')
    np.savetxt(out_dir_name+'/indices_trval.dat',ind_tv, fmt='%i')

    # Save the data sets as SDF+CSV files
    if len(ind_te) > 0: ds.write_sdf_dataset( out_dir_name+'/test', indices=ind_te )
    if len(ind_va) > 0: ds.write_sdf_dataset( out_dir_name+'/valid',indices=ind_va )
    if len(ind_tr) > 0: ds.write_sdf_dataset( out_dir_name+'/train',indices=ind_tr )
    if len(ind_tv) > 0: ds.write_sdf_dataset( out_dir_name+'/trval',indices=ind_tv )
                     
    return
    
    
############
# - MAIN - #

if __name__ == "__main__":
   
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( '-n',  dest='data_name',  default='cyp2c9', help="name of the dataset as used in the folders" )
    parser.add_argument( '-i',  dest='data_file',  default='datasets_raw/cyp2c9.csv', help="input file" )
    parser.add_argument( '-o',  dest='out_dir',    default='datasets_processed/cyp2c9', help="output directory" )
    parser.add_argument( '-s',  dest='split',      default='random', help="which way to split the dataset" )
    parser.add_argument( '-l',  dest='labels',     default=['TRANSFORMED_ACTIVITY'], help="labels to include", type=str, nargs='+' )
    parser.add_argument( '-id', dest='id_name',    default='L_Numbers', type=str, help="header of the column to use as ID" )
    parser.add_argument( '-ma', dest='max_num_at', default=60, type=int, help="maximum number of atoms per molecule" )
    parser.add_argument( '-bl', dest='badlist',    default=None, help="list with bad molecules to exclude" )
    parser.add_argument( '-el', dest='sel_elem',   default=['C','O','N','Cl','S','F'], help="elements to include", type=str, nargs='+' )
    parser.add_argument( '-nc', dest='num_conf',   default=1, type=int, help="number of conformers per molecule" )
    args = parser.parse_args()
 
    selected_elements = args.sel_elem 
    print('Will include molecules with ...')
    print(' ... up to %i atoms and ...'%int(args.max_num_at)) 
    print(' ... the following elements:',selected_elements)
    
    # Create the data set
    ds = MoleculesDataset(args.data_file, args.labels, name=args.data_name, id_name=args.id_name,
                          elements=selected_elements, max_num_at=args.max_num_at, badlist=args.badlist,
                          num_conf=args.num_conf, order_atoms=False, shuffle_atoms=False, add_h=False)
    print('Created dataset with %i molecules.'%len(ds.smiles))


    # Create the directory for this data set
    try: os.mkdir(args.out_dir)
    except FileExistsError: pass

    # Dump the entire dataset
    ds_file = open(args.out_dir+'/dataset.pkl', 'wb')
    pickle.dump(ds, ds_file)

    # Load the dataset
    ds_file = open(args.out_dir+'/dataset.pkl', 'rb')
    ds = pickle.load(ds_file)

    # Write the dataset in NPZ format
    write_dataset(ds, args.out_dir, split=args.split)


