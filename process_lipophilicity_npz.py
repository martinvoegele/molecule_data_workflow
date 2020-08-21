# Standard modules
from __future__ import print_function, division
import os
import pickle
import pandas as pd
import numpy as np

# Pytorch for data set
import torch
from torch.utils.data import Dataset, DataLoader

# RDkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Molecules dataset definition
from molecules_dataset import *



def create_dataset(data_file, out_dir_name,  
                   elements=None, badlist=None,
                   num_conf=1, order_atoms=False, shuffle_atoms=False, 
                   add_h=False, noise = 0, max_num_at=None ):
    """Creates a molecules dataset and saves it to a pickle file."""
    
    # Create the directory for this data set
    try:
        os.makedirs(out_dir_name)
    except FileExistsError:
        pass

    # Create the data set
    name = 'Lipophilicity'
    id_name = 'smiles'
    labels  = ['exp']
    ds = MoleculesDataset(data_file, labels, name=name, id_name=id_name, 
                          num_conf=num_conf, order_atoms=order_atoms, shuffle_atoms=shuffle_atoms,
                          max_num_at=max_num_at, add_h=add_h, elements=elements, badlist=badlist)
    print('Created dataset with %i molecules.'%len(ds.smiles))
    
    if noise > 0:
        ds.add_noise(noise)
    
    # Dump the entire dataset
    ds_file = open(out_dir_name+'/dataset.pkl', 'wb')
    pickle.dump(ds,ds_file)

    return ds

def write_dataset(ds, out_dir_name, train_split=None, vali_split=0.1, test_split=0.1, seed=42, datatypes=None):

    if datatypes == None:
        datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    # Create the directory for this data set
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    # Define indices to split the data set
    test_indices, vali_indices, train_indices = ds.split_randomly(train_split=train_split,vali_split=vali_split,test_split=test_split,random_seed=seed)
    print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(train_indices),len(vali_indices),len(test_indices)))

    # Save the indices for the splitfmt='%i%'
    np.savetxt(out_dir_name+'/indices_test.dat', test_indices, fmt='%i')
    np.savetxt(out_dir_name+'/indices_vali.dat', vali_indices, fmt='%i')
    np.savetxt(out_dir_name+'/indices_train.dat',train_indices,fmt='%i')

    # Save the data sets as compressed numpy files
    test_file_name  = out_dir_name+'/test.npz'
    vali_file_name  = out_dir_name+'/valid.npz'
    train_file_name = out_dir_name+'/train.npz'
    if len(test_indices) > 0: ds.write_compressed(test_file_name, indices=test_indices, datatypes=datatypes )
    if len(vali_indices) > 0: ds.write_compressed(vali_file_name, indices=vali_indices, datatypes=datatypes )
    if len(train_indices) > 0: ds.write_compressed(train_file_name, indices=train_indices, datatypes=datatypes )
                     
    return
    
    
############
# - MAIN - #

if __name__ == "__main__":
    
    data_file = 'datasets_raw/lipophilicity/Lipophilicity.csv'
    dir_name  = 'datasets_processed/lipophilicity'

    # Create the internal data set
    ds = create_dataset(data_file, dir_name)

    # Load the dataset
    ds_file = open(dir_name+'/dataset.pkl', 'rb')
    ds = pickle.load(ds_file)

    # Write the dataset in NPZ format
    write_dataset(ds, dir_name, train_split=0.8, vali_split=0.1, test_split=0.1)


