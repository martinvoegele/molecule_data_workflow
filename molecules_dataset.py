# Standard modules
from __future__ import print_function, division
import os
from io import StringIO
import sys
import pickle
import pandas as pd
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# RDKit 
from rdkit import Chem
from rdkit.Chem import AllChem



######################
#  Helper functions  #
######################


def read_smiles(smiles,add_h=True):
    """Reads a molecule from a SMILES string
    
    Args:
        add_h (bool): Adds hydrogens. Default: True
        
    """
    
    mol = Chem.MolFromSmiles(smiles)
    
    # Add hydrogens
    if add_h and (mol is not None):
        mol = Chem.AddHs(mol)
    
    return mol


def read_inchi(inchi, add_h=True):
    """Reads a molecule from an INCHI string
    
    Args:
        add_h (bool): Adds hydrogens. Default: True
        
    """

    mol_raw = Chem.MolFromInchi(inchi)

    # Add hydrogens
    if add_h and (mol is not None):
        mol = Chem.AddHs(mol)
    
    return mol


def read_sdf_to_mol(sdf_file,sanitize=True, add_h=False, remove_h=False):
    """Reads a list of molecules from an SDF file.
    
    Args:
        add_h (bool): Adds hydrogens. Default: False
        remove_h (bool): Removes hydrogen. Default: False
        
    """
    
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=sanitize, removeHs=remove_h)
    
    molecules = [mol for mol in suppl]
    
    if add_h:
        for mol in molecules:
            if mol is not None:
                mol = Chem.AddHs(mol, addCoords=True)

    return molecules


def valid_elements(symbols,reference):
    """Tests a list for elements that are not in the reference.
    
    Args:
        symbols (list): The list whose elements to check.
        reference (list): The list containing all allowed elements.
    
    Returns:
        valid (bool): True if symbols only contains elements from the reference.
    
    """
    
    valid = True
    
    if reference is not None:
        for sym in symbols:
            if sym not in reference:
                valid = False
    
    return valid


def reshuffle_atoms(mol):
    """Reshuffles atoms in a molecule.
    
    Args:
        mol (mol): A molecule (RDKit format)
        
    Returns:
        new_mol (mol): A molecule with shuffled atoms.
        
    """  
    
    # Create an array with reshuffled indices
    num_at  = mol.GetNumAtoms()
    indices = np.arange(num_at)
    np.random.shuffle(indices)
    indices = [int(i) for i in indices]
    
    # Renumber the atoms according to the random order 
    new_mol = Chem.RenumberAtoms(mol,indices)
    
    return new_mol



def reorder_atoms(mol):
    """Reorders hydrogen atoms to appear following their heavy atoms in a molecule.
    
    Args:
        mol (Mol): A molecule (RDKit format)
        
    Returns:
        new_Mol (mol): A molecule with hydrogens after the corresponding heavy atoms.
        
    """  
    
    # Create a list with old indices in new order
    indices = []
    for i,at in enumerate(mol.GetAtoms()):
        # For each heavy atom
        if at.GetAtomicNum() != 1:
            # append its index
            indices.append(i)
            # search for its bonds
            for bond in at.GetBonds():
                end_idx = bond.GetEndAtomIdx()
                # check if the bond is to a hydrogen
                if mol.GetAtoms()[end_idx].GetAtomicNum() == 1:
                    # append the hydrogen's index (behind its heavy atom)
                    indices.append(end_idx)
    
    # Renumber the atoms according to the new order 
    new_mol = Chem.RenumberAtoms(mol,indices)
    
    return new_mol


def generate_conformers(mol, n):
    """Generates multiple conformers per molecule.
    
    Args:
        mol (Mol): molecule (RDKit Mol format)
        n (int): number of conformers to generate.
        
    """
    
    indices = AllChem.EmbedMultipleConfs(mol, numConfs=n)
    
    for i in indices:
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=i)
        except:
            print('Failed to optimize conformer #%i.'%(i))
        
    return


def get_coordinates_of_conformers(mol):
    """Reads the coordinates of the conformers.
    
    Args:
        mol (Mol): Molecule in RDKit format.
    
    Returns:
        all_conf_coord (list): Coordinates (one numpy array per conformer) 
        
    """
        
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    
    all_conf_coord = []
    
    for ic, conf in enumerate(mol.GetConformers()): 
        
        xyz = np.empty([mol.GetNumAtoms(),3])
        
        for ia, name in enumerate(symbols):
            
            position = conf.GetAtomPosition(ia)
            xyz[ia]  = np.array([position.x, position.y, position.z])
            
        all_conf_coord.append(xyz)
        
    return all_conf_coord


def get_connectivity_matrix(mol):
    """Generates the connection matrix from a molecule.
    
    Args:
        mol (Mol): a molecule in RDKit format
        
    Returns:
        connect_matrix (2D numpy array): connectivity matrix
    
    """
    
    # Initialization
    num_at = mol.GetNumAtoms()
    connect_matrix = np.zeros([num_at,num_at],dtype=int)
    
    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx()) 
            if bond is not None:
                connect_matrix[a.GetIdx(),b.GetIdx()] = 1
                
    return connect_matrix


def get_bonds_matrix(mol):
    """Provides bond types encoded as single (1.0). double (2.0), triiple (3.0), and aromatic (1.5).
    
    Args:
        mol (Mol): a molecule in RDKit format
        
    Returns:
        connect_matrix (2D numpy array): connectivity matrix
    
    """
    
    # Initialization
    num_at = mol.GetNumAtoms()
    bonds_matrix = np.zeros([num_at,num_at])
    
    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx()) 
            if bond is not None:
                bt = bond.GetBondTypeAsDouble()
                bonds_matrix[a.GetIdx(),b.GetIdx()] = bt
                
    return bonds_matrix


#######################
#  The Dataset Class  #
#######################


class MoleculesDataset(Dataset):
    """Dataset including coordinates and connectivity."""

    def __init__(self, csv_file, col_names, id_name='L_Numbers', sdf_file=None, sd_names=None, name='molecules', 
                 alt_labels=None, elements=None, add_h=True, order_atoms=False, shuffle_atoms=False, 
                 num_conf=1, bond_order=False, max_num_at=None, max_num_heavy_at=None, badlist=None,
                 train_indices_raw=[], vali_indices_raw=[], test_indices_raw=[]):
        """Initializes a data set from a column in a CSV file.
        
        Args:
            csv_file (str): Path to the csv file with the data.
            col_names (str): Name of the columns with the properties to be trained.
            id_name (str): Name of the column used as identifier. Default: 'L Numbers'. 
            sdf_file (str, opt.): Path to the sdf file with structures. Default: None. If None, structures will be generated.
            name (str, opt.): Name of the dataset. Default: 'molecules'.
            alt_labels (list, opt.): Alternative labels for the properties, must be same length as col_names.
            elements (list, opt.): List of permitted elements (Element symbol as str). Default: all elements permitted.
            add_h (bool, opt.): Add hydrogens to the molecules. Default: True.
            order_atoms (bool, opt.): Atoms are ordered such that hydrogens directly follow their heavy atoms. Default: False.
            shuffle_atoms (bool, opt.): Atoms are randomly reshuffled (even if order_atoms is True). Default: False.
            badlist (str, opt.): List of molecules to exclude (first column: L numbers, second column: SMILES)
        
        """
        
        # Read raw data to data frame
        file_type = csv_file.split('.')[-1]
        if file_type == 'csv':
            data_frame = pd.read_csv(csv_file, low_memory=False)
        elif file_type == 'tsv': 
            data_frame = pd.read_table(csv_file, low_memory=False)
        else:
            raise NotImplementedError('Can only read csv or tsv files.')
            
        # Get data from all selected columns
        raw_data   = [ data_frame[col] for col in col_names ]

        # Extract molecule identifiers 
        # (we use L Numbers, but can be anything)
        raw_lnum   = data_frame[id_name]
        
        # Create RDKit molecules from SMILES strings or read them from a file (SDF format)
        if sdf_file is None:
            smiles_key = data_frame.keys()[np.where([k.lower() == 'smiles' for k in data_frame.keys()])[0]]
            if len(smiles_key) == 0: raise KeyError('No column named SMILES or smiles was found.')
            raw_smiles = data_frame[smiles_key[0]]
            raw_mol    = [read_smiles(s,add_h=add_h) for s in raw_smiles]
        else:
            # This assumes that molecules in the SDF file are in the same order as in the CSV file!!!
            raw_mol    = read_sdf_to_mol(sdf_file,add_h=add_h) 
        
        # Read list of L numbers and SMILES of molecules to exclude 
        if badlist is not None:
            bad_lnums, bad_smiles = np.loadtxt(badlist, dtype=str, comments='#').T
        
        # If standard deviations are given ...
        if sd_names is not None: 
            # ... they should uniquely correspond to the respective values
            assert len(sd_names) == len(col_names) 
            raw_sd = [ pd.read_csv(csv_file)[col] for col in sd_names ]
        
        # Intitialize lists for filtered data
        self.smiles    = []
        self.lnum      = []
        self.num_at    = [] # number of atoms in each molecule
        self.symbols   = [] # lists of element symbols of each atom
        self.at_nums   = [] # lists of atomic numbers of each atom
        self.bonds     = []
        self.coords    = []
        self.data      = []
        self.mol       = []
        self.stdev     = []
        
        self.train_idx = []
        self.vali_idx  = []
        self.test_idx  = []
        
        # Name of the dataset (some output options require it)
        self.name = name 
        
        # Labels for the data
        self.columns = col_names
        if alt_labels is None:
            self.labels = col_names
        else:
            self.labels = alt_labels
        
        # Save properties
        self.atoms_ordered  = order_atoms
        self.atoms_shuffled = shuffle_atoms
        self.num_conformers = num_conf
        self.bond_order     = bond_order
        
        # Initialize new index
        new_index = 0

        # For each molecule ...
        for im, m in enumerate(raw_mol):
            
            print('Processing '+str(im)+'/'+str(len(raw_mol))+': '+raw_smiles[im]+'.')
            
            if m is None:
                print('Unable to parse molecule. Excluded from dataset.')
                continue
            
            if badlist is not None and raw_smiles[im] in bad_smiles:
                print('Bad SMILES detected. Excluded from dataset.')
                continue
                
            if badlist is not None and raw_lnum[im] in bad_lnums:
                print('Bad L number detected. Excluded from dataset.')
                continue

            # Read numer of atoms and of heavy atoms
            raw_num_at    = m.GetNumAtoms()
            raw_num_heavy = m.GetNumHeavyAtoms()
            
            # Check if the molecule is small enough
            small_enough = True
            if max_num_at is not None:
                if raw_num_at > max_num_at:
                    small_enough = False
                    print('Too many atoms. Excluded from dataset.')
            if max_num_heavy_at is not None:
                if raw_num_heavy > max_num_heavy_at:
                    small_enough = False
                    print('Too many heavy atoms. Excluded from dataset.')      
            if not small_enough: 
                continue

            # Read all atom names and numbers
            new_symbols = [a.GetSymbol() for a in m.GetAtoms()]
            new_at_nums = [a.GetAtomicNum() for a in m.GetAtoms()]
            
            # Check for undesired elements
            if not valid_elements(new_symbols,elements): 
                print('Contains undesired elements. Excluded from dataset.')
                continue
                    
            # Track error messages (for conformer generation)
            Chem.WrapLogs()
            sio = sys.stderr = StringIO()
            # Add hydrogens for better conformer generation
            m = Chem.AddHs(m)
            # Generate the desired number of conformers
            generate_conformers(m, num_conf)
            if 'ERROR' in sio.getvalue():
                conf_coord = []
                print(sio.getvalue())
            else:
                # Read the list of the coordinates of all conformers
                conf_coord = get_coordinates_of_conformers(m)

            # Remove hydrogen atoms
            if not add_h:
                m = Chem.RemoveHs(m)
            
            # only proceed if successfully generated conformers
            if len(conf_coord) == 0: 
                print('No conformers were generated. Excluded from dataset.')
                continue
            
            # Shuffle the atom order
            if order_atoms:
                m = reorder_atoms(m)
            if shuffle_atoms:
                m = reshuffle_atoms(m)
                
            # Re-read all atom names and numbers
            new_symbols = [a.GetSymbol() for a in m.GetAtoms()]
            new_at_nums = [a.GetAtomicNum() for a in m.GetAtoms()]
                
            if self.bond_order:
                # Generate the connectivity matrix with bond orders encoded as (1,1.5,2,3)
                conmat = get_bonds_matrix(m)
            else:
                # Generate the connectivity matrix without bond orders
                conmat = get_connectivity_matrix(m)
            # Append the molecule
            self.mol.append(m)
            # Append the SMILES
            self.smiles.append( raw_smiles[im] )
            # Append the L number
            self.lnum.append( raw_lnum[im] )
            # Append the number of atoms
            self.num_at.append( raw_num_at )
            # Append all atom names and numbers
            self.symbols.append( new_symbols )
            self.at_nums.append( new_at_nums )
            # Append connectivity matrix and coordinates
            self.bonds.append(conmat)
            self.coords.append(conf_coord)
            # Append the values of the learned quantities
            self.data.append([col[im] for col in raw_data])
            # Append the standard deviations (assign 0 if none are given)
            if sd_names is not None:
                self.stdev.append([col[im] for col in raw_sd])
            else:
                self.stdev.append([0 for col in raw_data])
            print('Added to the dataset.')
            if im in train_indices_raw:
                self.train_idx.append(new_index)
            if im in vali_indices_raw:
                self.vali_idx.append(new_index)
            if im in test_indices_raw:
                self.test_idx.append(new_index)
            new_index += 1                       
              

    def __len__(self):
        """Provides the number of molecules in a data set"""
        
        return len(self.smiles)

    
    def __getitem__(self, idx):
        """Provides a molecule from the data set.
        
        Args:
            idx (int): The index of the desired element.

        Returns:
            sample (dict): The name of a property as a key and the property itself as a value.
        
        """
        
        sample = {'smiles': self.smiles[idx],\
                  'lnumber': self.lnum[idx],\
                  'num_at': self.num_at[idx],\
                  'symbols': self.symbols[idx],\
                  'atomic numbers': self.at_nums[idx],\
                  'bonds': self.bonds[idx],\
                  'coords': self.coords[idx],\
                  'data': self.data[idx],\
                  'stdev': self.stdev[idx]}

        return sample
    
    
    
    def add_noise(self,width,distribution='uniform'):
        """ Adds uniform or Gaussian noise to all coordinates. 
            Coordinates are in nanometers.
        
        Args:
            width (float): The width of the distribution generating the noise.
            distribution(str): The distribution from with to draw. Either normal or uniform. Default: uniform.
            
        """
        
        # For each molecule ...
        for i,mol in enumerate(self.coords):
            # ... for each conformer ...
            for j,conf in enumerate(mol):
                # ... and for each atom
                for k,atom in enumerate(conf):
                    if distribution == 'normal':
                        # add random numbers from a normal distribution.
                        self.coords[i][j][k] += np.random.normal(0.0,width,3)
                    elif distribution == 'uniform':
                        # add random numbers from a uniform distribution.
                        self.coords[i][j][k] += width*(np.random.rand(3) - 0.5)
        
        return
    
    
    def split_randomly(self,train_split=None,vali_split=0.1,test_split=0.1,shuffle=True,random_seed=42):
        """Creates random indices for training, validation, and test splits.
        
        Args:
            vali_split (float): fraction of data used for validation. Default: 0.1
            test_split (float): fraction of data used for testing. Default: 0.1
            shuffle (bool):     indices are shuffled. Default: True
            random_seed (int):  specifies random seed for shuffling. Default: 42
            
        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.
            
        """
        
        dataset_size = len(self)
        indices = np.arange(dataset_size,dtype=int)
        
        # Calculate the numbers of elements per split
        vsplit = int(np.floor(vali_split * dataset_size))
        tsplit = int(np.floor(test_split * dataset_size))
        if train_split is not None:
            train = int(np.floor(train_split * dataset_size))
        else:
            train = dataset_size-vsplit-tsplit
        
        # Shuffle the dataset if desired
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)

        # Determine the indices of each split
        indices_test  = indices[:tsplit]
        indices_vali  = indices[tsplit:tsplit+vsplit]
        indices_train = indices[tsplit+vsplit:tsplit+vsplit+train]
        
        return indices_test, indices_vali, indices_train

    
    def split_by_list(self, train_list='training.txt', vali_list='validation.txt', test_list='test.txt', 
                      identifier='smiles', shuffle=True, random_seed=None):
        """Creates indices for training, validation, and test splits from lists of identifiers (SMILES strings or L Numbers).
        
        Args:
            train_list (float): list of data points used for training. Default: train.txt
            vali_list (float): list of data points used for validation. Default: vali.txt
            test_list (float): list of data points used for testing. Default: test.txt
            identifier (str): field by which to sort the splits ("smiles"/"lnumber")
            shuffle (bool): indices are shuffled. Default: True
            random_seed (int): specifies random seed for shuffling. Default: None
            
        Returns:
            indices_test (int[]):  indices of the test set.
            indices_vali (int[]):  indices of the validation set.
            indices_train (int[]): indices of the training set.
            (NOTE THE ORDER!)
            
        """
        
        dataset_size = len(self)
        indices = np.arange(dataset_size,dtype=int)
        
        # Shuffle the dataset if desired
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        # Load identifier lists
        molid_tr = np.loadtxt(train_list, dtype=str)
        molid_va = np.loadtxt(vali_list, dtype=str)
        molid_te = np.loadtxt(test_list, dtype=str)

        # Get SMILES strings or L Numbers
        if identifier.lower() == 'smiles':
            molid = self.smiles
        elif identifier.lower() == 'lnumber':
            molid = self.lnum
        else:
            raise ValueError('identifier must be smiles or lnumber!')
        
        # Initialize indices
        indices_tr, indices_va, indices_te = [], [], []
        
        # Go through all data points
        for i, idx in enumerate(indices):
            if i%100 == 0: 
                print('Looked up %i out of %i molecules'%(i,len(indices)))
                sys.stdout.flush()
            if molid[idx] in molid_tr:
                indices_tr.append(idx)
            if molid[idx] in molid_va:
                indices_va.append(idx)        
            if molid[idx] in molid_te:
                indices_te.append(idx)        

        # Make arrays
        indices_test  = np.array(indices_te) 
        indices_vali  = np.array(indices_va) 
        indices_train = np.array(indices_tr) 
        
        return indices_test, indices_vali, indices_train
    
    
    def element_statistics(self):
        """ Prints the numbers of molecules containing specific elements of the periodic table.
        """
        
        pte = Chem.GetPeriodicTable()

        el_names = [Chem.PeriodicTable.GetElementSymbol(pte,n) for n in range(1,113)]
        num_el_contained = np.zeros(len(el_names),dtype=int)

        for i,element in enumerate(el_names):
            el_count = np.array( [molsym.count(element) for molsym in self.symbols] )
            el_contained = el_count > 0
            num_el_contained[i] = np.sum(el_contained)

        sortidx = np.argsort(num_el_contained)
        sortidx = np.flip(sortidx)

        el_names = [el_names[i] for i in sortidx]
        num_el_contained = [num_el_contained[i] for i in sortidx]

        for line in np.array([el_names, num_el_contained]).T:
            if int(line[1]) > 0: print('%-2s %5i'%(line[0],int(line[1])))
                
        return el_names, num_el_contained
    
    
    def write_xyz(self,filename,prop_idx=0,indices=None):
        """Writes (a subset of) the data set as xyz file.
        
        Args:
            filename (str):  The name of the output file. 
            prop_idx (int):  The index of the property to be trained for.
            indices (int[]): The indices of the molecules to write data for.
        
        """
    
        # Initialization
        if indices is None: indices = np.arange(len(self))
            
        with open(filename,'w') as out_file:

            # Header (only number of molecules)
            out_file.write(str(len(indices))+'\n')

            # For each molecule ...
            for i in indices:
                sample = self[i]
                # ... for  each conformer ...
                for pos in sample['coords']:
                    # write number of atoms
                    out_file.write(str(sample['num_at'])+'\n')
                    # write property to be trained for (now: only the first one)
                    out_file.write(str(sample['data'][prop_idx])+'\n')
                    # ... and for each atom:
                    for ia in range(sample['num_at']):
                        # write the coordinates.
                        out_file.write("%s %8.5f %8.5f %8.5f\n"%(sample['symbols'][ia], pos[ia,0], pos[ia,1], pos[ia,2]))

        return
    
    
    def write_connectivity_matrices(self,filename,prop_idx=0,indices=None,convert_atom_numbers=False):
        """Writes (a subset of) the data set as connectivity matrices.
        
        Args:
            filename (str):  The name of the output file. 
            prop_idx (int):  The index of the property to be trained for.
            indices (int[]): The indices of the molecules to write data for.
        
        """
        
        # Initialization
        if indices is None: indices = np.arange(len(self))
        
        # Mapping of atom numbers for old Cormorant
        at_num_map = {1:2,6:1,7:4,8:3,9:5,15:6,16:7,17:8,5:9,35:10,53:11,14:12,34:13}
        
        with open(filename,'w') as out_file:

            # Header
            out_file.write('# ' + self.name+' '+self.labels[prop_idx]+'\n')
            out_file.write('\n'.join(self.labels)+'\n')
            out_file.write(str(len(indices))+'\n')

            # Molecule-specific information
            for i,idx in enumerate(indices):
                
                sample = self[idx]

                # numbers of atoms and value of the property
                out_file.write(str(sample['num_at']) + ' ' + str(sample['data'][prop_idx]) + '\n')
                
                # connectivity matrix
                for line in sample['bonds']:
                    pline = ' '.join( str(l) for l in line )
                    out_file.write(pline+'\n')
                    
                # atomic numbers
                atomic_numbers = sample['atomic numbers']
                if convert_atom_numbers:
                    atomic_numbers = [at_num_map[a] for a in atomic_numbers]
                for an in atomic_numbers:
                    out_file.write(str(an)+'\n')
    
    
    def write_compressed(self, filename, indices = None, datatypes = None, write_bonds = False,
                         element_list = None):
        """Writes (a subset of) the data set as compressed numpy arrays.
        Args:
            filename (str):  The name of the output file. 
            indices (int[]): The indices of the molecules to write data for.
        """

        # Define which molecules to use 
        # (counting indices of processed data set)
        if indices is None:
            indices = np.arange(len(self))
            
        # Convert element symbols to atomic numbers
        if element_list is not None:
            pte = Chem.GetPeriodicTable()
            atomic_numbers_list = [ Chem.PeriodicTable.GetAtomicNumber(pte,el) for el in element_list ]
            
        # All charges and position arrays have the same size
        # (the one of the biggest molecule)
        size = np.max( self.num_at )
        
        # Initialize arrays
        num_atoms = np.zeros(len(indices))
        charges   = np.zeros([len(indices),size])
        positions = np.zeros([len(indices),size,3])
        if write_bonds:
            # Are bond orders encoded or not?
            if self.bond_order:
                bond_order_range = np.array([0,1,1.5,2,3])
            else: 
                bond_order_range = np.array([0,1])
            bonds = np.zeros([len(indices)*self.num_conformers,size,size,len(bond_order_range)],dtype=int)
        
        # For each molecule ...
        for j,idx in enumerate(indices):
            # load the data
            sample = self[idx]
            # ... for  each conformer ...
            for pos in sample['coords']:
                # assign per-molecule data
                num_atoms[j] = sample['num_at']
                # ... and for each atom:
                for ia in range(sample['num_at']):
                    new_charge = sample['atomic numbers'][ia]
                    if element_list is not None:
                        # replace all atomic numbers that are not in the list by 121
                        new_charge = new_charge if new_charge in atomic_numbers_list else 121
                    charges[j,ia] = new_charge 
                    positions[j,ia,0] = pos[ia][0] 
                    positions[j,ia,1] = pos[ia][1] 
                    positions[j,ia,2] = pos[ia][2]
                if write_bonds:
                    # add bonds. Bond order is one-hot encoded
                    bonds_matrix  = sample['bonds'] 
                    bonds_one_hot = (bond_order_range == bonds_matrix[...,None]).astype(int)
                    bonds[j,:sample['num_at'],:sample['num_at'],:] = bonds_one_hot

        # Create a dictionary with all the values to save
        save_dict = {}
        # Add the label data (dynamically)
        for ip,prop in enumerate(self.labels):
            selected_data = [self.data[idx] for idx in indices]
            locals()[prop] = [col[ip] for col in selected_data]
            # Use only those quantities that are of one of the defined data types
            if datatypes is not None and np.array(locals()[prop]).dtype in datatypes:
                save_dict[prop] = locals()[prop]

        # Add the atom data
        save_dict['num_atoms'] = num_atoms
        save_dict['charges']   = charges
        save_dict['positions'] = positions
        if write_bonds:
            save_dict['bonds'] = bonds

        # Save as a compressed array 
        np.savez_compressed(filename,**save_dict)
        
        return
    
    
