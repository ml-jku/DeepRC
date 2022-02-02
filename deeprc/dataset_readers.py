# -*- coding: utf-8 -*-
"""
Reading repertoire datasets from hdf5 containers.

See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
See `deeprc/examples/` for examples.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
import numpy as np
import h5py
import pandas as pd
from typing import Tuple, Callable, Union
import torch
from torch.utils.data import Dataset, DataLoader
from deeprc.dataset_converters import DatasetToHDF5
from deeprc.task_definitions import TaskDefinition


def log_sequence_count_scaling(seq_counts: np.ndarray):
    """Scale sequence counts `seq_counts` using a natural element-wise logarithm. Values `< 1` are set to `1`.
    To be used for `deeprc.dataset_readers.make_dataloaders`.
    
    Parameters
    ----------
    seq_counts
        Sequence counts as numpy array.
    
    Returns
    ---------
    scaled_seq_counts
        Scaled sequence counts as numpy array.
    """
    return np.log(np.maximum(seq_counts, 1))


def no_sequence_count_scaling(seq_counts: np.ndarray):
    """No scaling of sequence counts `seq_counts`. Values `< 0` are set to `0`.
    To be used for `deeprc.dataset_readers.make_dataloaders`.
    
    Parameters
    ----------
    seq_counts
        Sequence counts as numpy array.
    
    Returns
    ---------
    scaled_seq_counts
        Scaled sequence counts as numpy array.
    """
    return np.maximum(seq_counts, 0)


def make_dataloaders(task_definition: TaskDefinition, metadata_file: str, repertoiresdata_path: str,
                     split_inds: list = None, n_splits: int = 5, cross_validation_fold: int = 0, rnd_seed: int = 0,
                     n_worker_processes: int = 4, batch_size: int = 4,
                     inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                     sample_n_sequences: int = 10000,
                     metadata_file_id_column: str = 'ID', metadata_file_column_sep: str = '\t',
                     sequence_column: str = 'amino_acid', sequence_counts_column: str = 'templates',
                     repertoire_files_column_sep: str = '\t', filename_extension: str = '.tsv', h5py_dict: dict = None,
                     sequence_counts_scaling_fn: Callable = no_sequence_count_scaling, verbose: bool = True) \
        -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get data loaders for a dataset
    
    Get data loaders for training set in training mode (with random subsampling) and training-, validation-, and
    test-set in evaluation mode (without random subsampling).
    Creates PyTorch data loaders for hdf5 containers or `.tsv`/`.csv` files, which will be converted to hdf5 containers
    on-the-fly (see dataset_converters.py).
    If provided,`set_inds` will determine which sample belongs to which split, otherwise random assignment of 
    3/5, 1/5, and 1/5 samples to the three splits is performed. Indices in `set_inds` correspond to line indices
    (excluding the header line) in `metadata_file`.
    
    See `deeprc/examples/` for examples with custom datasets and datasets used in papers.
    See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
    
    Parameters
    ----------
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
    metadata_file : str
        Filepath of metadata .tsv file with targets.
    repertoiresdata_path : str
        Filepath of hdf5 file containing repertoire sequence data or filepath of folder containing the repertoire
        `.tsv`/`.csv` files. `.tsv`/`.csv` will be converted to a hdf5 file.
    split_inds : list of iterable
        Optional: List of iterables of repertoire indices. Each iterable in `split_inds` represents a dataset split.
        For 5-fold cross-validation, `split_inds` should contain 5 lists of repertoire indices, with non-overlapping
        repertoire indices.
        Indices in `set_inds` correspond to line indices (excluding the header line) in `metadata_file`.
        If None, the repertoire indices will be assigned to `n_splits` different splits randomly using `rnd_seed`.
    n_splits
        Optional: If `split_inds` is None, `n_splits` random dataset splits for the cross-validation are created.
    cross_validation_fold : int
        Specify the fold of the cross-validation the dataloaders should be computed for.
    rnd_seed : int
        Seed for the random generator to create the random dataset splits. Only used if `split_inds=None`.
    n_worker_processes : int
        Number of background processes to use for converting dataset to hdf5 container and trainingset dataloader.
    batch_size : int
        Number of repertoires per minibatch during training.
    inputformat : 'NCL' or 'NLC'
        Format of input feature array;
        'NCL' -> (batchsize, channels, seq.length);
        'LNC' -> (seq.length, batchsize, channels);
    keep_dataset_in_ram : bool
        It is faster to load the full hdf5 file into the RAM instead of keeping it on the disk.
        If False, the hdf5 file will be read from the disk and consume less RAM.
    sample_n_sequences : int
        Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
        Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
        random indices have been drawn multiple times.
        If None, all sequences will be loaded for each repertoire.
    metadata_file_id_column : str
        Name of column holding the repertoire names in`metadata_file`.
    metadata_file_column_sep : str
        The column separator in `metadata_file`.
    sequence_column : str
        Optional: The name of the column that includes the AA sequences (only for hdf5-conversion).
    sequence_counts_column : str
        Optional: The name of the column that includes the sequence counts (only for hdf5-conversion).
    repertoire_files_column_sep : str
        Optional: The column separator in the repertoire files (only for hdf5-conversion).
    filename_extension : str
        Filename extension of the metadata and repertoire files. (For repertoire files only for hdf5-conversion.)
    h5py_dict : dict ot None
        Dictionary with kwargs for creating h5py datasets;
        Defaults to `gzip` compression at level `4` if None; (only for hdf5-conversion)
    sequence_counts_scaling_fn
        Scaling function for sequence counts. E.g. `deeprc.dataset_readers.log_sequence_count_scaling` or
        `deeprc.dataset_readers.no_sequence_count_scaling`.
    verbose : bool
        Activate verbose mode
        
    Returns
    ---------
    trainingset_dataloader: DataLoader
        Dataloader for trainingset with active `sample_n_sequences` (=random subsampling/dropout of repertoire
        sequences)
    trainingset_eval_dataloader: DataLoader
        Dataloader for trainingset with deactivated `sample_n_sequences`
    validationset_eval_dataloader: DataLoader
        Dataloader for validationset with deactivated `sample_n_sequences`
    testset_eval_dataloader: DataLoader
        Dataloader for testset with deactivated `sample_n_sequences`
    """
    #
    # Convert dataset to hdf5 container if no hdf5 container was specifies
    #
    try:
        with h5py.File(repertoiresdata_path, 'r') as hf:
            n_repertoires = hf['metadata']['n_samples'][()]
        hdf5_file = repertoiresdata_path
    except Exception:
        # Convert to hdf5 container if no hdf5 container was given
        hdf5_file = repertoiresdata_path + ".hdf5"
        user_input = None
        while user_input != 'y':
            user_input = input(f"Path {repertoiresdata_path} is not a hdf container. "
                               f"Should I create an hdf5 container {hdf5_file}? (y/n)")
            if user_input == 'n':
                print("Process aborted by user")
                exit()
        if verbose:
            print(f"Converting: {repertoiresdata_path}\n->\n{hdf5_file} @{n_worker_processes} processes")
        converter = DatasetToHDF5(
                repertoiresdata_directory=repertoiresdata_path, sequence_column=sequence_column,
                sequence_counts_column=sequence_counts_column, column_sep=repertoire_files_column_sep,
                filename_extension=filename_extension, h5py_dict=h5py_dict, verbose=verbose)
        converter.save_data_to_file(output_file=hdf5_file, n_workers=n_worker_processes)
        with h5py.File(hdf5_file, 'r') as hf:
            n_repertoires = hf['metadata']['n_samples'][()]
        if verbose:
            print(f"\tSuccessfully created {hdf5_file}!")
    
    #
    # Create dataset
    #
    if verbose:
        print(f"Creating dataloader from repertoire files in {hdf5_file}")
    full_dataset = RepertoireDataset(metadata_filepath=metadata_file, hdf5_filepath=hdf5_file,
                                     sample_id_column=metadata_file_id_column,
                                     metadata_file_column_sep=metadata_file_column_sep,
                                     task_definition=task_definition, keep_in_ram=keep_dataset_in_ram,
                                     inputformat=inputformat, sequence_counts_scaling_fn=sequence_counts_scaling_fn)
    n_samples = len(full_dataset)
    if verbose:
        print(f"\tFound and loaded a total of {n_samples} samples")
    
    #
    # Create dataset split indices
    #
    if split_inds is None:
        if verbose:
            print("Computing random split indices")
        n_repertoires_per_split = int(n_repertoires / n_splits)
        rnd_gen = np.random.RandomState(rnd_seed)
        shuffled_repertoire_inds = rnd_gen.permutation(n_repertoires)
        split_inds = [shuffled_repertoire_inds[s_i*n_repertoires_per_split:(s_i+1)*n_repertoires_per_split]
                      if s_i != n_splits-1 else
                      shuffled_repertoire_inds[s_i*n_repertoires_per_split:]  # Remaining repertoires to last split
                      for s_i in range(n_splits)]
    else:
        split_inds = [np.array(split_ind, dtype=np.int) for split_ind in split_inds]
    
    if cross_validation_fold >= len(split_inds):
        raise ValueError(f"Demanded `cross_validation_fold` {cross_validation_fold} but only {len(split_inds)} splits "
                         f"exist in `split_inds`.")
    testset_inds = split_inds.pop(cross_validation_fold)
    validationset_inds = split_inds.pop(cross_validation_fold-1)
    trainingset_inds = np.concatenate(split_inds)
    
    #
    # Create datasets and dataloaders for splits
    #
    if verbose:
        print("Creating dataloaders for dataset splits")
    
    training_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=trainingset_inds, sample_n_sequences=sample_n_sequences)
    trainingset_dataloader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker_processes,
            collate_fn=no_stack_collate_fn)

    training_eval_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=trainingset_inds, sample_n_sequences=None)
    trainingset_eval_dataloader = DataLoader(
            training_eval_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=no_stack_collate_fn)
    
    validationset_eval_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=validationset_inds, sample_n_sequences=None)
    validationset_eval_dataloader = DataLoader(
            validationset_eval_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=no_stack_collate_fn)
    
    testset_eval_dataset = RepertoireDatasetSubset(
            dataset=full_dataset, indices=testset_inds, sample_n_sequences=None)
    testset_eval_dataloader = DataLoader(
            testset_eval_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=no_stack_collate_fn)
    
    if verbose:
        print("\tDone!")
    
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader


def no_stack_collate_fn(batch_as_list: list):
    """Function to be passed to `torch.utils.data.DataLoader` as `collate_fn`
    
    Instead of stacking the samples in a minibatch into one torch.tensor object, sample entries will be individually
    converted to torch.tensor objects and packed into a list instead.
    Objects that could not be converted to torch.tensor objects are packed into a list without conversion.
    """
    # Go through all samples, convert entries that are numpy to tensor and put entries in lists
    list_batch = [[torch.from_numpy(sample[entry_i]) for sample in batch_as_list]
                  if isinstance(batch_as_list[0][entry_i], np.ndarray) else
                  [sample[entry_i] for sample in batch_as_list]
                  for entry_i in range(len(batch_as_list[0]))]
    return list_batch


def str_or_byte_to_str(str_or_byte: Union[str, bytes], decoding: str = 'utf8') -> str:
    """Convenience function to increase compatibility with different h5py versions"""
    return str_or_byte.decode(decoding) if isinstance(str_or_byte, bytes) else str_or_byte


class RepertoireDataset(Dataset):
    def __init__(self, metadata_filepath: str, hdf5_filepath: str, inputformat: str = 'NCL',
                 sample_id_column: str = 'ID', metadata_file_column_sep: str = '\t',
                 task_definition: TaskDefinition = None,
                 keep_in_ram: bool = True, sequence_counts_scaling_fn: Callable = no_sequence_count_scaling,
                 sample_n_sequences: int = None, verbose: bool = True):
        """PyTorch Dataset class for reading repertoire dataset from metadata file and hdf5 file
        
        See `deeprc.dataset_readers.make_dataloaders` for simple loading of datasets via PyTorch data loader.
        See `deeprc.dataset_readers.make_dataloaders` or `dataset_converters.py` for conversion of `.tsv`/`.csv` files
         to hdf5 container.
        
        Parameters
        ----------
        metadata_filepath : str
            Filepath of metadata `.tsv`/`.csv` file with targets used by `task_definition`.
        hdf5_filepath : str
            Filepath of hdf5 file containing repertoire sequence data.
        inputformat : 'NCL' or 'NLC'
            Format of input feature array;
            'NCL' -> (batchsize, channels, seq_length);
            'LNC' -> (seq_length, batchsize, channels);
        task_definition: TaskDefinition
            TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for
             examples.
        keep_in_ram : bool
            It is faster to load the hdf5 file into the RAM as dictionary instead of keeping it on the disk.
            If False, the hdf5 file will be read from the disk dynamically, which is slower but consume less RAM.
        sequence_counts_scaling_fn
            Scaling function for sequence counts. E.g. `deeprc.dataset_readers.log_sequence_count_scaling` or
            `deeprc.dataset_readers.no_sequence_count_scaling`.
        sample_n_sequences : int
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, all sequences will be loaded for each repertoire.
            Can be set for individual samples using `sample_n_sequences` parameter of __getitem__() method.
        verbose : bool
            Activate verbose mode
        """
        self.metadata_filepath = metadata_filepath
        self.filepath = hdf5_filepath
        self.inputformat = inputformat
        self.task_definition = task_definition
        self.sample_id_column = sample_id_column
        self.keep_in_ram = keep_in_ram
        self.sequence_counts_scaling_fn = sequence_counts_scaling_fn
        self.metadata_file_column_sep = metadata_file_column_sep
        self.sample_n_sequences = sample_n_sequences
        self.sequence_counts_hdf5_key = 'sequence_counts'
        self.sequences_hdf5_key = 'sequences'
        self.verbose = verbose
        
        if self.inputformat not in ['NCL', 'LNC']:
            raise ValueError(f"Unsupported input format {self.inputformat}")
        
        # Read target data from csv file
        self.metadata = pd.read_csv(self.metadata_filepath, sep=self.metadata_file_column_sep, header=0, dtype=str)
        self.metadata.index = self.metadata[self.sample_id_column].values
        self.sample_keys = np.array([os.path.splitext(k)[-1] for k in self.metadata[self.sample_id_column].values])
        self.n_samples = len(self.sample_keys)
        self.target_features = self.task_definition.get_targets(self.metadata)
        
        # Read sequence data from hdf5 file
        with h5py.File(self.filepath, 'r') as hf:
            metadata = hf['metadata']
            # Add characters for 3 position features to list of AAs
            self.aas = str_or_byte_to_str(metadata['aas'][()])
            self.aas += ''.join(['<', '>', '^'])
            self.n_features = len(self.aas)
            self.stats = str_or_byte_to_str(metadata['stats'][()])
            self.n_samples = metadata['n_samples'][()]
            hdf5_sample_keys = [str_or_byte_to_str(os.path.splitext(k)[-1]) for k in metadata['sample_keys'][:]]
            
            # Mapping metadata sample indices -> hdf5 file sample indices
            unfound_samples = np.array([sk not in hdf5_sample_keys for sk in self.sample_keys], dtype=np.bool)
            if np.any(unfound_samples):
                raise KeyError(f"Samples {self.sample_keys[unfound_samples]} "
                               f"could not be found in hdf5 file. Please add the samples and re-create the hdf5 file "
                               f"or remove the sample keys from the used samples of the metadata file.")
            self.hdf5_inds = np.array([hdf5_sample_keys.index(sk) for sk in self.sample_keys], dtype=np.int)
            
            # Support old hdf5 format and check for missing hdf5 keys
            if self.sequence_counts_hdf5_key not in hf['sampledata'].keys():
                if 'duplicates_per_sequence' in hf['sampledata'].keys():
                    self.sequence_counts_hdf5_key = 'duplicates_per_sequence'
                elif 'counts_per_sequence' in hf['sampledata'].keys():
                    self.sequence_counts_hdf5_key = 'counts_per_sequence'
                else:
                    raise KeyError(f"Could not locate entry {self.sequence_counts_hdf5_key}, which should contains "
                                   f"sequence counts, in hdf5 file. Only found keys {list(hf['sampledata'].keys())}.")
            if self.sequences_hdf5_key not in hf['sampledata'].keys():
                if 'amino_acid_sequences' in hf['sampledata'].keys():
                    self.sequences_hdf5_key = 'amino_acid_sequences'
                else:
                    raise KeyError(f"Could not locate entry {self.sequences_hdf5_key}, which should contains "
                                   f"sequence counts, in hdf5 file. Only found keys {list(hf['sampledata'].keys())}.")
            
            if keep_in_ram:
                sampledata = dict()
                sampledata['seq_lens'] = hf['sampledata']['seq_lens'][:]
                sampledata[self.sequence_counts_hdf5_key] =\
                    np.array(hf['sampledata'][self.sequence_counts_hdf5_key][:], dtype=np.float32)
                if np.any(sampledata[self.sequence_counts_hdf5_key] <= 0):
                    print(f"Warning: Found {(sampledata[self.sequence_counts_hdf5_key] <= 0).sum()} sequences with "
                          f"counts <= 0. They will be handled as specified in the sequence_counts_scaling_fn "
                          f"{sequence_counts_scaling_fn} passed to RepertoireDataset.")
                sampledata[self.sequences_hdf5_key] = hf['sampledata'][self.sequences_hdf5_key][:]
                self.sampledata = sampledata
            else:
                self.sampledata = None
            
            sample_sequences_start_end = hf['sampledata']['sample_sequences_start_end'][:]
            self.sample_sequences_start_end = sample_sequences_start_end[self.hdf5_inds]
            
        self._vprint("File Stats:")
        self._vprint("  " + "  \n".join(self.stats.split('; ')))
        self._vprint(f"Used samples: {self.n_samples}")
    
    def get_sample(self, idx: int, sample_n_sequences: Union[None, int] = None):
        """ Return repertoire with index idx from dataset, randomly sub-/up-sampled to `sample_n_sequences` sequences
        
        Parameters
        ----------
        idx: int
            Index of repertoire to return
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, will use `sample_n_sequences` as specified when creating `RepertoireDataset` instance.
        
        Returns
        ---------
        aa_sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts per sequence in repertoire.
        """
        sample_sequences_start_end = self.sample_sequences_start_end[idx]
        if sample_n_sequences:
            rnd_gen = np.random.RandomState()  # TODO: Add shared memory integer random seed for dropout
            sample_sequence_inds = np.unique(rnd_gen.randint(
                    low=sample_sequences_start_end[0], high=sample_sequences_start_end[1],
                    size=sample_n_sequences))
            if self.sampledata is None:
                # Compatibility for indexing hdf5 file
                sample_sequence_inds = list(sample_sequence_inds)
        else:
            sample_sequence_inds = slice(sample_sequences_start_end[0], sample_sequences_start_end[1])
    
        with h5py.File(self.filepath, 'r') as hf:
            if self.sampledata is not None:
                sampledata = self.sampledata
            else:
                sampledata = hf['sampledata']
            
            seq_lens = sampledata['seq_lens'][sample_sequence_inds]
            sample_max_seq_len = seq_lens.max()
            aa_sequences = sampledata[self.sequences_hdf5_key][sample_sequence_inds, :sample_max_seq_len]
            counts_per_sequence = \
                self.sequence_counts_scaling_fn(sampledata[self.sequence_counts_hdf5_key][sample_sequence_inds])
    
        if self.inputformat.startswith('LN'):
            aa_sequences = np.swapaxes(aa_sequences, 0, 1)
        return aa_sequences, seq_lens, counts_per_sequence
    
    def inds_to_aa(self, inds: np.array):
        """Convert array of AA indices to character array (see also `self.inds_to_aa_ignore_negative()`)"""
        lookup = np.chararray(shape=(len(self.aas),))
        lookup[:] = list(self.aas)
        char_array = lookup[inds]
        return char_array
    
    def inds_to_aa_ignore_negative(self, inds: np.array):
        """Convert array of AA indices to character array, ignoring '-1'-padding to equal sequence length"""
        lookup = np.chararray(shape=(len(self.aas),))
        lookup[:] = list(self.aas)
        char_array = lookup[inds[inds >= 0]].tostring().decode('utf8')
        return char_array
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx, sample_n_sequences: Union[None, int] = None):
        """ Return repertoire with index idx from dataset, randomly sub-/up-sampled to `sample_n_sequences` sequences
        
        Parameters
        ----------
        idx: int
            Index of repertoire to return
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, will use `sample_n_sequences` as specified when creating `RepertoireDataset` instance.
        
        Returns
        ---------
        target_features: numpy float32 array
            Target feature vector.
        sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts per sequence in repertoire.
        sample_id: str
            Sample ID.
        """
        target_features = self.target_features[idx]
        sample_id = str(self.sample_keys[idx])
        if sample_n_sequences is None:
            sample_n_sequences = self.sample_n_sequences
        sequences, seq_lens, counts_per_sequence = self.get_sample(idx, sample_n_sequences)
        return target_features, sequences, seq_lens, counts_per_sequence, sample_id
    
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class RepertoireDatasetSubset(Dataset):
    def __init__(self, dataset: RepertoireDataset, indices: Union[list, np.ndarray], sample_n_sequences: int = None):
        """Create subset of `deeprc.dataset_readers.RepertoireDataset` dataset
        
        Parameters
        ----------
        dataset
            A `deeprc.dataset_readers.RepertoireDataset` dataset instance
        indices
            List of indices that the subset of `dataset` should contain
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, all sequences will be loaded as specified in `dataset`.
            Can be set for individual samples using `sample_n_sequences` parameter of __getitem__() method.
        """
        self.indices = np.asarray(indices, dtype=np.int)
        self.sample_n_sequences = sample_n_sequences
        self.repertoire_reader = dataset
        
        self.inds_to_aa = self.repertoire_reader.inds_to_aa
        self.aas = self.repertoire_reader.aas
        self.inds_to_aa_ignore_negative = self.repertoire_reader.inds_to_aa_ignore_negative
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx, sample_n_sequences: Union[None, int] = None):
        """ Return repertoire with index idx from dataset, randomly sub-/up-sampled to `sample_n_sequences` sequences
        
        Parameters
        ----------
        idx: int
            Index of repertoire to return
        sample_n_sequences : int or None
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, will use `sample_n_sequences` as specified when creating `RepertoireDatasetSubset` instance.
        
        Returns
        ---------
        target_features: numpy float32 array
            Target feature vector.
        sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts per sequence in repertoire.
        sample_id: str
            Sample ID.
        """
        if sample_n_sequences is None:
            sample_n_sequences = self.sample_n_sequences
        target_features, sequences, seq_lens, counts_per_sequence, sample_id = \
            self.repertoire_reader.__getitem__(self.indices[idx], sample_n_sequences=sample_n_sequences)
        return target_features, sequences, seq_lens, counts_per_sequence, sample_id
