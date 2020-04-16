# -*- coding: utf-8 -*-
""" PyTorch Dataset class for reading dataset from hdf5 container

See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from deeprc.deeprc_binary.dataset_converters import DatasetToHDF5


def make_dataloaders(dataset_path: str,
                     split_inds: list = None, cross_validation_fold: int = 0, rnd_seed: int = 0,
                     n_worker_processes: int = 4, batch_size: int = 4,
                     inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                     sample_n_sequences: int = 10000, target_label: str = 'Status', true_class_label_value: str = '+',
                     id_column: str = 'ID',
                     single_class_label_columns: tuple = ('Status',), multi_class_label_columns: tuple = (),
                     sequence_column: str = 'amino_acid', sequence_counts_column: str = 'templates',
                     column_sep: str = '\t', filename_extension: str = '.tsv', h5py_dict: dict = None,
                     verbose: bool = True):
    """Get data loaders for dataset
    
    Get data loaders for training set and training-, validation-, and test-set in evaluation mode
    (=no random subsampling).
    Creates PyTorch data loaders for hdf5 containers or `.tsv`/`.csv` files, which will be converted to hdf5 container
    (see dataset_converters.py).
    If provided,`set_inds` will determine which sample belongs to which split, otherwise random assignment of 
    3/5, 1/5, and 1/5 samples to the three splits is performed.
    See `deeprc/deeprc_binary/predefined_datasets.py` for examples and pre-defined datasets.
    See end of file `deeprc/deeprc_binary/dataset_readers.py` for advanced examples.
    See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
    
    Parameters
    ----------
    dataset_path : str
        Filepath of dataset.
        Can be `.csv`/`.tsv` metadata file or hdf5 container. .csv`/`.tsv` will be converted to hdf5 container.
        See `deeprc_binary.dataset_converters` for conversion. See `datasets/README.md` for supported dataset format.
    split_inds : list of iterable
        Optional: List of iterables of repertoire indices. Each iterable in `split_inds` represents a dataset split.
        For 5-fold cross-validation, `split_inds` should contain 5 lists of repertoire indices, with non-overlapping
        repertoire indices.
        If None, the repertoire indices will be assigned to 5 different splits randomly using `rnd_seed`.
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
    target_label : str
        Name of target class. This is the column name used in the .csv/.tsv metadata file.
    true_class_label_value : str
        String specifying which entries in the `target_label` column should be considered the positive
        class. Other entries will be considered the negative class.
    id_column : str
        Name of column holding the repertoire names in metadata file (only for hdf5-conversion)
    single_class_label_columns : tuple of str
        Tuple of names of columns holding the repertoire status information of single class labels;
        Status may be arbitrary strings (e.g. 'diseased', 'healthy', and 'unknown'); (only for hdf5-conversion)
    multi_class_label_columns : tuple of str
        Tuple of names of columns holding the repertoire status information of multi-class labels;
        Multiple labels per repertoire must be separated using a space character;
        Example status entries: 'class_a class_b', 'class_b', 'class_c class_b class_a'); (only for hdf5-conversion)
    sequence_column : str
        The name of the column that includes the AA sequences (only for hdf5-conversion)
    sequence_counts_column : str
        The name of the column that includes the sequence counts (only for hdf5-conversion)
    column_sep : str
        The column separator (only for hdf5-conversion)
    filename_extension : str
        Filename extension of the metadata and repertoire files (only for hdf5-conversion)
    h5py_dict : dict ot None
        Dictionary with kwargs for creating h5py datasets;
        Defaults to `gzip` compression at level `4` if None; (only for hdf5-conversion)
    verbose : bool
        Activate verbose mode
        
    Returns
    ---------
    trainingset_dataloader: torch.utils.data.DataLoader
        Dataloader for trainingset with active `sample_n_sequences` (=random subsampling/dropout of repertoire
        sequences)
    trainingset_eval_dataloader: torch.utils.data.DataLoader
        Dataloader for trainingset with deactivated `sample_n_sequences`
    validationset_eval_dataloader: torch.utils.data.DataLoader
        Dataloader for validationset with deactivated `sample_n_sequences`
    testset_eval_dataloader: torch.utils.data.DataLoader
        Dataloader for testset with deactivated `sample_n_sequences`
        
    Examples
    ---------
    
    >>> # Get data loaders from hdf5 container
    >>> trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders('dataset.hdf5')
    
    >>> # Get data loaders from text-based dataset (see `deeprc/datasets/README.md` for format)
    >>> trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders('metadata.tsv')
    
    >>> # Use data loaders to train a model
    >>> from deeprc.deeprc_binary.architectures import DeepRC
    >>> from deeprc.deeprc_binary.training import train, evaluate
    >>> model = DeepRC(n_input_features=23, n_output_features=1, max_seq_len=30)
    >>> train(model, trainingset_dataloader=trainingset, trainingset_eval_dataloader=trainingset_eval,
    >>>       validationset_eval_dataloader=validationset_eval)
    >>> roc_auc, bacc, f1, scoring_loss = evaluate(model=model, dataloader=testset_eval)
    """
    #
    # Convert dataset to hdf5 container if no hdf5 container was specifies
    #
    try:
        with h5py.File(dataset_path, 'r') as hf:
            n_repertoires = hf['metadata']['n_samples'][()]
        if verbose:
            print(f"Creating dataloader from file {dataset_path}")
        hdf5_file = dataset_path
    except Exception:
        # Convert to hdf5 container if no hdf5 container was given
        metadata_file = dataset_path
        hdf5_file = '.'.join(metadata_file.split('.')[:-1]) + f".hdf5"
        user_input = None
        while user_input != 'y':
            user_input = input(f"File {dataset_path} is not a hdf container. "
                               f"Should I create an hdf5 container {hdf5_file}? (y/n)")
            if user_input == 'n':
                print("Process aborted by user")
                exit()
        if verbose:
            print(f"Converting: {metadata_file}\n->\n{hdf5_file} @{n_worker_processes} processes")
        converter = DatasetToHDF5(
                metadata_file=metadata_file, id_column=id_column, single_class_label_columns=single_class_label_columns,
                multi_class_label_columns=multi_class_label_columns, sequence_column=sequence_column,
                sequence_counts_column=sequence_counts_column, column_sep=column_sep,
                filename_extension=filename_extension, h5py_dict=h5py_dict, verbose=verbose)
        converter.save_data_to_file(output_file=hdf5_file, n_workers=n_worker_processes)
        with h5py.File(hdf5_file, 'r') as hf:
            n_repertoires = hf['metadata']['n_samples'][()]
        if verbose:
            print("  Done!")
    
    #
    # Create dataset splits
    #
    if split_inds is None:
        if verbose:
            print("\tComputing random splits...")
        n_splits = 5
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
    # Create datasets and dataloaders
    #
    
    # Load dataset into RAM for better performance
    pre_loaded_hdf5_file = None
    if keep_dataset_in_ram:
        with h5py.File(hdf5_file, 'r') as hf:
            pre_loaded_hdf5_file = dict()
            pre_loaded_hdf5_file['seq_lens'] = hf['sampledata']['seq_lens'][:]
            pre_loaded_hdf5_file['counts_per_sequence'] = hf['sampledata']['counts_per_sequence'][:]
            pre_loaded_hdf5_file['amino_acid_sequences'] = hf['sampledata']['amino_acid_sequences'][:]
    
    training_dataset = RepertoireDataReaderBinary(
            hdf5_filepath=hdf5_file, inputformat=inputformat, set_inds=trainingset_inds,
            sample_n_sequences=sample_n_sequences,  target_label=target_label,
            true_class_label_value=true_class_label_value, pre_loaded_hdf5_file=pre_loaded_hdf5_file, verbose=verbose)
    trainingset_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                                         num_workers=n_worker_processes, collate_fn=no_stack_collate_fn)
    
    training_eval_dataset = RepertoireDataReaderBinary(
            hdf5_filepath=hdf5_file, inputformat=inputformat, set_inds=trainingset_inds,
            sample_n_sequences=None,  target_label=target_label,
            true_class_label_value=true_class_label_value, pre_loaded_hdf5_file=pre_loaded_hdf5_file, verbose=verbose)
    trainingset_eval_dataloader = torch.utils.data.DataLoader(training_eval_dataset, batch_size=batch_size,
                                                              shuffle=False, num_workers=1,
                                                              collate_fn=no_stack_collate_fn)
    
    validationset_eval_dataset = RepertoireDataReaderBinary(
            hdf5_filepath=hdf5_file, inputformat=inputformat, set_inds=validationset_inds,
            sample_n_sequences=None,  target_label=target_label,
            true_class_label_value=true_class_label_value, pre_loaded_hdf5_file=pre_loaded_hdf5_file, verbose=verbose)
    validationset_eval_dataloader = torch.utils.data.DataLoader(validationset_eval_dataset, batch_size=1,
                                                                shuffle=False, num_workers=1,
                                                                collate_fn=no_stack_collate_fn)
    
    testset_eval_dataset = RepertoireDataReaderBinary(
            hdf5_filepath=hdf5_file, inputformat=inputformat, set_inds=testset_inds,
            sample_n_sequences=None,  target_label=target_label,
            true_class_label_value=true_class_label_value, pre_loaded_hdf5_file=pre_loaded_hdf5_file, verbose=verbose)
    testset_eval_dataloader = torch.utils.data.DataLoader(testset_eval_dataset, batch_size=1,
                                                          shuffle=False, num_workers=1,
                                                          collate_fn=no_stack_collate_fn)
    
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader


def no_stack_collate_fn(batch_as_list: list):
    """Function to be passed to torch.utils.data.DataLoader as collate_fn
    
    Instead of stacking the sample entries at index no_stack_dims in sample into one tensor, entries will be
    individually converted to tensors and packed into a list instead.
    """
    # Go through all samples, convert entries that are numpy to tensor and put entries in lists
    list_batch = [[torch.from_numpy(sample[entry_i]) for sample in batch_as_list]
                  if isinstance(batch_as_list[0][entry_i], np.ndarray) else
                  [sample[entry_i] for sample in batch_as_list]
                  for entry_i in range(len(batch_as_list[0]))]
    return list_batch


class RepertoireDataReaderBinary(Dataset):
    def __init__(self, hdf5_filepath: str, inputformat: str = 'NCL', set_inds: list = None,
                 sample_n_sequences: int = None, target_label: str = 'Status', true_class_label_value: str = '+',
                 pre_loaded_hdf5_file: dict = None, verbose: bool = True):
        """PyTorch Dataset class for reading dataset from hdf5 container
        
        See dataset_converters.py for conversion of to hdf5 container.
        See make_dataloaders() or predefined_datasets.py for simple loading of datasets via PyTorch data loader.
        
        Parameters
        ----------
        hdf5_filepath : str
            Filepath of dataset
        inputformat : 'NCL' or 'NLC'
            Format of input feature array;
            'NCL' -> (batchsize, channels, seq.length);
            'LNC' -> (seq.length, batchsize, channels);
        set_inds : list or np.ndarray
            Optional: Repertoire indices to load from hdf5 container. Will be sorted automatically.
            If None, all repertoires will be loaded.
        sample_n_sequences : int
            Optional: Random sub-sampling of `sample_n_sequences` sequences per repertoire.
            Number of sequences per repertoire might be smaller than `sample_n_sequences` if repertoire is smaller or
            random indices have been drawn multiple times.
            If None, all sequences will be loaded for each repertoire.
        target_label : str
            Name of target class. This is the column name used in the .csv/.tsv metadata file.
        true_class_label_value : str
            String specifying which entries in the `target_label` column should be considered the positive
            class. Other entries will be considered the negative class.
        pre_loaded_hdf5_file : dict
            Optional: It is faster to load the hdf5 file into the RAM as dictionary instead of keeping it on the disk.
            `pre_loaded_hdf5_file` is the loaded hdf5 file as dictionary.
            If None, the hdf5 file will be read from the disk and consume less RAM.
        verbose : bool
            Activate verbose mode
        """
        self.filepath = hdf5_filepath
        self.inputformat = inputformat
        self.set_inds = set_inds
        self.sample_n_sequences = sample_n_sequences
        self.target_label = target_label
        self.true_class_label_value = true_class_label_value
        self.pre_loaded_hdf5_file = pre_loaded_hdf5_file
        self.verbose = verbose
        
        if self.inputformat not in ['NCL', 'LNC']:
            raise ValueError(f"Unsupported input format {self.inputformat}")
        
        with h5py.File(self.filepath, 'r') as hf:
            metadata = hf['metadata']
            # Add characters for 3 position features to list of AAs
            self.aas = metadata['aas'][()]
            self.aas += ''.join(['<', '>', '^'])
            self.n_features = len(self.aas)
            
            self.label_names = metadata['label_names'][self.target_label][:]
            
            self.true_class_label_inds = np.where(self.label_names == np.array(self.true_class_label_value))[0]
            self.n_classes = 2
            self.stats = metadata['stats'][()]
            
            if set_inds is None:
                self.set_inds = slice(0, None)
                self.n_samples = metadata['n_samples'][()]
            else:
                self.set_inds = sorted(list(set_inds))
                self.n_samples = len(self.set_inds)
            
            self.labels = metadata['labels'][self.target_label][self.set_inds]
            self.labels = self.labels[:, [self.true_class_label_inds[0], self.true_class_label_inds[0]]]
            self.labels[:, 0] = np.logical_not(self.labels[:, 1])
            self.label_names = np.array(['False', self.true_class_label_value], dtype=np.object)
            
            self.sample_keys = metadata['sample_keys'][self.set_inds]
            if pre_loaded_hdf5_file is not None:
                self.sampledata = pre_loaded_hdf5_file
            else:
                self.sampledata = None
            
            sampledata = hf['sampledata']
            self.sample_sequences_start_end = sampledata['sample_sequences_start_end'][self.set_inds]
            self.sample_max_seq_len = sampledata['sample_max_seq_len'][self.set_inds]
            
            self.dataset_max_seq_len = self.sample_max_seq_len.max()
            
            self.class_counts = self.labels.sum(axis=0, dtype=np.float32)
            self.class_probs = self.class_counts / self.n_samples
            
            self._vprint("File Stats:")
            self._vprint("  " + "  \n".join(self.stats.split('; ')))
            self._vprint(f"Used samples: {self.n_samples}")
            self._vprint(f"  samples per class: {self.class_counts}")
            self._vprint(f"  class probabilities: {self.class_probs}")
    
    def get_sample(self, idx):
        """ Return repertoire with index idx from dataset

        Parameters
        ----------
        idx: int
            Index of repertoire to return
        
        Returns
        ---------
        aa_sequences: numpy int8 array
            Repertoire sequences in shape 'NCL' or 'LNC' depending on initialization of class.
            AAs are represented by their index in self.aas.
            Sequences are padded to equal length with value `-1`.
        seq_lens: numpy integer array
            True lengths of sequences in aa_sequences
        counts_per_sequence: numpy integer array
            Counts of sequence in repertoire (as taken from .csv/.tsv repertoire files) as
            `np.log(np.maximum(count, 1))`.
        """
        rnd_gen = np.random.RandomState()
        sample_sequences_start_end = self.sample_sequences_start_end[idx]
        
        if self.sample_n_sequences:
            sample_sequence_inds = np.unique(rnd_gen.randint(
                    low=sample_sequences_start_end[0], high=sample_sequences_start_end[1],
                    size=self.sample_n_sequences))
        else:
            sample_sequence_inds = np.arange(sample_sequences_start_end[0], sample_sequences_start_end[1])
        
        with h5py.File(self.filepath, 'r') as hf:
            if self.sampledata is not None:
                sampledata = self.sampledata
            else:
                sampledata = hf['sampledata']
                sample_sequence_inds = list(sample_sequence_inds)
            
            seq_lens = sampledata['seq_lens'][sample_sequence_inds]
            sample_max_seq_len = seq_lens.max()
            aa_sequences = sampledata['amino_acid_sequences'][sample_sequence_inds, :sample_max_seq_len]
            counts_per_sequence = np.log(np.maximum(sampledata['counts_per_sequence'][sample_sequence_inds], 1))
        
        if self.inputformat.startswith('LN'):
            aa_sequences = np.swapaxes(aa_sequences, 0, 1)
        return aa_sequences, seq_lens, counts_per_sequence
    
    def inds_to_aa(self, inds: np.array):
        """Convert array of AA indices to character array (see also inds_to_aa_ignore_negative())"""
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
    
    def __getitem__(self, idx):
        label = np.asarray(self.labels[idx], dtype=np.float32)
        sample_id = str(self.sample_keys[idx])
        sequences, seq_lens, counts_per_sequence = self.get_sample(idx)
        return label, sequences, seq_lens, counts_per_sequence, sample_id
    
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


if __name__ == '__main__':
    #
    # Data readers -- advanced examples
    # See `deeprc/deeprc_binary/predefined_datasets.py` for simple examples and pre-defined datasets.
    #
    
    # Convert dataset
    metadata_file = f"../datasets/example_dataset_format/metadata.tsv"
    output_file = f"../datasets/example_dataset_format.hdf5"
    n_worker_processes = 5
    
    print(f"Converting: {metadata_file}\n->\n{output_file} @{n_worker_processes} processes")
    converter = DatasetToHDF5(metadata_file=metadata_file)
    converter.save_data_to_file(output_file=output_file, n_workers=n_worker_processes)
    print("Done")
    
    # Read full dataset (all sequences per repertoire and read from disk)
    print("\n\n")
    print("Read full dataset (all sequences per repertoire and read from disk)")
    print()
    dataset = RepertoireDataReaderBinary(hdf5_filepath=output_file, target_label='Status', true_class_label_value='+')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                             collate_fn=no_stack_collate_fn)
    for data in dataloader:
        label, sequences, seq_lens, counts_per_sequence, sample_id = data
        print(f"sample_id: {sample_id}")
        print(f"label: {label}")
        print(f"sequences: {sequences}")
        print(f"seq_lens: {seq_lens}")
        print(f"counts_per_sequence: {counts_per_sequence}")
        print()
    
    # Read only repertoires with index 0 and 2 from dataset, sample 2 sequences per repertoire and read from disk
    print("\n\n")
    print("Read only repertoires with index 0 and 2 from dataset, sample 2 sequences per repertoire and read from disk")
    print()
    dataset = RepertoireDataReaderBinary(hdf5_filepath=output_file, target_label='Status', true_class_label_value='+',
                                         set_inds=[0, 2], sample_n_sequences=2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                             collate_fn=no_stack_collate_fn)
    for data in dataloader:
        label, sequences, seq_lens, counts_per_sequence, sample_id = data
        print(f"sample_id: {sample_id}")
        print(f"label: {label}")
        print(f"sequences: {sequences}")
        print(f"seq_lens: {seq_lens}")
        print(f"counts_per_sequence: {counts_per_sequence}")
        print()
    
    # Read only repertoires with index 0 and 2 from dataset, sample 2 sequences per repertoire and read from disk, 
    # and pre-load hdf5 file
    print("\n\n")
    print("Read only repertoires with index 0 and 2 from dataset, sample 2 sequences per repertoire and read from disk,"
          "and pre-load hdf5 file")
    print()
    with h5py.File(output_file, 'r') as hf:
        sampledata = dict()
        sampledata['seq_lens'] = hf['sampledata']['seq_lens'][:]
        sampledata['counts_per_sequence'] = hf['sampledata']['counts_per_sequence'][:]
        sampledata['counts_per_sequence'] = hf['sampledata']['counts_per_sequence'][:]
        sampledata['amino_acid_sequences'] = hf['sampledata']['amino_acid_sequences'][:]
    dataset = RepertoireDataReaderBinary(hdf5_filepath=output_file, target_label='Status', true_class_label_value='+',
                                         set_inds=[0, 2], sample_n_sequences=2, pre_loaded_hdf5_file=sampledata)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                             collate_fn=no_stack_collate_fn)
    for data in dataloader:
        label, sequences, seq_lens, counts_per_sequence, sample_id = data
        print(f"sample_id: {sample_id}")
        print(f"label: {label}")
        print(f"sequences: {sequences}")
        print(f"seq_lens: {seq_lens}")
        print(f"counts_per_sequence: {counts_per_sequence}")
        print()
