# -*- coding: utf-8 -*-
"""
Conversion of text-based dataset to hdf5 container

See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.
"""
import os
import sys
from collections import OrderedDict
import glob
import numpy as np
import pandas as pd
import h5py
import tqdm
import multiprocessing


class DatasetToHDF5(object):
    def __init__(self, metadata_file: str, id_column: str = 'ID',
                 single_class_label_columns: tuple = ('status',), multi_class_label_columns: tuple = (),
                 sequence_column: str = 'amino_acid', sequence_counts_column: str = 'templates',
                 column_sep: str = '\t', filename_extension: str = '.tsv',
                 h5py_dict: dict = None, verbose: bool = True):
        """Convert data from raw dataset to hdf5 container
        
        Converts dataset consisting of multiple `.tsv` or `.csv` files to optimized hdf5 container;
        Repertoire files must be located in the same directory or subdirectories of `metadata_file` file;
        See `datasets/README.md` for more information and examples on supported dataset structures;

        Parameters
        ----------
        metadata_file : str
            Input file containing the metadata for the dataset;
            Repertoire files must be located in the same directory or subdirectories of `metadata_file` file;
            Must by `.tsv` or `.csv` file containing:
            1 column holding the repertoire names (Default column name: `ID`) and
            1 column holding the the labels (Default column name: `status`);
        id_column : str
            Name of column holding the repertoire names in metadata file
        single_class_label_columns : tuple of str
            Tuple of names of columns holding the repertoire status information of single class labels;
            Status may be arbitrary strings (e.g. 'diseased', 'healthy', and 'unknown');
        multi_class_label_columns : tuple of str
            Tuple of names of columns holding the repertoire status information of multi-class labels;
            Multiple labels per repertoire must be separated using a space character;
            Example status entries: 'class_a class_b', 'class_b', 'class_c class_b class_a');
        sequence_column : str
            The name of the column that includes the AA sequences
        sequence_counts_column : str
            The name of the column that includes the sequence counts
        column_sep : str
            The column separator
        filename_extension : str
            Filename extension of the metadata and repertoire files
        h5py_dict : dict ot None
            Dictionary with kwargs for creating h5py datasets;
            Defaults to `gzip` compression at level `4` if None;
        verbose : bool
            Activate verbose mode
        
        Examples
        ----------
        >>> n_worker_processes = 5
        >>> metadata_file = f"datasets/example_dataset_format/metadata.tsv"
        >>> output_file = f"datasets/example_dataset_format.hdf5"
        >>> print(f"Converting: {metadata_file} to {output_file}")
        >>> converter = DatasetToHDF5(metadata_file=metadata_file)
        >>> converter.save_data_to_file(output_file=output_file, n_workers=n_worker_processes)
        >>> print("  Done!")
        """
        self.metadata_file = metadata_file
        self.id_column = id_column
        self.single_class_label_columns = single_class_label_columns
        self.multi_class_label_columns = multi_class_label_columns
        self.sequence_column = sequence_column
        self.sequence_counts_column = sequence_counts_column
        self.col_sep = column_sep
        self.filename_extension = filename_extension
        self.h5py_dict = h5py_dict if h5py_dict is not None else dict(compression="gzip", compression_opts=4,
                                                                      chunks=True, shuffle=True)
        self.verbose = verbose
        self.dataset_path = os.path.dirname(metadata_file)
        
        # Define AA characters
        self.aas = sorted(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.aa_ind_dict = OrderedDict(zip(self.aas, range(len(self.aas))))
        self.n_aa = len(self.aas)
        
        # Read metadata file
        self.metadata = pd.read_csv(self.metadata_file, sep=self.col_sep, index_col=False,
                                    keep_default_na=False, header=0, dtype=str)
        self._vprint(f"Extracting data of shape {self.metadata.shape} from csv file...")
        
        # Process sample keys
        self.sample_keys = self.metadata[self.id_column].values
        
        # Search for sample files
        found_sampledata_files = glob.glob(os.path.join(self.dataset_path, "**", f"*{self.filename_extension}"),
                                           recursive=True)
        
        # Check if all IDs in metadata file correspond to a found files
        self.sampledata_files = []
        unfound_samplefiles = []
        for sample_key in self.sample_keys:
            sample_file = [filename for filename in found_sampledata_files if sample_key in os.path.basename(filename)]
            if len(sample_file) == 0:
                unfound_samplefiles.append(sample_key)
            elif len(sample_file) > 1:
                raise ValueError(f"Multiple files found for ID {sample_key}!")
            self.sampledata_files += sample_file
        
        self.n_samples = len(self.sampledata_files)
        self.sample_keys = [os.path.basename(filename) for filename in self.sampledata_files]
        
        # Check if all found files correspond to an ID in metadata file
        file_keys = [os.path.basename(sdf) for sdf in found_sampledata_files]
        unmappable_files = [sdf for sdf in file_keys if sdf not in self.sample_keys]
        if len(unmappable_files):
            self._vprint(f"\tWarning: Unused files {unmappable_files}")
        
        if len(unfound_samplefiles):
            raise ValueError(f"Could not find files for IDs {unfound_samplefiles}!")
        
        self._vprint(f"\tLocated {self.n_samples} files listed in metadata file")
        
        # Extract labels
        label_entries = OrderedDict([(lc, list(self.metadata[lc].astype(str).values))
                                     for lc in self.single_class_label_columns + self.multi_class_label_columns])
        for mlc in self.multi_class_label_columns:
            label_entries[mlc] = [line.split(' ') for line in label_entries[mlc]]
        
        label_names = OrderedDict([(lc, np.unique(np.array(label_entries[lc])))
                                   for lc in self.single_class_label_columns + self.multi_class_label_columns])
        for mlc in self.multi_class_label_columns:
            label_names[mlc] = np.unique(np.array([entry
                                                   for entries in label_entries[mlc]
                                                   for entry in entries if entry is not '']))
        self.label_names = label_names
        self.label_name_ind_dict = OrderedDict([(key, OrderedDict([(k, v) for v, k in enumerate(lns)]))
                                                for key, lns in self.label_names.items()])
        self.labels = OrderedDict()
        self.label_name_counts = OrderedDict()
        for key in self.label_names.keys():
            self.labels[key] = np.zeros((self.n_samples, len(self.label_names[key])), dtype=np.bool)
            self.label_name_counts[key] = OrderedDict()
            for label_name in self.label_names[key]:
                for entry_i, entry in enumerate(label_entries[key]):
                    if isinstance(entry, list):
                        if label_name in entry:
                            self.labels[key][entry_i, self.label_name_ind_dict[key][label_name]] = True
                    else:
                        if label_name == entry:
                            self.labels[key][entry_i, self.label_name_ind_dict[key][label_name]] = True
                self.label_name_counts[key][label_name] = \
                    np.sum(self.labels[key][:, self.label_name_ind_dict[key][label_name]])
        
        self.seq_lens = None
        self._vprint("Metadata summary:")
        self._vprint(" \n".join(self._get_meta_stats().split('; ')))
    
    def _get_repertoire_sequence_lengths(self, filename):
        """Read repertoire file and determine the number of sequences and validity"""
        try:
            valid = True
            sampledata = pd.read_csv(filename, sep=self.col_sep, index_col=False,
                                     keep_default_na=False, header=0, low_memory=False)
            
            # Exclude sequences that have wrong entries in 'frame_type' (see CMV dataset)
            if 'frame_type' in sampledata.columns:
                frame_type_str = sampledata['frame_type'].values
                sampledata = sampledata[frame_type_str == 'In']
            
            # Get all sequences and filter out entries with invalid (non-AA) characters
            sequences_str = sampledata[self.sequence_column].values
            sampledata = sampledata[[all([True if c in self.aas else False for c in str(seq)])
                                     for seq in sequences_str]]
            sequences_str = sampledata[self.sequence_column].values
            
            # Get sequence counts
            try:
                counts_per_sequence = np.asarray(sampledata[self.sequence_counts_column].values, dtype=np.int)
            except ValueError:
                counts_per_sequence = sampledata[self.sequence_counts_column].values
                counts_per_sequence[counts_per_sequence == 'null'] = 0
                counts_per_sequence = np.asarray(counts_per_sequence, dtype=np.int)
                valid = False
            
            # Set -1 sequence counts to 1 (should occur only 3 times in CMV dataset)
            if counts_per_sequence.min() < 0:
                self._vprint(f"Warning: template count of -1 found in sample {filename} -> changed to 1!")
                sys.stdout.flush()
                counts_per_sequence[counts_per_sequence < 0] = 1
            seq_lens = np.array([len(sequence) for sequence in sequences_str], dtype=np.int)
            n_sequences = len(sequences_str)
            
            # Calculate sequence length stats
            min_seq_len = seq_lens.min()
            max_seq_len = seq_lens.max()
            avg_seq_len = ((seq_lens * counts_per_sequence) / counts_per_sequence.sum()).sum()
        except Exception as e:
            print(f"Failure in file {filename}")
            raise e
        return counts_per_sequence, seq_lens, min_seq_len, max_seq_len, avg_seq_len, n_sequences, valid
    
    def _read_aa_sequence(self, filename):
        """Read AAs of repertoire file and convert to numpy int8 array"""
        try:
            sampledata = pd.read_csv(filename, sep=self.col_sep, index_col=False,
                                     keep_default_na=False, header=0, low_memory=False)
            
            # Exclude sequences that have wrong entries in 'frame_type' (see CMV dataset)
            if 'frame_type' in sampledata.columns:
                frame_type_str = sampledata['frame_type'].values
                sampledata = sampledata[frame_type_str == 'In']
            
            # Get all sequences and filter out entries with invalid (non-AA) characters
            sequences_str = sampledata[self.sequence_column].values
            sampledata = sampledata[[all([True if c in self.aas else False for c in str(seq)])
                                     for seq in sequences_str]]
            sequences_str = sampledata[self.sequence_column].values
            
            # Get max. sequence length
            seq_lens = np.array([len(sequence) for sequence in sequences_str])
            max_seq_len = seq_lens.max()
            
            # Convert AA strings to numpy int8 array (padded with -1)
            amino_acid_sequences = np.full(shape=(len(sequences_str), max_seq_len), dtype=np.int8, fill_value=-1)
            for i, sequence_str in enumerate(sequences_str):
                amino_acid_sequences[i, :seq_lens[i]] = [self.aa_ind_dict[aa] for aa in sequence_str]
        except Exception as e:
            print(f"Failure in file {filename}")
            raise e
        return amino_acid_sequences
    
    def save_data_to_file(self, output_file: str, n_workers: int = 50):
        """ Read repertoire files and convert dataset to hdf5 container
        
        Parameters
        ----------
        output_file : str
            File-path of hdf5 output file to create.
            Warning: If this file already exists, it will be overwritten!
        n_workers : int
            Number of parallel worker processes
        """
        self._vprint(f"Saving dataset to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, 'w') as hf:
            # Get number of sequences and check validity
            with multiprocessing.Pool(processes=n_workers) as pool:
                samples_infos = []
                for worker_rets in tqdm.tqdm(pool.imap(self._get_repertoire_sequence_lengths, self.sampledata_files),
                                             desc='Getting n_sequences per sample',
                                             total=len(self.sampledata_files)):
                    samples_infos.append(worker_rets)
            
            (counts_per_sequence, seq_lens, min_seq_len, max_seq_len, avg_seq_len, n_sequences_per_sample,
             valid_samples) = zip(*samples_infos)
            counts_per_sequence = np.concatenate(counts_per_sequence, axis=0)
            seq_lens = np.concatenate(seq_lens, axis=0)
            sample_min_seq_len = np.asarray(min_seq_len, dtype=np.int)
            sample_max_seq_len = np.asarray(max_seq_len, dtype=np.int)
            sample_avg_seq_len = np.asarray(avg_seq_len, dtype=np.float)
            n_sequences_per_sample = np.asarray(n_sequences_per_sample, dtype=np.int)
            sample_sequences_start_end = np.empty(shape=(*n_sequences_per_sample.shape, 2), dtype=np.int)
            sample_sequences_start_end[:, 1] = np.cumsum(n_sequences_per_sample)
            sample_sequences_start_end[1:, 0] = sample_sequences_start_end[:-1, 1]
            sample_sequences_start_end[0, 0] = 0
            valid_samples = np.asarray(valid_samples, dtype=np.bool)
            self.seq_lens = seq_lens
            
            # Get AA sequences and store in one pre-allocated numpy int8 array (padding with -1)
            amino_acid_sequences = np.full(shape=(n_sequences_per_sample.sum(), sample_max_seq_len.max()),
                                           dtype=np.int8, fill_value=-1)
            
            with multiprocessing.Pool(processes=n_workers) as pool:
                for sample_i, amino_acid_sequence_sample in tqdm.tqdm(enumerate(pool.imap(self._read_aa_sequence,
                                                                                          self.sampledata_files)),
                                                                      desc='Reading AA sequences',
                                                                      total=len(self.sampledata_files)):
                    sample_seqs = slice(sample_sequences_start_end[sample_i, 0],
                                        sample_sequences_start_end[sample_i, 1])
                    amino_acid_sequences[sample_seqs, :amino_acid_sequence_sample.shape[1]] = amino_acid_sequence_sample
            
            # Store in hdf5 container
            group = hf.create_group('sampledata')
            group.create_dataset('seq_lens', data=seq_lens, **self.h5py_dict)
            group.create_dataset('sample_sequences_start_end', data=sample_sequences_start_end, **self.h5py_dict)
            group.create_dataset('sample_min_seq_len', data=sample_min_seq_len, **self.h5py_dict)
            group.create_dataset('sample_max_seq_len', data=sample_max_seq_len, **self.h5py_dict)
            group.create_dataset('sample_avg_seq_len', data=sample_avg_seq_len, **self.h5py_dict)
            group.create_dataset('n_sequences_per_sample', data=n_sequences_per_sample, **self.h5py_dict)
            group.create_dataset('counts_per_sequence', data=counts_per_sequence, **self.h5py_dict)
            group.create_dataset('amino_acid_sequences', data=amino_acid_sequences, dtype=np.int8, **self.h5py_dict)
            metadata_group = hf.create_group('metadata')
            metadata_group.create_dataset('valid_samples', data=valid_samples, dtype=np.bool, **self.h5py_dict)
            metadata_group.create_dataset('sample_keys', data=np.array(self.sample_keys, dtype=object),
                                          dtype=h5py.special_dtype(vlen=str), **self.h5py_dict)
            
            labels_group = metadata_group.create_group('labels')
            for key in self.labels.keys():
                labels_group.create_dataset(key, data=self.labels[key], **self.h5py_dict)
            label_names_group = metadata_group.create_group('label_names')
            for key in self.label_names.keys():
                label_names_group.create_dataset(key, data=[x.encode('utf8') for x in self.label_names[key]],
                                                 dtype=h5py.special_dtype(vlen=str))
            label_counts_group = metadata_group.create_group('label_counts')
            for key in self.label_name_counts.keys():
                label_counts_group.create_dataset(key, data=np.array(list(self.label_name_counts[key].values())))
            
            metadata_group.create_dataset('n_samples', data=self.n_samples)
            metadata_group.create_dataset('aas', data=''.join(self.aas))
            metadata_group.create_dataset('n_classes', data=len(self.label_names))
            metadata_group.create_dataset('stats', data=self._get_stats())
        
        # Create a small log-file with information about the dataset
        with open(output_file + 'info', 'w') as lf:
            print(f"Input: {self.metadata_file}", file=lf)
            print(f"Output: {output_file}", file=lf)
            print(f"  " + "  \n".join(self._get_stats().split('; ')), file=lf)
    
    def _get_stats(self):
        """Get full dataset stats as string"""
        stat_str = (f"n_samples={self.n_samples}; max_seq_len={self.seq_lens.max()}; "
                    f" avg_seq_len={self.seq_lens.mean()}; min_seq_len={self.seq_lens.min()}; "
                    f"label_name_counts={self.label_name_counts}; aa_ind_dict={self.aa_ind_dict}; "
                    f"label_name_ind_dict={self.label_name_ind_dict}")
        return stat_str
    
    def _get_meta_stats(self):
        """Get preliminary dataset stats as string"""
        stat_str = (f"n_samples={self.n_samples}; label_name_counts={self.label_name_counts}; "
                    f"aa_ind_dict={self.aa_ind_dict}")
        return stat_str
    
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


if __name__ == '__main__':
    n_worker_processes = 5
    metadata_file = f"datasets/example_dataset_format/metadata.tsv"
    output_file = f"datasets/example_dataset_format.hdf5"
    print(f"Converting: {metadata_file} to {output_file}")
    converter = DatasetToHDF5(metadata_file=metadata_file)
    converter.save_data_to_file(output_file=output_file, n_workers=n_worker_processes)
    print("  Done!")
