# -*- coding: utf-8 -*-
"""
Conversion of text-based dataset to hdf5 container

See `deeprc/datasets/README.md` for information on supported dataset formats for custom datasets.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
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
    def __init__(self, repertoiresdata_directory: str, sequence_column: str = 'amino_acid',
                 sequence_counts_column: str = 'templates', column_sep: str = '\t', filename_extension: str = '.tsv',
                 sequence_characters: tuple =
                 ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'),
                 exclude_rows: tuple = (), include_rows: dict = (),  h5py_dict: dict = None,
                 verbose: bool = True):
        """Converts dataset consisting of multiple `.tsv`/`.csv` repertoire files to optimized hdf5 container
        
        Converts dataset consisting of multiple `.tsv`/`.csv` repertoire files to optimized hdf5 container.
        `repertoiresdata_directory` and its subdirectories will be searched for repertoire files, ending in
        `filename_extension`.
        
        Repertoire files must contain:
        - 1 column `sequence_column` that holds the sequences as strings (each sequence is an instance)
        - 1 column `sequence_counts_column` holding information on the number of occurences of the individual sequences
         in the repertoire.
        
        See `datasets/README.md` for more information and examples on supported dataset structures.
        
        Parameters
        ----------
        repertoiresdata_directory : str
            Directory containing the repertoire files of the dataset;
            `repertoiresdata_directory` and its subdirectories will be searched for filenames ending in
            `filename_extension` as repertoire files;
        sequence_column : str
            The name of the column that includes the AA sequences
        sequence_counts_column : str
            The name of the column that includes the sequence counts;
            If None, all sequences will be assigned a count of 1;
        column_sep : str
            The column separator
        filename_extension : str
            Filename extension of the repertoire files
        sequence_characters : tuple of str
            A Tuple containing all characters that might be occur in sequences
        exclude_rows : tuple
            Optional identifiers that result in excluding individual rows the repertoire file.
            `include_rows` is applied before `exclude_rows`.
            Format: `exclude_lines_dict = ((column_name, exclude_value), (column_name2, exclude_value2), ...)`.
            Example: `exclude_lines_dict = (('frame_type', 'Out'), ('valid', 'False'))` will exclude all rows that
            have a value 'Out' in column 'frame_type' or value 'False' in column 'valid'.
        include_rows : tuple
            Optional identifiers that result in including individual rows the repertoire file.
            `include_rows` is applied before `exclude_rows`.
            Format: `include_rows = ((column_name, include_value), (column_name2, include_value2), ...)`.
            Example: `include_rows = (('frame_type', 'In'), ('valid', 'True')` will only include rows that
            have value 'In' in column 'frame_type' or value 'True' in column 'valid'.
        h5py_dict : dict ot None
            Dictionary with kwargs for creating h5py datasets;
            Defaults to `gzip` compression at level `4` if None;
        verbose : bool
            Activate verbose mode
        
        Examples
        ----------
        >>> n_worker_processes = 5
        >>> repertoiresdata_directory = f"datasets/example_dataset_format/repertoires"
        >>> output_file = f"datasets/example_dataset_format/repertoires.hdf5"
        >>> print(f"Converting: {repertoiresdata_directory} to {output_file}")
        >>> converter = DatasetToHDF5(repertoiresdata_directory=repertoiresdata_directory)
        >>> converter.save_data_to_file(output_file=output_file, n_workers=n_worker_processes)
        >>> print("  Done!")
        """
        self.repertoiresdata_directory = repertoiresdata_directory
        self.sequence_column = sequence_column
        self.sequence_counts_column = sequence_counts_column
        self.col_sep = column_sep
        self.filename_extension = filename_extension
        self.exclude_rows = exclude_rows
        self.include_rows = include_rows
        self.h5py_dict = h5py_dict if h5py_dict is not None else dict(compression="gzip", compression_opts=4,
                                                                      chunks=True, shuffle=True)
        self.verbose = verbose
        
        # Define AA characters
        self.aas = sequence_characters
        self.aa_ind_dict = OrderedDict(zip(self.aas, range(len(self.aas))))
        self.n_aa = len(self.aas)
        
        # Search for repertoire files
        self._vprint(f"Searching for repertoire files in {self.repertoiresdata_directory}")
        self.repertoire_files = sorted(glob.glob(os.path.join(self.repertoiresdata_directory, "**",
                                                              f"*{self.filename_extension}"),
                                                 recursive=True))
        self.repertoire_files = [rf for rf in self.repertoire_files if not os.path.isdir(rf)]
        self.sample_keys = [os.path.basename(filename) for filename in self.repertoire_files]
        self.n_samples = len(self.sample_keys)
        
        # Check if filenames are unique
        unique_keys, counts = np.unique(self.sample_keys, return_counts=True)
        if np.any(counts != 1):
            raise ValueError(f"Repertoire filenames must be unique but {unique_keys[counts != 1]} "
                             f"wer found {counts[counts != 1]} times")
        self._vprint(f"\tLocated {self.n_samples} repertoire files")
        
        self.seq_lens = None
        
    def filter_repertoire_sequences(self, repertoire_data: pd.DataFrame):
        """Filter repertoire sequences based on exclusion and inclusion criteria and valid sequence characters"""
        if len(self.exclude_rows) or len(self.include_rows):
            if len(self.include_rows):
                rows_mask = np.zeros_like(repertoire_data[self.sequence_column].values, dtype=np.bool)
                for incl_col, incl_val in self.include_rows:
                    rows_mask = np.logical_or(rows_mask, repertoire_data[incl_col].values == incl_val)
            else:
                rows_mask = np.ones_like(repertoire_data[self.sequence_column].values, dtype=np.bool)
            for excl_col, excl_val in self.exclude_rows:
                rows_mask = np.logical_and(rows_mask, repertoire_data[excl_col].values != excl_val)
            repertoire_data = repertoire_data[rows_mask]
        
        # Filter out entries with invalid characters
        sequences_str = repertoire_data[self.sequence_column].values
        repertoire_data = repertoire_data[[all([True if c in self.aas else False for c in str(seq)])
                                           for seq in sequences_str]]
        return repertoire_data
    
    def _get_repertoire_sequence_lengths(self, filename):
        """Read repertoire file and determine the number of sequences and validity"""
        try:
            repertoire_data = pd.read_csv(filename, sep=self.col_sep, index_col=False,
                                          keep_default_na=False, header=0, low_memory=False)
            
            # Filter out invalid or excluded/not included sequences
            repertoire_data = self.filter_repertoire_sequences(repertoire_data)
            
            # Get sequence counts
            if self.sequence_counts_column is None:
                counts_per_sequence = np.ones_like(repertoire_data[self.sequence_column].values, dtype=np.int)
            else:
                try:
                    counts_per_sequence = np.asarray(repertoire_data[self.sequence_counts_column].values, dtype=np.int)
                except ValueError:
                    counts_per_sequence = repertoire_data[self.sequence_counts_column].values
                    counts_per_sequence[counts_per_sequence == 'null'] = 0
                    counts_per_sequence = np.asarray(counts_per_sequence, dtype=np.int)
                
                # Set sequence counts < 1 to 1
                if counts_per_sequence.min() < 1:
                    self._vprint(f"Warning: template count of < 1 found in sample {filename} -> changed to 1!")
                    sys.stdout.flush()
                    counts_per_sequence[counts_per_sequence < 0] = 1
            
            seq_lens = np.array([len(sequence) for sequence in repertoire_data[self.sequence_column]], dtype=np.int)
            n_sequences = len(repertoire_data)
            
            # Calculate sequence length stats
            min_seq_len = seq_lens.min()
            max_seq_len = seq_lens.max()
            avg_seq_len = ((seq_lens * counts_per_sequence) / counts_per_sequence.sum()).sum()
        except Exception as e:
            print(f"Failure in file {filename}")
            raise e
        return counts_per_sequence, seq_lens, min_seq_len, max_seq_len, avg_seq_len, n_sequences
    
    def _read_aa_sequence(self, filename):
        """Read sequences of repertoire file and convert to numpy int8 array"""
        try:
            repertoire_data = pd.read_csv(filename, sep=self.col_sep, index_col=False,
                                          keep_default_na=False, header=0, low_memory=False)

            # Filter out invalid or excluded/not included sequences
            repertoire_data = self.filter_repertoire_sequences(repertoire_data)
            
            # Get max. sequence length
            sequences_str = repertoire_data[self.sequence_column].values
            seq_lens = np.array([len(sequence) for sequence in sequences_str])
            max_seq_len = seq_lens.max()
            
            # Convert AA strings to numpy int8 array (padded with -1)
            amino_acid_sequences = np.full(shape=(len(sequences_str), max_seq_len), dtype=np.int8, fill_value=-1)
            for i, sequence_str in enumerate(sequences_str):
                amino_acid_sequences[i, :seq_lens[i]] = [self.aa_ind_dict[aa] for aa in sequence_str]
        except Exception as e:
            print(f"\n\n\nFailure in file {filename}\n\n\n")
            raise e
        return amino_acid_sequences
    
    def save_data_to_file(self, output_file: str, n_workers: int = 50, large_repertoires: bool = False):
        """ Read repertoire files and convert dataset to hdf5 container
         
         Set `large_repertoires` to True for large repertoire files if you experience memory problems during
         multiprocessing.
        
        Parameters
        ----------
        output_file : str
            File-path of hdf5 output file to create.
            Warning: If this file already exists, it will be overwritten!
        n_workers : int
            Number of parallel worker processes
        large_repertoires : bool
            Very large repertoire files might cause memory errors during multiprocessing. Set `large_repertoires` to
            True if you experience such error messages (e.g. "... < number < ..." errors).
        """
        self._vprint(f"Saving dataset to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, 'w') as hf:
            # Get number of sequences and check validity
            with multiprocessing.Pool(processes=n_workers) as pool:
                samples_infos = []
                for worker_rets in tqdm.tqdm(pool.imap(self._get_repertoire_sequence_lengths, self.repertoire_files),
                                             desc='Getting n_sequences per repertoire',
                                             total=len(self.repertoire_files)):
                    samples_infos.append(worker_rets)
            
            (counts_per_sequence, seq_lens, min_seq_len, max_seq_len, avg_seq_len,
             n_sequences_per_sample) = zip(*samples_infos)
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
            self.seq_lens = seq_lens
            
            # Get AA sequences and store in one pre-allocated numpy int8 array (padding with -1)
            amino_acid_sequences = np.full(shape=(n_sequences_per_sample.sum(), sample_max_seq_len.max()),
                                           dtype=np.int8, fill_value=-1)
            
            with multiprocessing.Pool(processes=n_workers) as pool:
                if large_repertoires:
                    mapping_function = map
                else:
                    mapping_function = pool.imap
                for sample_i, amino_acid_sequence_sample in tqdm.tqdm(enumerate(mapping_function(
                        self._read_aa_sequence, self.repertoire_files)),
                        desc='Reading repertoire sequences', total=len(self.repertoire_files)):
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
            group.create_dataset('sequence_counts', data=counts_per_sequence, **self.h5py_dict)
            group.create_dataset('sequences', data=amino_acid_sequences, dtype=np.int8, **self.h5py_dict)
            metadata_group = hf.create_group('metadata')
            metadata_group.create_dataset('sample_keys', data=np.array(self.sample_keys, dtype=object),
                                          dtype=h5py.special_dtype(vlen=str), **self.h5py_dict)
            metadata_group.create_dataset('n_samples', data=self.n_samples)
            metadata_group.create_dataset('aas', data=''.join(self.aas))
            metadata_group.create_dataset('stats', data=self._get_stats())
        
        # Create a small log-file with information about the dataset
        with open(output_file + 'info', 'w') as lf:
            print(f"Input: {self.repertoiresdata_directory}", file=lf)
            print(f"Output: {output_file}", file=lf)
            print(f"{self._get_stats()}\n", file=lf)
    
    def _get_stats(self):
        """Get full dataset stats as string"""
        stat_str = []
        stat_str += [f"n_samples={self.n_samples}"]
        if self.seq_lens is not None:
            stat_str += [f"max_seq_len={self.seq_lens.max()}"]
            stat_str += [f"avg_seq_len={self.seq_lens.mean()}"]
            stat_str += [f"min_seq_len={self.seq_lens.min()}"]
            stat_str += [f"aa_ind_dict={self.aa_ind_dict}"]
        stat_str = '\n'.join(stat_str)
        return stat_str
    
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


if __name__ == '__main__':
    n_worker_processes = 5
    repertoiresdata_directory = f"datasets/example_dataset_format/repertoires"
    output_file = f"datasets/example_dataset_format/repertoires.hdf5"
    print(f"Converting: {repertoiresdata_directory} to {output_file}")
    converter = DatasetToHDF5(repertoiresdata_directory=repertoiresdata_directory)
    converter.save_data_to_file(output_file=output_file, n_workers=n_worker_processes)
    print("  Done!")
