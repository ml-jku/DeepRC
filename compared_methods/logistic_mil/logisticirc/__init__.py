import ctypes
import h5py
import json
import multiprocessing
import numpy as np
import pickle
import sklearn.metrics as metrics
import sys
import time
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from enum import Enum
from itertools import product, zip_longest
from multiprocessing.sharedctypes import RawArray
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

# Global variable for initialising pre-processing worker processes.
child_vars = {}

# Set sharing strategy for e.g. data loaders of PyTorch (to keep total count of file handles low).
torch.multiprocessing.set_sharing_strategy(r'file_system')


class Atchley(object):
    """
    Solving the protein sequence metric problem [1].

    [1] Atchley, W.R., Zhao, J., Fernandes, A.D. and Drüke, T., 2005. Solving the protein sequence metric problem.
        Proceedings of the National Academy of Sciences, 102(18), pp.6395-6400.
    """

    def __new__(cls) -> r'Atchley':
        """
        Create new Atchley instance, if none is available. Otherwise, return the previously instantiated one.

        :return: Atchley instance
        """
        if not hasattr(cls, r'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        """
        Initialise internal Atchley factor dictionary. The five factors correspond to (in this order):
            1) hydrophobicity
            2) secondary structure
            3) size/mass
            4) codon degeneracy
            5) electric charge
        """
        self.__keys = (r'A', r'C', r'D', r'E', r'F', r'G', r'H', r'I', r'K', r'L',
                       r'M', r'N', r'P', r'Q', r'R', r'S', r'T', r'V', r'W', r'Y')
        self.__factors = {
            r'A': torch.tensor((-0.591, -1.302, -0.733, +1.570, -0.146), dtype=torch.float32),
            r'C': torch.tensor((-1.343, +0.465, -0.862, -1.020, -0.255), dtype=torch.float32),
            r'D': torch.tensor((+1.050, +0.302, -3.656, -0.259, -3.242), dtype=torch.float32),
            r'E': torch.tensor((+1.357, -1.453, +1.477, +0.113, -0.837), dtype=torch.float32),
            r'F': torch.tensor((-1.006, -0.590, +1.891, -0.397, +0.412), dtype=torch.float32),
            r'G': torch.tensor((-0.384, +1.652, +1.330, +1.045, +2.064), dtype=torch.float32),
            r'H': torch.tensor((+0.336, -0.417, -1.673, -1.474, -0.078), dtype=torch.float32),
            r'I': torch.tensor((-1.239, -0.547, +2.131, +0.393, +0.816), dtype=torch.float32),
            r'K': torch.tensor((+1.831, -0.561, +0.533, -0.277, +1.648), dtype=torch.float32),
            r'L': torch.tensor((-1.019, -0.987, -1.505, +1.266, -0.912), dtype=torch.float32),
            r'M': torch.tensor((-0.663, -1.524, +2.219, -1.005, +1.212), dtype=torch.float32),
            r'N': torch.tensor((+0.945, +0.828, +1.299, -0.169, +0.933), dtype=torch.float32),
            r'P': torch.tensor((+0.189, +2.081, -1.628, +0.421, -1.392), dtype=torch.float32),
            r'Q': torch.tensor((+0.931, -0.179, -3.005, -0.503, -1.853), dtype=torch.float32),
            r'R': torch.tensor((+1.538, -0.055, +1.502, +0.440, +2.897), dtype=torch.float32),
            r'S': torch.tensor((-0.228, +1.399, -4.760, +0.670, -2.647), dtype=torch.float32),
            r'T': torch.tensor((-0.032, +0.326, +2.213, +0.908, +1.313), dtype=torch.float32),
            r'V': torch.tensor((-1.337, -0.279, -0.544, +1.242, -1.262), dtype=torch.float32),
            r'W': torch.tensor((-0.595, +0.009, +0.672, -2.128, -0.184), dtype=torch.float32),
            r'Y': torch.tensor((+0.260, +0.830, +3.097, -0.838, +1.512), dtype=torch.float32)
        }
        self.__depth = self.__factors[self.__keys[0]].shape[0]

    def __getitem__(self, item: Union[torch.FloatTensor, str]) -> Union[str, torch.FloatTensor]:
        """
        Return Atchley factors of specified amino acid. If the specified key (amino acid) is not present in the
        underlying dictionary, a corresponding zero-tensor is returned.

        :param item: amino acid for which to get the Atchley factors
        :return: Atchley factors of specified amino acid
        """
        return self.__factors.get(item, torch.zeros((1, 5), dtype=torch.float32))

    def reverse_lookup(self, item: torch.Tensor) -> str:
        """
        The corresponding amino acid with respect to the specified Atchley factors is returned. If the specified
        key (Atchley factor) is not present in the underlying dictionary, a corresponding empty string is returned.

        :param item: Atchley factors for which to get the amino acid
        :return: amino acid of specified Atchley factors
        """
        for key, value in self.__factors.items():
            if item.float().isclose(value, atol=1e-3).any():
                return key
        return r''

    @property
    def keys(self) -> Tuple[str, ...]:
        return self.__keys

    @property
    def depth(self) -> int:
        return self.__depth


class LogisticMILModule(nn.Module):
    """
    Biophysicochemical Motifs in T-cell Receptor Sequences Distinguish Repertoires from Tumor-Infiltrating Lymphocyte
    and Adjacent Healthy Tissue [1].

    [1] Ostmeyer, J., Christley, S., Toby, I.T. and Cowell, L.G., 2019. Biophysicochemical Motifs in T-cell Receptor
        Sequences Distinguish Repertoires from Tumor-Infiltrating Lymphocyte and Adjacent Healthy Tissue.
        Cancer research, 79(7), pp.1671-1680.
    """

    def __init__(self, kmer_size: int = 4):
        """
        Initialise the Ostmeyer model based on logistic regression.
        Note, this class only applies a linear mapping, the final logistic function needs to be applied
        manually afterwards (to enable a numerically more stable training procedure).

        :param kmer_size: size of a k-mer to extract
        """
        super(LogisticMILModule, self).__init__()
        self.linear_mapping = nn.Linear(in_features=Atchley().depth * kmer_size + 1, out_features=1, bias=True)
        with torch.no_grad():
            self.linear_mapping.weight.normal_(mean=0.0, std=1.0 / self.linear_mapping.in_features)
            self.linear_mapping.weight[:, -1].fill_(value=0.0)
            self.linear_mapping.bias.zero_()

    def forward(self, atchley_factors_abundance: torch.Tensor) -> torch.Tensor:
        """
        Apply linear mapping on Atchley factors including relative abundance term.

        :param atchley_factors_abundance: factors according to Atchley et al. including abundance term (b x 21)
        :return: preliminary result of the logistic MIL model (pre-activation)
        """
        return self.linear_mapping(input=atchley_factors_abundance)


class LogisticMILDataReader(data.Dataset):
    """
    Data reader according to logistic MIL (as defined in class <LogisticMILModule>).
    """

    # Class attributes for pre-processing data.
    spawn_method = r'forkserver'
    tasks_per_child = None
    compression_algorithm = r'gzip'
    compression_shuffle = False
    compression_level = 4

    class RelativeAbundance(Enum):
        """
        Enumeration of supported relative abundance terms.
        """
        KMER = r'kmer'
        TCRB = r'tcrb'

        def __str__(self) -> str:
            return self.value

    def __init__(self, file_path: Path, relative_abundance: RelativeAbundance, indices: List[int] = None,
                 unique: bool = False, dtype: torch.dtype = torch.float32):
        """
        Initialise data reader according to logistic MIL (as defined in class <LogisticMILModule>).

        :param file_path: data file to read from (h5py)
        :param relative_abundance: type of relative abundance term as defined in Ostmeyer et al., equations (A) and (B)
        :param indices: indices of repertoires which are considered (all others are ignored)
        :param unique: only include unique kmers on a per sample basis
        :param dtype: type of Tensor to use
        """
        assert Path.exists(file_path) and h5py.is_hdf5(file_path), r'Invalid data file specified!'
        assert (indices is None) or all(_ >= 0 for _ in indices), r'Invalid repertoire indices specified!'

        self.__file_path = file_path
        self.__indices = indices
        self.__dtype = dtype

        with h5py.File(self.__file_path, r'r') as data_file:
            if indices is None:
                self.__size = data_file[r'metadata'][r'n_samples'][()].item()
                self.__indices = range(self.__size)
            else:
                self.__size = len(indices)
                assert data_file[r'metadata'][r'n_samples'][()] >= len(indices), r'Invalid <indices>!'

            # Fetch auxiliary features.
            starts = data_file[r'sampledata'][r'kmer_sequences_start_end'][self.__indices]
            ends = data_file[r'sampledata'][r'kmer_sequences_start_end'][(_ + 1 for _ in self.__indices)]
            try:
                relative_abundance_key = relative_abundance.name.lower()
            except AttributeError:
                raise ValueError(r'Invalid <relative_abundance> specified! Aborting...')
            self.__sample_mean = np.concatenate((
                data_file[r'sampledata'][f'sample_mean'][self.__indices],
                data_file[r'sampledata'][
                    f'relative_abundance_{relative_abundance_key}_mean'][self.__indices].reshape((-1, 1))), axis=1)
            self.__sample_stdv = np.concatenate((
                data_file[r'sampledata'][f'sample_stdv'][self.__indices],
                data_file[r'sampledata'][
                    f'relative_abundance_{relative_abundance_key}_stdv'][self.__indices].reshape((-1, 1))), axis=1)

            # Filter non-unique kmers and compute adapted sample sizes.
            if unique:

                # Define auxiliary variables and functions for unique kmer filtering.
                kmer_sequences: List[Optional[torch.Tensor]] = [None for _ in range(self.__size)]
                progress_bar = tqdm(desc=r'Loading', unit=r'sa', total=len(kmer_sequences), file=sys.stdout)

                def _kmer_sequence_unique_callback(_unique_result: Tuple[np.ndarray, int]) -> None:
                    kmer_sequences[_unique_result[1]] = torch.tensor(_unique_result[0], dtype=self.__dtype)
                    progress_bar.update(1)

                with multiprocessing.get_context(method=self.spawn_method).Pool(
                        processes=multiprocessing.cpu_count(), maxtasksperchild=self.tasks_per_child) as sample_pool:

                    sample_futures = []
                    for sample_index, (start, end, index) in enumerate(zip(starts, ends, self.__indices)):
                        current_repertoire = np.concatenate((
                            data_file[r'sampledata'][r'kmer_sequences'][start:end],
                            data_file[r'sampledata'][f'relative_abundance_{relative_abundance_key}'][index][
                                data_file[r'sampledata'][r'kmer_indices'][start:end]].reshape(-1, 1)), axis=1)
                        sample_futures.append(sample_pool.apply_async(
                            self.kmer_sequence_unique_worker, (current_repertoire, sample_index),
                            callback=_kmer_sequence_unique_callback,
                            error_callback=lambda _: print(f'Failed to process sample {sample_index + 1}!\n{_}\n')))

                    # Wait for remaining tasks to be finished (remove futures to free any remaining alloc. resources).
                    while True:
                        finished_futures = sorted([
                            index for index, _ in enumerate(sample_futures) if _.ready()], reverse=True)
                        for finished_future in finished_futures:
                            del sample_futures[finished_future]
                        if len(sample_futures) <= 0:
                            break
                        time.sleep(10)
                progress_bar.refresh()
                progress_bar.close()
                self.__kmer_sequences = kmer_sequences
            else:
                self.__kmer_sequences = [torch.tensor(np.concatenate((
                    data_file[r'sampledata'][r'kmer_sequences'][start:end],
                    data_file[r'sampledata'][f'relative_abundance_{relative_abundance_key}'][index][
                        data_file[r'sampledata'][r'kmer_indices'][start:end]].reshape(-1, 1)
                ), axis=1), dtype=self.__dtype) for start, end, index in zip(starts, ends, self.__indices)]
            self.__sample_sizes = np.asarray([_.shape[0] for _ in self.__kmer_sequences])
            self.__targets = [
                torch.from_numpy(data_file['metadata'][r'labels'][index].reshape(1)).to(dtype=self.__dtype)
                for index in self.__indices]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch entry from data set specified by an index.

        :param index: index of entry to fetch
        :return: data set entry with specified index
        """
        return self.__kmer_sequences[index], self.__targets[index]

    def __len__(self) -> int:
        """
        Fetch the size of the data set. It is defined as the number of repertoires (not sequences).

        :return: size of data set in terms of number of entries
        """
        return self.__size

    @classmethod
    def adapt(cls, file_path: Path, store_path: Path, kmer_size: int = 4, num_workers: int = 0,
              dtype: Type = np.float32) -> None:
        """
        Adapt data set to be compatible with logistic MIL.

        :param file_path: data file to read from (h5py)
        :param store_path: path to resulting data file (h5py)
        :param kmer_size: size of a k-mer to extract
        :param num_workers: amount of worker processes (data reading)
        :param dtype: type of Tensor to use
        :return: None
        """
        with h5py.File(file_path, r'r') as data_file:
            size = data_file[r'metadata'][r'n_samples'][()]
            alphabet_size = len(data_file[r'metadata'][r'aas'][()])

        # Compute statistics of data set.
        abundance = [{} for _ in range(size)]
        abundance_max = [{} for _ in range(len(abundance))]
        abundance_total = np.zeros((len(abundance),), dtype=np.long)
        kmer_counts = np.zeros((len(abundance),), dtype=np.long)
        progress_bar_1 = tqdm(desc=r'[1/3] Compute sample statistics', unit=r'sa', total=size, file=sys.stdout)
        progress_bar_2 = tqdm(desc=r'[2/3] Compute relative abundance', unit=r'sa', total=size, file=sys.stdout)
        progress_bar_3 = tqdm(desc=r'[3/3] Compute kmer-sequences', unit=r'sa', total=size, file=sys.stdout)
        progress_lock = multiprocessing.Lock()

        def _sample_callback(sample_result: Tuple[int, Dict[int, int], Dict[int, int], int, int]) -> None:
            """
            Append result of sample analysis to overall statistics collections.

            :param sample_result: result of sample analysis
            :return: None
            """
            _sample_index, _abundance, _abundance_max, _abundance_total, _kmer_count = sample_result
            abundance[_sample_index] = _abundance
            abundance_max[_sample_index] = _abundance_max
            abundance_total[_sample_index] = _abundance_total
            kmer_counts[_sample_index] = _kmer_count
            with progress_lock:
                progress_bar_1.update(1)

        # Apply sample computations asynchronously.
        num_tasks = multiprocessing.cpu_count() if num_workers <= 0 else num_workers
        with multiprocessing.get_context(method=cls.spawn_method).Pool(
                processes=num_tasks, maxtasksperchild=cls.tasks_per_child) as sample_pool:

            # Compute sample statistics.
            sample_futures = []
            with h5py.File(file_path, r'r') as data_file:
                if type(data_file['metadata'][r'labels']) == h5py.Group:
                    labels = np.where(data_file['metadata'][r'labels'][r'Known CMV status'])[1]
                else:
                    labels = np.where(data_file['metadata'][r'labels'])[1]
                for sample_index in range(size):
                    start, end = data_file[r'sampledata'][r'sample_sequences_start_end'][sample_index]
                    sequence_lengths = data_file[r'sampledata'][r'seq_lens'][start:end]
                    sequence_counts = data_file[r'sampledata'][r'duplicates_per_sequence'][start:end]
                    sequences = data_file[r'sampledata'][r'amino_acid_sequences'][start:end]
                    sample_futures.append(sample_pool.apply_async(
                        cls.sample_worker, (sample_index, sequence_lengths, sequence_counts, sequences, kmer_size),
                        callback=_sample_callback,
                        error_callback=lambda _: print(f'Failed to process sample {sample_index + 1}!\n{_}\n')))

            # Wait for remaining tasks to be finished (remove futures to free any remaining allocated resources).
            while True:
                finished_futures = sorted([index for index, _ in enumerate(sample_futures) if _.ready()], reverse=True)
                for finished_future in finished_futures:
                    del sample_futures[finished_future]
                if len(sample_futures) <= 0:
                    break
                time.sleep(10)

        # Free unnecessary memory.
        del sample_futures

        # Compute relative abundance term.
        keys = [r'_'.join(map(str, _)) for _ in product(range(alphabet_size), repeat=kmer_size)]
        relative_abundance_kmer_samples = [dict(zip_longest(keys, (), fillvalue=dtype(0))) for _ in range(size)]
        relative_abundance_tcrb_samples = [dict(zip_longest(keys, (), fillvalue=dtype(0))) for _ in range(size)]
        relative_abundance_kmer_mean = np.zeros((size,), dtype=dtype)
        relative_abundance_kmer_stdv = np.zeros((size,), dtype=dtype)
        relative_abundance_tcrb_mean = np.zeros((size,), dtype=dtype)
        relative_abundance_tcrb_stdv = np.zeros((size,), dtype=dtype)

        for sample_index in range(size):
            current_relative_abundance_kmer = np.zeros(len(abundance[sample_index]), dtype=dtype)
            current_relative_abundance_tcrb = np.zeros(len(abundance[sample_index]), dtype=dtype)

            for key_index, current_key in enumerate(abundance[sample_index].keys()):
                if abundance_total[sample_index] != 0:
                    relative_abundance_kmer_samples[sample_index][current_key] = np.log(
                        abundance[sample_index][current_key] / abundance_total[sample_index],
                        dtype=np.float64).astype(dtype)
                    relative_abundance_tcrb_samples[sample_index][current_key] = np.log(
                        abundance_max[sample_index][current_key] / abundance_total[sample_index],
                        dtype=np.float64).astype(dtype)
                current_relative_abundance_kmer[key_index] = relative_abundance_kmer_samples[sample_index][current_key]
                current_relative_abundance_tcrb[key_index] = relative_abundance_tcrb_samples[sample_index][current_key]

            # Compute statistics for abundance normalisation.
            relative_abundance_kmer_mean[sample_index] = current_relative_abundance_kmer.mean(
                dtype=np.float64).astype(dtype=dtype)
            relative_abundance_kmer_stdv[sample_index] = current_relative_abundance_kmer.std(
                ddof=1, dtype=np.float64).astype(dtype=dtype)
            relative_abundance_tcrb_mean[sample_index] = current_relative_abundance_tcrb.mean(
                dtype=np.float64).astype(dtype=dtype)
            relative_abundance_tcrb_stdv[sample_index] = current_relative_abundance_tcrb.std(
                ddof=1, dtype=np.float64).astype(dtype=dtype)
            progress_bar_2.update(1)

        # Free unnecessary memory.
        del abundance
        del abundance_max
        del abundance_total

        # Adapt data set (add relative abundance terms) and free unnecessary memory.
        with h5py.File(store_path, r'w') as data_file:

            # Write metadata (amount of samples as well as the target labels).
            data_file.require_dataset(
                r'metadata/n_samples', shape=(1,), data=size, dtype=np.long)
            data_file.require_dataset(
                r'metadata/labels', shape=labels.shape, data=labels, compression=cls.compression_algorithm,
                shuffle=cls.compression_shuffle, compression_opts=cls.compression_level, dtype=labels.dtype)
            data_file.flush()
            del labels

            # Write relative abundance term following the <KMER> computation.
            data_file.require_dataset(
                r'sampledata/relative_abundance_kmer', shape=(size, len(keys)), data=np.stack(
                    np.stack([list(_.values()) for _ in relative_abundance_kmer_samples])),
                compression=cls.compression_algorithm,
                shuffle=cls.compression_shuffle, compression_opts=cls.compression_level, dtype=dtype)
            data_file.flush()
            del relative_abundance_kmer_samples

            # Write relative abundance statistics (mean) following the <KMER> computation.
            data_file.require_dataset(
                r'sampledata/relative_abundance_kmer_mean', shape=relative_abundance_kmer_mean.shape,
                data=relative_abundance_kmer_mean,
                compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=dtype)
            data_file.flush()
            del relative_abundance_kmer_mean

            # Write relative abundance statistics (standard deviation) following the <KMER> computation.
            data_file.require_dataset(
                r'sampledata/relative_abundance_kmer_stdv', shape=relative_abundance_kmer_stdv.shape,
                data=relative_abundance_kmer_stdv,
                compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=dtype)
            data_file.flush()
            del relative_abundance_kmer_stdv

            # Write relative abundance term following the <TCRB> computation.
            data_file.require_dataset(
                r'sampledata/relative_abundance_tcrb', shape=(size, len(keys)), data=np.stack(
                    np.stack([list(_.values()) for _ in relative_abundance_tcrb_samples])),
                compression=cls.compression_algorithm,
                shuffle=cls.compression_shuffle, compression_opts=cls.compression_level, dtype=dtype)
            data_file.flush()
            del relative_abundance_tcrb_samples

            # Write relative abundance statistics (mean) following the <TCRB> computation.
            data_file.require_dataset(
                r'sampledata/relative_abundance_tcrb_mean', shape=relative_abundance_tcrb_mean.shape,
                data=relative_abundance_tcrb_mean,
                compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=dtype)
            data_file.flush()
            del relative_abundance_tcrb_mean

            # Write relative abundance statistics (standard deviation) following the <TCRB> computation.
            data_file.require_dataset(
                r'sampledata/relative_abundance_tcrb_stdv', shape=relative_abundance_tcrb_stdv.shape,
                data=relative_abundance_tcrb_stdv,
                compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=dtype)
            data_file.flush()
            del relative_abundance_tcrb_stdv
            del keys

        kmer_sequences_start_end = np.cumsum(np.asarray([0] + [_ for _ in kmer_counts]), dtype=np.long)
        kmer_counts_total = np.sum(kmer_counts, dtype=np.long)
        kmer_indices = np.zeros((kmer_counts_total,), dtype=np.long)
        kmer_sequences = np.zeros((kmer_counts_total, Atchley().depth * kmer_size), dtype=dtype)
        sample_mean = np.zeros((size, kmer_sequences.shape[1]), dtype=dtype)
        sample_stdv = np.zeros((size, kmer_sequences.shape[1]), dtype=dtype)

        def _kmer_callback(kmer_result: Tuple[int, np.ndarray, np.ndarray]) -> None:
            """
            Append result of kmer-extraction to overall statistics collection.

            :param kmer_result: result of kmer-extraction
            :return: None
            """
            current_sample_index, sample_kmers, sample_kmer_indices = kmer_result
            current_start, current_end = kmer_sequences_start_end[current_sample_index], kmer_sequences_start_end[
                current_sample_index + 1]
            kmer_indices[current_start:current_end] = sample_kmer_indices
            kmer_sequences[current_start:current_end, :] = sample_kmers
            sample_mean[current_sample_index] = np.mean(sample_kmers, axis=0, dtype=np.float64).astype(dtype=dtype)
            sample_stdv[current_sample_index] = np.std(sample_kmers, ddof=1, dtype=np.float64, axis=0).astype(
                dtype=dtype)
            with progress_lock:
                progress_bar_3.update(1)

        # Apply kmer computations asynchronously.
        with multiprocessing.get_context(method=cls.spawn_method).Pool(
                processes=num_tasks, maxtasksperchild=cls.tasks_per_child) as sample_pool:

            # Extract kmer-sequences.
            sample_futures = []
            with h5py.File(file_path, r'r') as data_file:
                for sample_index in range(size):
                    start, end = data_file[r'sampledata'][r'sample_sequences_start_end'][sample_index]
                    sequence_lengths = data_file[r'sampledata'][r'seq_lens'][start:end]
                    sequences = data_file[r'sampledata'][r'amino_acid_sequences'][start:end]
                    kmer_count = kmer_counts[sample_index]
                    sample_futures.append(sample_pool.apply_async(
                        cls.kmer_worker,
                        (sample_index, sequence_lengths, sequences, kmer_count, kmer_size, alphabet_size, dtype),
                        callback=_kmer_callback,
                        error_callback=lambda _: print(f'Failed to process sample {sample_index + 1}!\n{_}\n')))

            # Wait for remaining tasks to be finished (remove futures to free any remaining allocated resources).
            while True:
                finished_futures = sorted([index for index, _ in enumerate(sample_futures) if _.ready()], reverse=True)
                for finished_future in finished_futures:
                    del sample_futures[finished_future]
                if len(sample_futures) <= 0:
                    break
                time.sleep(10)

        # Free unnecessary memory.
        del sample_futures
        del kmer_counts

        # Adapt data set (add extracted kmer-sequences) and free unnecessary memory.
        with h5py.File(store_path, r'r+') as data_file:

            # Write start and end indices of extracted kmer sequences.
            data_file.require_dataset(
                r'sampledata/kmer_sequences_start_end', shape=kmer_sequences_start_end.shape,
                data=kmer_sequences_start_end, compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=kmer_sequences_start_end.dtype)
            data_file.flush()
            del kmer_sequences_start_end

            # Write kmer sequences.
            data_file.require_dataset(
                r'sampledata/kmer_sequences', shape=kmer_sequences.shape,
                data=kmer_sequences, compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=kmer_sequences.dtype)
            data_file.flush()
            del kmer_sequences

            # Write kmer indices in order to fetch the corresponding abundance term.
            data_file.require_dataset(
                r'sampledata/kmer_indices', shape=kmer_indices.shape,
                data=kmer_indices, compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=kmer_indices.dtype)

            # Write sample statistics (mean) for feature normalisation.
            data_file.require_dataset(
                r'sampledata/sample_mean', shape=sample_mean.shape,
                data=sample_mean, compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=sample_mean.dtype)
            data_file.flush()
            del sample_mean

            # Write sample statistics (standard deviation) for feature normalisation.
            data_file.require_dataset(
                r'sampledata/sample_stdv', shape=sample_stdv.shape,
                data=sample_stdv, compression=cls.compression_algorithm, shuffle=cls.compression_shuffle,
                compression_opts=cls.compression_level, dtype=sample_stdv.dtype)
            data_file.flush()
            del sample_stdv

        # Close progress bars.
        progress_bar_3.close()
        progress_bar_2.close()
        progress_bar_1.close()

    @staticmethod
    def kmer_sequence_unique_worker(kmer_sequences: np.ndarray, sample_index: int) -> Tuple[np.ndarray, int]:
        return np.unique(kmer_sequences, axis=0), sample_index

    @staticmethod
    def sample_worker(sample_index: int, sequence_lengths: np.ndarray, sequence_counts: np.ndarray,
                      sequences: np.ndarray,
                      kmer_size: int) -> Tuple[int, Dict[str, int], Dict[str, int], int, int]:
        """
        Analyse sample with respect to the relative abundance term.

        :param sample_index: index of specific sample to analyse
        :param sequence_lengths: lengths of specified sequences to analyse
        :param sequence_counts: counts of specified sequences to analyse
        :param sequences: sequences to be analysed
        :param kmer_size: size of a k-mer to extract
        :return: sample index as well as relative abundance statistics
        """
        abundance = {}
        abundance_max = {}
        abundance_total = 0
        kmer_count = 0

        # Compute relative abundance term for each <kmer> in the current sample.
        for sequence_index, (sequence_length, sequence_count, sequence) in enumerate(
                zip(sequence_lengths, sequence_counts, sequences)):

            # Due to the sequence pre-processing of Ostmeyer et al., the sequences need to have a specific min. length.
            if sequence_length < (6 + kmer_size):
                continue

            trimmed_sequence = sequence[3:sequence_length - 3]
            for kmer_index in range(trimmed_sequence.shape[0] - kmer_size + 1):
                current_kmer = trimmed_sequence[kmer_index:kmer_index + kmer_size]
                current_key = r'_'.join(current_kmer.astype(str).tolist())
                abundance[current_key] = abundance.setdefault(current_key, 0) + sequence_count.item()
                abundance_max[current_key] = max(abundance_max.setdefault(current_key, 0), sequence_count.item())
                abundance_total += sequence_count.item()
                kmer_count += 1

        return sample_index, abundance, abundance_max, abundance_total, kmer_count

    @staticmethod
    def kmer_worker(sample_index: int, sequence_lengths: np.ndarray, sequences: np.ndarray, kmer_count: int,
                    kmer_size: int,
                    alphabet_size: int, dtype: Type = np.float32) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Extract kmer-sequences from specified sample.

        :param sample_index: index of specific sample to analyse
        :param sequence_lengths: lengths of specified sequences to analyse
        :param sequences: sequences to be analysed
        :param kmer_size: size of a k-mer to extract
        :param kmer_count: amount of kmers of specified sample
        :param alphabet_size: number of different elements of the alphabet (amino acids)
        :param dtype: type of resulting kmer Tensor to use
        :return: sample index and statistics as well as extracted kmer-sequences
        """
        atchley = Atchley()
        position = 0
        sample_kmer = np.zeros((kmer_count, atchley.depth * kmer_size), dtype=dtype)
        sample_kmer_index = np.zeros((kmer_count,), dtype=np.long)

        # Compute relative abundance term for each <kmer> in the current sample.
        keys = [r'_'.join(map(str, _)) for _ in product(range(alphabet_size), repeat=kmer_size)]
        keys = dict(zip(keys, range(len(keys))))
        for sequence_index, (sequence_length, sequence) in enumerate(zip(sequence_lengths, sequences)):

            # Due to the sequence pre-processing of Ostmeyer et al., the sequences need to have a specific min. length.
            if sequence_length < (6 + kmer_size):
                continue

            trimmed_sequence = sequence[3:sequence_length - 3]
            for kmer_index in range(trimmed_sequence.shape[0] - kmer_size + 1):
                current_kmer = trimmed_sequence[kmer_index:kmer_index + kmer_size]
                sample_kmer[position] = np.stack(
                    [atchley[atchley.keys[_]].numpy().astype(sample_kmer.dtype) for _ in current_kmer],
                    axis=0).reshape(1, -1)
                sample_kmer_index[position] = keys[r'_'.join(current_kmer.astype(str).tolist())]
                position += 1

        return sample_index, sample_kmer, sample_kmer_index

    @staticmethod
    def init_child(sequences_buffer: np.ndarray, sequences_shape: Tuple[int, int]) -> None:
        """
        Initialise variables of pre-processing worker processes from global memory.

        :param sequences_buffer: presence buffer of kmer-sequences in sample
        :param sequences_shape: shape of <sequences_buffer>
        :return: None
        """
        global child_vars

        child_vars[r'sequences_buffer'] = sequences_buffer
        child_vars[r'sequences_shape'] = sequences_shape

    @property
    def sample_means(self) -> np.ndarray:
        return self.__sample_mean

    @property
    def sample_standard_deviations(self) -> np.ndarray:
        return self.__sample_stdv

    @property
    def sample_sizes(self) -> np.ndarray:
        return self.__sample_sizes


class LogisticMIL(object):
    """
    Supervisory instance for operating logistic MIL (as defined in class <LogisticMILModule>).
    """

    def __init__(self, file_path: Path, relative_abundance: LogisticMILDataReader.RelativeAbundance,
                 fold_info: Union[None, int, Path] = 5, num_workers: int = 2, device: str = r'cpu',
                 dtype: torch.dtype = torch.float32, test_mode: bool = False, offset: int = 0):
        """
        Initialise supervisory instance for operating logistic MIL (as defined in class <LogisticMILModule>).

        :param file_path: data file to read from (h5py)
        :param relative_abundance: type of relative abundance term as defined in Ostmeyer et al., equations (A) and (B)
        :param fold_info: number of folds for cross-validation (or <None> to use the whole data set)
        :param num_workers: number of workers deployed for reading/loading data sets
        :param device: device to use for heavy computations
        :param dtype: type of Tensor to use
        :param test_mode: flag if LogisticMIL should be loaded in test mode
        :param offset: offset used to define evaluation, test, and training folds
        """
        assert Path.exists(file_path) and h5py.is_hdf5(file_path), r'Invalid data file specified!'
        assert (fold_info is None) or ((type(fold_info) == int) and (fold_info > 1)) or Path.is_file(fold_info)

        self.__relative_abundance = relative_abundance
        self.__fold_info = fold_info
        self.__num_folds = None
        self.__num_workers = num_workers if num_workers >= 0 else multiprocessing.cpu_count()
        self.__device = torch.device(device)
        self.__dtype = dtype

        with h5py.File(file_path, r'r') as data_file:
            self.__kmer_size = data_file[r'sampledata'][r'kmer_sequences'].shape[1] // Atchley().depth

        self.__data_reader = []
        self.__test_reader = None
        self.__indices_test_resort = None
        if fold_info is None:
            self.__data_reader.append(LogisticMILDataReader(
                file_path=file_path, relative_abundance=relative_abundance,
                indices=None, unique=False, dtype=self.__dtype))
        elif type(fold_info) == int:
            self.__num_folds = fold_info
            with h5py.File(file_path, r'r') as data_file:
                num_repertoires = data_file[r'metadata'][r'n_samples'][()]
                assert num_repertoires >= fold_info
                indices = np.arange(num_repertoires, dtype=np.long)
                np.random.shuffle(indices)
                indices = np.array_split(indices, indices_or_sections=fold_info, axis=0)
                for fold_indices in indices:
                    self.__data_reader.append(LogisticMILDataReader(
                        file_path=file_path, relative_abundance=relative_abundance,
                        indices=list(np.sort(fold_indices, axis=0)), unique=True, dtype=self.__dtype))
        else:
            with open(self.__fold_info, r'br') as pickle_file:
                self.__fold_info = pickle.load(pickle_file)
                indices_folds = self.__fold_info[r'inds']
                eval_offset = min(offset, len(indices_folds) - 1)
                test_offset = 0 if ((eval_offset + 1) >= len(indices_folds)) else (eval_offset + 1)
                indices_train = np.concatenate([
                    fold for _, fold in enumerate(indices_folds) if all([_ != eval_offset, _ != test_offset])], axis=0)
                indices_eval = indices_folds[eval_offset]
                indices_test = indices_folds[test_offset]
            self.__num_folds = 1

            if test_mode:
                self.__test_reader = LogisticMILDataReader(
                    file_path=file_path, relative_abundance=relative_abundance,
                    indices=list(np.sort(indices_test, axis=0)), unique=False, dtype=self.__dtype)
                self.__indices_test_resort = np.argsort(indices_test, axis=0)
            else:
                self.__data_reader.append(LogisticMILDataReader(
                    file_path=file_path, relative_abundance=relative_abundance,
                    indices=list(np.sort(indices_eval, axis=0)), unique=True, dtype=self.__dtype))
                self.__data_reader.append(LogisticMILDataReader(
                    file_path=file_path, relative_abundance=relative_abundance,
                    indices=list(np.sort(indices_train, axis=0)), unique=True, dtype=self.__dtype))

    @staticmethod
    def _compute_pooled_variance(variances: np.ndarray, means: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute pooled variance (unbiased).

        :param variances: variances to pool
        :param means:
        :param weights: sample counts to weigh the corresponding variance
        :return: pooled variance (unbiased)
        """
        assert len(variances) == len(weights)
        pooled_mean = np.average(means, axis=0, weights=weights)
        numerator = ((weights - 1).reshape((-1, 1)) * variances + weights.reshape((-1, 1)) * np.square(
            means - pooled_mean)).sum(axis=0)
        return numerator / (weights.sum() - 1)

    @staticmethod
    def _inference_step(x: torch.Tensor, indices: torch.Tensor, logistic_mil_module: LogisticMILModule,
                        top_n: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one inference step and return index of the maximum activation as well as the activation itself.

        :param x: data on which the logistic MIL module is applied
        :param indices: border indices of the repertoires
        :param logistic_mil_module: the logistic MIL module to apply
        :param top_n: consider the top <n> highest scoring activations
        :return: indices of the maximum activations, the activations themselves as well as the amount of active <n>
        """
        logistic_mil_module.eval()
        with torch.no_grad():
            repertoire_prediction = logistic_mil_module.forward(atchley_factors_abundance=x)
            maximum_predictions, maximum_indices, active_n = [], [], []
            for start_index, end_index in zip(indices[:-1], indices[1:]):
                current_top_n = min(top_n, end_index - start_index)
                current_top_k = repertoire_prediction[start_index:end_index].topk(
                    k=current_top_n, dim=0, largest=True, sorted=False)
                maximum_predictions.extend(current_top_k[0])
                maximum_indices.extend(start_index + current_top_k[1])
                active_n.append(current_top_n)
        active_n = torch.tensor(active_n, device=repertoire_prediction.device)
        return torch.cat(maximum_indices, dim=0), torch.cat(maximum_predictions, dim=0), active_n

    @staticmethod
    def reshape_collate(batch_data: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Custom collation for a reshaped stacking of tensors with variable sizes.

        :param batch_data: variable sized tensors to reshape and stack (<data> and <target>)
        :return: reshaped and stacked tensors (<data>, <target> and <indices>)
        """
        num_samples = len(batch_data)
        kmer_counts = [_[0].shape[0] for _ in batch_data]
        num_kmers = sum(kmer_counts)
        result = (torch.zeros((num_kmers, batch_data[0][0].shape[1]), dtype=batch_data[0][0].dtype),
                  torch.zeros((num_samples,), dtype=batch_data[0][1].dtype),
                  torch.as_tensor([0] + kmer_counts, dtype=torch.long).cumsum(dim=0))
        for sample_index, (sample_data, sample_target) in enumerate(batch_data):
            start_index = result[2][sample_index].item()
            result[0][start_index:start_index + sample_data.shape[0]] = sample_data
            result[1][sample_index] = sample_target
        return result

    @classmethod
    def _compute_normalisation_statistics(
            cls, data_readers: List[LogisticMILDataReader]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute statistics of features needed for normalisation.

        :param data_readers: sources to use for statistic computations
        :return: feature mean and standard deviation
        """
        sample_weights = np.concatenate([_.sample_sizes for _ in data_readers], axis=0)
        sample_means = np.concatenate([_.sample_means for _ in data_readers], axis=0)
        sample_variances = np.concatenate([np.square(_.sample_standard_deviations) for _ in data_readers], axis=0)
        pooled_stdv = np.sqrt(
            cls._compute_pooled_variance(variances=sample_variances, means=sample_means, weights=sample_weights))
        return np.average(sample_means, axis=0, weights=sample_weights), pooled_stdv

    def _train(self, data_reader: data.Dataset, batch_size: int, top_n: int, randomise: bool,
               logistic_mil_module: LogisticMILModule, loss_function: nn.BCEWithLogitsLoss, optimiser: optim.Adam,
               normalise_mean: torch.Tensor = None, normalise_stdv: torch.Tensor = None,
               repetitions: int = 0) -> None:
        """
        Train specified logistic MIL module for one epoch.

        :param data_reader: data reader responsible for yielding repertoire data
        :param batch_size: number of repertoires to process before a module update
        :param top_n: consider the top <n> highest scoring activations
        :param randomise: shuffle order of batches
        :param logistic_mil_module: logistic MIL module to train
        :param loss_function: binary cross-entropy function requiring pre-sigmoid activations
        :param optimiser: stochastic gradient descent optimiser to apply
        :param normalise_mean: feature mean to use for normalisation
        :param normalise_stdv: feature standard deviation to use for normalisation
        :param repetitions: number of training repetitions ("inner" epochs)
        :return: None
        """
        optimiser.zero_grad()
        data_loader = data.DataLoader(
            dataset=data_reader, batch_size=batch_size, shuffle=randomise, num_workers=self.__num_workers,
            collate_fn=self.reshape_collate)

        for _ in range(repetitions + 1):
            for batch_index, (repertoire_data, repertoire_target, repertoire_index) in tqdm(
                    enumerate(data_loader, start=1), desc=r'Train', unit=r'sa', total=len(data_loader),
                    leave=False, file=sys.stdout):

                repertoire_data = repertoire_data.to(device=self.__device)
                repertoire_target = repertoire_target.to(device=self.__device)
                if (normalise_mean is not None) and (normalise_stdv is not None):
                    repertoire_data -= normalise_mean
                    repertoire_data /= normalise_stdv

                # Get index of maximum pre-activation (in eval mode, to save memory).
                repertoire_predictions, _, active_n = self._inference_step(
                    x=repertoire_data, indices=repertoire_index, logistic_mil_module=logistic_mil_module, top_n=top_n)
                selected_exemplar = repertoire_data[repertoire_predictions].detach()

                # Compute module gradients with respect to instance with maximum activation.
                logistic_mil_module.train()
                repertoire_prediction = logistic_mil_module.forward(
                    atchley_factors_abundance=selected_exemplar).reshape(-1)

                repertoire_loss = loss_function.forward(
                    input=repertoire_prediction, target=repertoire_target.repeat_interleave(repeats=active_n))
                repertoire_loss.backward()

                # Update module parameters.
                optimiser.step()
                optimiser.zero_grad()

    def _evaluate(self, logistic_mil_module: LogisticMILModule,
                  data_reader: Union[data.ConcatDataset, LogisticMILDataReader], batch_size: int,
                  normalise_mean: torch.Tensor = None, normalise_stdv: torch.Tensor = None,
                  top_n: int = 1, average_loss: bool = True,
                  collect_predictions: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                              Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate logistic MIL module.

        :param logistic_mil_module: pre-trained <LogisticModule> instance
        :param data_reader: data reader responsible for yielding repertoire data
        :param batch_size: number of repertoires to process at once
        :param normalise_mean: feature mean to use for normalisation
        :param normalise_stdv: feature standard deviation to use for normalisation
        :param top_n: consider the top <n> highest scoring activations
        :param average_loss: compute average loss with respect to samples
        :param collect_predictions: collect predictions additionally to confusion matrix and loss
        :return: confusion matrix indicating current model performance, as well as corresponding loss
        """
        data_loader = data.DataLoader(
            dataset=data_reader, batch_size=batch_size, shuffle=False, num_workers=self.__num_workers,
            collate_fn=self.reshape_collate)
        loss_function = nn.BCEWithLogitsLoss(reduction=r'mean' if average_loss else r'sum').to(device=self.__device)
        loss_function.eval()

        loss = torch.zeros((), dtype=self.__dtype, device=self.__device)
        confusion_matrix = torch.zeros((2, 2), dtype=self.__dtype).numpy()
        prediction = []
        target = []

        with torch.no_grad():
            for repertoire_data, repertoire_target, repertoire_index in tqdm(
                    data_loader, desc=r'Evaluate', unit=r'sa', total=len(data_loader),
                    leave=False, file=sys.stdout):

                repertoire_data = repertoire_data.to(device=self.__device)
                repertoire_target = repertoire_target.to(device=self.__device)
                if (normalise_mean is not None) and (normalise_stdv is not None):
                    repertoire_data -= normalise_mean
                    repertoire_data /= normalise_stdv

                _, predictions, active_n = self._inference_step(
                    x=repertoire_data, indices=repertoire_index, logistic_mil_module=logistic_mil_module, top_n=top_n)
                predictions = predictions.reshape(-1)
                repertoire_target = repertoire_target.repeat_interleave(repeats=active_n)
                loss += loss_function.forward(input=predictions, target=repertoire_target)
                predictions = torch.sigmoid(predictions)
                if collect_predictions:
                    prediction.append(predictions)
                    target.append(repertoire_target)
                confusion_matrix += metrics.confusion_matrix(
                    y_true=repertoire_target.cpu().long().numpy(),
                    y_pred=predictions.round().long().cpu().numpy(), labels=[0, 1])

            # Optionally average loss and concatenate predictions.
            loss = (loss.cpu() / len(data_loader)) if average_loss else loss.cpu()
            if collect_predictions:
                prediction = torch.cat(tensors=prediction)
                target = torch.cat(tensors=target)

        return torch.from_numpy(confusion_matrix), loss, (prediction, target)

    def optimise(self, epochs: int, batch_sizes: Sequence[int], learning_rates: Sequence[float],
                 betas_one: Sequence[float], betas_two: Sequence[float], weight_decays: Sequence[float],
                 amsgrad: bool, epsilon: float, top_n: Sequence[int], normalise: bool,
                 normalise_abundance: bool, average_loss: bool = True, randomise: bool = True, repetitions: int = 0,
                 seed: int = 42, log_dir: Path = None) -> Dict[str, Union[int, float, bool]]:
        """
        Optimise hyperparameters of logistic MIL module according to balanced accuracy.

        :param epochs: number of epochs to train
        :param batch_sizes: range of number of repertoires to process before a module update
        :param learning_rates: range of step size exponent of Adam optimiser
        :param betas_one: range of beta 1 (Adam)
        :param betas_two: range of beta 2 (Adam)
        :param weight_decays: range of weight decay exponent of Adam optimiser
        :param amsgrad: use AMSGrad version of Adam
        :param epsilon: epsilon to use for numerical stability
        :param top_n: range of <n>, whereas <n> denotes the top <n> samples to consider per repertoire
        :param normalise: rescale features to have zero mean and unit variance
        :param normalise_abundance: rescale relative abundance term to have zero mean and unit variance
        :param average_loss: compute average loss with respect to samples
        :param randomise: shuffle order of batches
        :param repetitions: number of training repetitions ("inner" epochs)
        :param seed: seed to be used for reproducibility
        :param log_dir: directory to store TensorBoard logs
        :return: best hyperparameters found by cross-validation
        """

        # Create grid of hyperparameter settings for hyperparameter optimisation.
        hyperparameter_settings = list(product(batch_sizes, learning_rates, betas_one, betas_two, weight_decays, top_n))

        log_writers = []
        if log_dir is not None:
            log_writers = [SummaryWriter(log_dir=str(
                (log_dir / f'trial_{_ + 1}'))) for _ in range(len(hyperparameter_settings))]

        best_hyperparameters = {r'epochs': int(), r'batch_size': int(), r'top_n': int(), r'learning_rate': float(),
                                r'beta_one': float(), r'beta_two': float(), r'weight_decay': float(),
                                r'amsgrad': amsgrad, r'epsilon': epsilon, r'normalise_mean': [], r'normalise_stdv': []}
        best_performance = -np.inf
        normalise_mean = []
        normalise_stdv = []
        progress_bar_1 = tqdm(
            total=len(hyperparameter_settings), desc=r'Trial', unit=r'tr', position=0, file=sys.stdout)
        progress_bar_2 = tqdm(
            total=epochs * (repetitions + 1), desc=r'Epoch', unit=r'ep', position=1, file=sys.stdout)
        progress_bar_3 = tqdm(
            total=self.__num_folds, desc=r'Fold', unit=r'fo', position=2, file=sys.stdout)

        # Perform grid search to optimise hyperparameters.
        for trial, current_setting in enumerate(hyperparameter_settings):
            torch.manual_seed(seed=seed)
            np.random.seed(seed)

            # Unpack current hyperparameter settings.
            batch_size, learning_rate, beta_one, beta_two, weight_decay, top_n_samples = current_setting

            # Create logistic MIL modules and accompanying instances.
            logistic_mil_modules = []
            loss_functions = []
            optimisers = []
            for fold in range(self.__num_folds):
                logistic_mil_modules.append(LogisticMILModule(kmer_size=self.__kmer_size).to(device=self.__device))
                loss_functions.append(nn.BCEWithLogitsLoss(
                    reduction=r'mean' if average_loss else r'sum').to(device=self.__device))
                optimisers.append(optim.Adam(params=logistic_mil_modules[-1].parameters(), lr=learning_rate,
                                             betas=(beta_one, beta_two), eps=epsilon,
                                             weight_decay=weight_decay, amsgrad=amsgrad))

                # Compute normalisation statistics.
                if normalise and (len(normalise_mean) < self.__num_folds):
                    sample_mean, sample_stdv = self._compute_normalisation_statistics(
                        [data_set for _, data_set in enumerate(self.__data_reader) if _ != fold])
                    sample_stdv = np.where(sample_stdv > 0, sample_stdv, 1)
                    normalise_mean.append(torch.from_numpy(sample_mean).to(dtype=self.__dtype, device=self.__device))
                    normalise_stdv.append(torch.from_numpy(sample_stdv).to(dtype=self.__dtype, device=self.__device))
                    if not normalise_abundance:
                        normalise_mean[-1][-1] = 0
                        normalise_stdv[-1][-1] = 1

            # Save current hyperparameters.
            if log_dir is not None:
                with open(str((log_dir / f'trial_{trial + 1}' / r'hyperparameters.json')),
                          r'w') as hyperparameters_json:
                    json.dump({r'epochs': epochs, r'batch_size': batch_size, r'top_n': top_n_samples,
                               r'learning_rate': learning_rate, r'beta_one': beta_one,
                               r'beta_two': beta_two, r'weight_decay': weight_decay,
                               r'amsgrad': amsgrad, r'epsilon': epsilon,
                               r'normalise_mean': [_.tolist() for _ in normalise_mean],
                               r'normalise_stdv': [_.tolist() for _ in normalise_stdv]}, hyperparameters_json)

            # Train logistic MIL modules.
            for epoch in range(1, epochs + 1):
                fold_losses = {r'train': torch.zeros((1,), dtype=self.__dtype),
                               r'eval': torch.zeros((1,), dtype=self.__dtype)}
                fold_confusion_matrices = {r'train': torch.zeros((2, 2), dtype=self.__dtype),
                                           r'eval': torch.zeros((2, 2), dtype=self.__dtype)}
                fold_roc_auc = {r'train': torch.zeros((1,), dtype=self.__dtype),
                                r'eval': torch.zeros((1,), dtype=self.__dtype)}
                fold_predictions = []
                for fold in range(self.__num_folds):
                    training_data_set = data.ConcatDataset(
                        [data_set for _, data_set in enumerate(self.__data_reader) if _ != fold])
                    self._train(data_reader=training_data_set,
                                batch_size=batch_size, top_n=top_n_samples, randomise=randomise,
                                logistic_mil_module=logistic_mil_modules[fold], loss_function=loss_functions[fold],
                                optimiser=optimisers[fold],
                                normalise_mean=normalise_mean[fold] if normalise else None,
                                normalise_stdv=normalise_stdv[fold] if normalise else None,
                                repetitions=repetitions)

                    # Evaluate current model on training set.
                    current_evaluation = self._evaluate(
                        logistic_mil_module=logistic_mil_modules[fold], data_reader=training_data_set,
                        normalise_mean=normalise_mean[fold] if normalise else None,
                        normalise_stdv=normalise_stdv[fold] if normalise else None,
                        top_n=top_n_samples, average_loss=True, collect_predictions=True, batch_size=batch_size)
                    fold_predictions.append(current_evaluation[2][0])
                    fold_losses[r'train'] += current_evaluation[1]
                    fold_confusion_matrices[r'train'] += current_evaluation[0]
                    fold_roc_auc[r'train'] += metrics.roc_auc_score(
                        y_true=current_evaluation[2][1].cpu().numpy(), y_score=current_evaluation[2][0].cpu().numpy())

                    # Evaluate current model on evaluation set.
                    current_evaluation = self._evaluate(
                        logistic_mil_module=logistic_mil_modules[fold], data_reader=self.__data_reader[fold],
                        normalise_mean=normalise_mean[fold] if normalise else None,
                        normalise_stdv=normalise_stdv[fold] if normalise else None,
                        top_n=top_n_samples, average_loss=True, collect_predictions=True, batch_size=batch_size)
                    fold_losses[r'eval'] += current_evaluation[1]
                    fold_confusion_matrices[r'eval'] += current_evaluation[0]
                    fold_roc_auc[r'eval'] += metrics.roc_auc_score(
                        y_true=current_evaluation[2][1].cpu().numpy(), y_score=current_evaluation[2][0].cpu().numpy())
                    progress_bar_3.update(1)

                # Compute performance metrics on training set.
                tn, fp, fn, tp = fold_confusion_matrices[r'train'].flatten()
                sensitivity = np.nan_to_num(tp / (tp + fn), nan=1.0)
                specificity = np.nan_to_num(tn / (tn + fp), nan=1.0)
                sensitivities = {r'train': sensitivity}
                specificities = {r'train': specificity}
                balanced_accuracies = {r'train': (sensitivity + specificity) / 2.0}
                fold_roc_auc[r'train'] /= self.__num_folds

                # Compute performance measures on evaluation set and compare with current best performing one.
                tn, fp, fn, tp = fold_confusion_matrices[r'eval'].flatten()
                sensitivity = np.nan_to_num(tp / (tp + fn), nan=1.0)
                specificity = np.nan_to_num(tn / (tn + fp), nan=1.0)
                sensitivities[r'eval'] = sensitivity
                specificities[r'eval'] = specificity
                balanced_accuracies[r'eval'] = (sensitivity + specificity) / 2.0
                fold_roc_auc[r'eval'] /= self.__num_folds
                if best_performance < fold_roc_auc[r'eval']:
                    best_performance = fold_roc_auc[r'eval']
                    best_hyperparameters[r'epochs'] = epoch
                    best_hyperparameters[r'batch_size'] = batch_size
                    best_hyperparameters[r'top_n'] = top_n_samples
                    best_hyperparameters[r'learning_rate'] = learning_rate
                    best_hyperparameters[r'beta_one'] = beta_one
                    best_hyperparameters[r'beta_two'] = beta_two
                    best_hyperparameters[r'weight_decay'] = weight_decay
                    if normalise:
                        best_hyperparameters[r'normalise_mean'] = [_.cpu().numpy().tolist() for _ in normalise_mean]
                        if normalise_abundance:
                            best_hyperparameters[r'normalise_stdv'] = [_.cpu().numpy().tolist() for _ in normalise_stdv]

                    # Save trained logistic MIL module.
                    if log_dir is not None and self.__num_folds == 1:
                        state_dict = logistic_mil_modules[0].state_dict()
                        state_dict[r'top_n'] = best_hyperparameters[r'top_n']
                        state_dict[r'abundance'] = str(self.__relative_abundance)
                        if normalise:
                            state_dict[r'normalise_mean'] = normalise_mean[0].cpu().numpy().tolist()
                            state_dict[r'normalise_stdv'] = normalise_stdv[0].cpu().numpy().tolist()
                        torch.save(state_dict, str(log_dir / f'final_model.pth'))

                if len(log_writers) > 0:
                    # Log performance metrics.
                    log_writers[trial].add_scalar(
                        tag=r'cross_entropy_loss/train', scalar_value=fold_losses[r'train'] / self.__num_folds,
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'cross_entropy_loss/eval', scalar_value=fold_losses[r'eval'] / self.__num_folds,
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'balanced_accuracy/train', scalar_value=balanced_accuracies[r'train'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'balanced_accuracy/eval', scalar_value=balanced_accuracies[r'eval'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'sensitivity/train', scalar_value=sensitivities[r'train'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'sensitivity/eval', scalar_value=sensitivities[r'eval'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'specificity/train', scalar_value=specificities[r'train'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'specificity/eval', scalar_value=specificities[r'eval'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'roc_auc/train', scalar_value=fold_roc_auc[r'train'],
                        global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_scalar(
                        tag=r'roc_auc/eval', scalar_value=fold_roc_auc[r'eval'],
                        global_step=epoch * (repetitions + 1))

                    # Log meta information.
                    if logistic_mil_modules[0].linear_mapping.bias is not None:
                        log_writers[trial].add_histogram(tag=r'biases', values=torch.cat([
                            logistic_mil_modules[_].linear_mapping.bias for _ in range(self.__num_folds)
                        ]), global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_histogram(
                        tag=r'weights/atchley_factors', values=torch.cat([
                            logistic_mil_modules[_].linear_mapping.weight[:, :-1] for _ in range(self.__num_folds)
                        ]), global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_histogram(
                        tag=r'weights/relative_abundance', values=torch.cat([
                            logistic_mil_modules[_].linear_mapping.weight[:, -1] for _ in range(self.__num_folds)
                        ]), global_step=epoch * (repetitions + 1))
                    log_writers[trial].add_histogram(
                        tag=r'predictions', values=torch.cat(fold_predictions),
                        global_step=epoch * (repetitions + 1))

                progress_bar_3.reset()
                progress_bar_2.update(repetitions + 1)
            progress_bar_2.reset()
            progress_bar_1.update(1)
        progress_bar_1.refresh()

        # Close summary writers.
        for log_writer in log_writers:
            log_writer.close()

        # Close progress bars.
        progress_bar_3.close()
        progress_bar_2.close()
        progress_bar_1.close()

        return best_hyperparameters

    def train(self, file_path_output: Path, epochs: int, batch_size: int, top_n: int, learning_rate: float,
              beta_one: float, beta_two: float, weight_decay: float, amsgrad: bool, epsilon: float, normalise: bool,
              normalise_abundance: bool, average_loss: bool = True, randomise: bool = True, seed: int = 42) -> None:
        """
        Train logistic MIL module and save the resulting state dictionary to disk.

        :param file_path_output: path to store resulting model
        :param epochs: number of epochs to train
        :param batch_size: number of repertoires to process before a module update
        :param top_n: consider the top <n> highest scoring activations
        :param learning_rate: step size of Adam optimiser
        :param beta_one: beta 1 of Adam optimiser
        :param beta_two: beta 2 of Adam optimiser
        :param weight_decay: weight decay term of Adam optimiser
        :param amsgrad: use AMSGrad version of Adam
        :param epsilon: epsilon to use for numerical stability
        :param normalise: rescale features to have zero mean and unit variance
        :param normalise_abundance: rescale relative abundance term to have zero mean and unit variance
        :param average_loss: compute average loss with respect to samples
        :param randomise: shuffle order of batches
        :param seed: seed to be used for reproducibility
        :return: None
        """
        torch.manual_seed(seed=seed)
        np.random.seed(seed)

        logistic_mil_module = LogisticMILModule(kmer_size=self.__kmer_size).to(device=self.__device)
        loss_function = nn.BCEWithLogitsLoss(reduction=r'mean' if average_loss else r'sum')
        optimiser = optim.Adam(params=logistic_mil_module.parameters(), lr=learning_rate, betas=(beta_one, beta_two),
                               eps=epsilon, weight_decay=weight_decay, amsgrad=amsgrad)

        # Compute normalisation statistics.
        normalise_mean = None
        normalise_stdv = None
        if normalise:
            sample_mean, sample_stdv = self._compute_normalisation_statistics(
                [data_set for data_set in self.__data_reader])
            sample_stdv = np.where(sample_stdv > 0, sample_stdv, 1)
            normalise_mean = torch.from_numpy(sample_mean).to(dtype=self.__dtype, device=self.__device)
            normalise_stdv = torch.from_numpy(sample_stdv).to(dtype=self.__dtype, device=self.__device)
            if not normalise_abundance:
                normalise_mean[-1] = 0
                normalise_stdv[-1] = 1

        # Train logistic MIL module.
        for epoch in range(epochs):
            self._train(data_reader=self.__data_reader[0], batch_size=batch_size, top_n=top_n,
                        logistic_mil_module=logistic_mil_module, randomise=randomise,
                        loss_function=loss_function, optimiser=optimiser,
                        normalise_mean=normalise_mean, normalise_stdv=normalise_stdv)

        # Save trained logistic MIL module.
        state_dict = logistic_mil_module.state_dict()
        state_dict[r'top_n'] = top_n
        state_dict[r'abundance'] = str(self.__relative_abundance)
        if normalise:
            state_dict[r'normalise_mean'] = normalise_mean.cpu().numpy().tolist()
            state_dict[r'normalise_stdv'] = normalise_stdv.cpu().numpy().tolist()
        torch.save(state_dict, str(file_path_output))

    def predict(self, logistic_mil_module: LogisticMILModule, data_reader: LogisticMILDataReader,
                normalise_mean: np.ndarray = None, normalise_stdv: np.ndarray = None,
                activations: bool = False) -> Tuple[List[Union[int, float]], Optional[float]]:
        """
        Predict per-repertoire label predictions/activations according to logistic MIL.

        :param logistic_mil_module: pre-trained <LogisticMILModule> instance
        :param data_reader: data reader responsible for yielding repertoire data
        :param normalise_mean: feature mean to use for normalisation
        :param normalise_stdv: feature standard deviation to use for normalisation
        :param activations: return activations instead of discrete predictions
        :return: per-repertoire label activations (and ROC AUC if predicted on test fold/set)
        """
        with torch.no_grad():

            result, roc_auc = None, None
            if self.__test_reader is None:

                # Apply logistic MIL model on specified data set.
                data_loader = data.DataLoader(
                    dataset=data_reader, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                    collate_fn=self.reshape_collate)

                result = [0.0] * len(data_reader)
                for batch_index, (repertoire_data, _, repertoire_index) in enumerate(data_loader):
                    repertoire_data = repertoire_data.reshape((-1, repertoire_data.shape[-1])).to(device=self.__device)
                    if (normalise_mean is not None) and (normalise_stdv is not None):
                        repertoire_data -= normalise_mean
                        repertoire_data /= normalise_stdv
                    result[batch_index] = torch.sigmoid(self._inference_step(
                        x=repertoire_data, indices=repertoire_index,
                        logistic_mil_module=logistic_mil_module, top_n=1)[1]).cpu().item()
                    if activations:
                        result[batch_index] = float(result[batch_index])
                    else:
                        result[batch_index] = int(round(result[batch_index]))

            else:

                # Evaluate logistic MIL model on test fold.
                current_evaluation = self._evaluate(
                    logistic_mil_module=logistic_mil_module, data_reader=data_reader, batch_size=1,
                    normalise_mean=normalise_mean, normalise_stdv=normalise_stdv,
                    top_n=1, average_loss=True, collect_predictions=True)

                result_resorted = current_evaluation[2][0].cpu().numpy()
                if not activations:
                    result_resorted = result_resorted.round().astype(dtype=np.int32)
                result = np.zeros_like(result_resorted)
                result[self.__indices_test_resort] = result_resorted
                result = list(result)
                roc_auc = metrics.roc_auc_score(
                    y_true=current_evaluation[2][1].cpu().numpy(), y_score=current_evaluation[2][0].cpu().numpy())

        return result, roc_auc

    def predict_from_path(self, file_path_model: Path,
                          activations: bool = False) -> Tuple[List[Union[int, float]], Optional[float]]:
        """
        Predict per-repertoire label activations according to logistic MIL.

        :param file_path_model: path to pre-trained <LogisticMILModule> instance
        :param activations: return activations instead of discrete predictions
        :return: per-repertoire label activations
        """
        logistic_mil_module = LogisticMILModule(kmer_size=self.__kmer_size).to(device=self.__device)
        state_dict = torch.load(str(file_path_model))
        normalise_mean = torch.from_numpy(np.asarray(state_dict[r'normalise_mean'])).to(
            device=self.__device) if r'normalise_mean' in state_dict else None
        normalise_stdv = torch.from_numpy(np.asarray(state_dict[r'normalise_stdv'])).to(
            device=self.__device) if r'normalise_stdv' in state_dict else None

        # Clean logistic MIL model state dict.
        del state_dict[r'top_n']
        del state_dict[r'abundance']
        if r'normalise_mean' in state_dict:
            del state_dict[r'normalise_mean']
        if r'normalise_stdv' in state_dict:
            del state_dict[r'normalise_stdv']

        logistic_mil_module.load_state_dict(state_dict)
        return self.predict(logistic_mil_module=logistic_mil_module,
                            data_reader=self.__test_reader if self.__test_reader is not None else self.__data_reader[0],
                            normalise_mean=normalise_mean, normalise_stdv=normalise_stdv, activations=activations)
