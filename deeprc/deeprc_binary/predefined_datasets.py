# -*- coding: utf-8 -*-
"""Pre-defined data loaders for the 4 dataset categories in paper
"""
import os
import dill as pkl
import requests
import shutil
import tqdm
import deeprc
from deeprc.deeprc_binary.dataset_readers import make_dataloaders

        
def _url_get_(url: str, dst: str, verbose: bool = True):
    """Download url to dst file"""
    stream = requests.get(url, stream=True)
    stream_size = int(stream.headers['Content-Length'])
    src = stream.raw
    windows = os.name == 'nt'
    copy_bufsize = 1024 * 1024 if windows else 64 * 1024
    update_progess_bar = tqdm.tqdm(total=stream_size, disable=not verbose,
                                   desc=f"Downloading {stream_size * 1e-9:0.3f}GB dataset")
    with open(dst, 'wb') as out_file:
        while True:
            buf = src.read(copy_bufsize)
            if not buf:
                break
            update_progess_bar.update(copy_bufsize)
            out_file.write(buf)
        shutil.copyfileobj(stream.raw, out_file)
    print()
    del stream


def simulated_dataset(dataset_path: str = None, dataset_id: int = 0,
                           cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                           inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                           sample_n_sequences: int = int(1e4), verbose: bool = True):
    """Get data loaders for category "simulated immunosequencing data".
     
     Warning: Filesize to download is ~20GB per dataset!
     Get data loaders for training set and training-, validation-, and test-set in evaluation mode
     (=no random subsampling) for datasets of category "simulated immunosequencing data".
     
    Parameters
    ----------
    dataset_path: str
        File path of dataset. If the dataset does not exist, the corresponding hdf5 container will be downloaded.
        Defaults to "deeprc_datasets/simulated_{dataset_id}.hdf5"
    dataset_id: int
        ID of dataset. Valid IDs: 0...17. See paper for specifications.
    cross_validation_fold : int
        Specify the fold of the cross-validation the dataloaders should be computed for.
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
    """
    # Get dataset file
    if dataset_path is None:
        dataset_path = os.path.join('deeprc_datasets', f'simulated_{dataset_id}.hdf5')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset to {dataset_path}")
        _url_get_(f"https://ml.jku.at/research/DeepRC/datasets/simulated_immunosequencing_data/hdf5/simulated_{dataset_id:03d}.hdf5",
                  dataset_path)
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper',
                              'simulated_immunosequencing.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    # Get data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(dataset_path=dataset_path, split_inds=split_inds, cross_validation_fold=cross_validation_fold,
                         n_worker_processes=n_worker_processes, batch_size=batch_size, inputformat=inputformat,
                         keep_dataset_in_ram=keep_dataset_in_ram, sample_n_sequences=sample_n_sequences,
                         target_label='status', true_class_label_value='+', verbose=verbose)
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader


def lstm_generated_dataset(dataset_path: str = None, dataset_id: int = 0,
                           cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                           inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                           sample_n_sequences: int = int(1e4), verbose: bool = True):
    """Get data loaders for category "LSTM-generated immunosequencing data with implanted signals".
     
     Get data loaders for training set and training-, validation-, and test-set in evaluation mode
     (=no random subsampling) for datasets of category "LSTM-generated immunosequencing data with implanted signals".
     
    Parameters
    ----------
    dataset_path: str
        File path of dataset. If the dataset does not exist, the corresponding hdf5 container will be downloaded.
        Defaults to "deeprc_datasets/LSTM_generated_{dataset_id}.hdf5"
    dataset_id: int
        ID of dataset.
        0 = "motif freq. 10%", 1 = "motif freq. 1%", 2 = "motif freq. 0.5%", 3 = "motif freq. 0.1%",
        4 = "motif freq. 0.05%"
    cross_validation_fold : int
        Specify the fold of the cross-validation the dataloaders should be computed for.
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
    """
    # Get dataset file
    if dataset_path is None:
        dataset_path = os.path.join('deeprc_datasets', f'LSTM_generated_{dataset_id}.hdf5')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset to {dataset_path}")
        _url_get_(f"https://ml.jku.at/research/DeepRC/datasets/LSTM_generated_data/hdf5/lstm_{dataset_id}.hdf5",
                  dataset_path)
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper',
                              'LSTM_generated.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    # Get data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(dataset_path=dataset_path, split_inds=split_inds, cross_validation_fold=cross_validation_fold,
                         n_worker_processes=n_worker_processes, batch_size=batch_size, inputformat=inputformat,
                         keep_dataset_in_ram=keep_dataset_in_ram, sample_n_sequences=sample_n_sequences,
                         target_label='status', true_class_label_value='+', verbose=verbose)
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader


def cmv_implanted_dataset(dataset_path: str = None, dataset_id: int = 0,
                          cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                          inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                          sample_n_sequences: int = int(1e4),  verbose: bool = True):
    """Get data loaders for category "real-world immunosequencing data with implanted signals".
     
     Get data loaders for training set and training-, validation-, and test-set in evaluation mode
     (=no random subsampling) for datasets of category "real-world immunosequencing data with implanted signals".
     
    Parameters
    ----------
    dataset_path: str
        File path of dataset. If the dataset does not exist, the corresponding hdf5 container will be downloaded.
        Defaults to "deeprc_datasets/CMV_with_implanted_signals_{dataset_id}.hdf5"
    dataset_id: int
        ID of dataset.
        0 = "One Motif 1%", 1 = "One 0.1%", 2 = "Multi 1%", 3 = "Multi 0.1%"
    cross_validation_fold : int
        Specify the fold of the cross-validation the dataloaders should be computed for.
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
    """
    # Get dataset file
    if dataset_path is None:
        dataset_path = os.path.join('deeprc_datasets', f'CMV_with_implanted_signals_{dataset_id}.hdf5')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset to {dataset_path}")
        _url_get_(f"https://ml.jku.at/research/DeepRC/datasets/CMV_data_with_implanted_signals/hdf5/implanted_signals_{dataset_id}.hdf5",
                  dataset_path)
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper',
                              'CMV_with_implanted_signals.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    # Get data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(dataset_path=dataset_path, split_inds=split_inds, cross_validation_fold=cross_validation_fold,
                         n_worker_processes=n_worker_processes, batch_size=batch_size, inputformat=inputformat,
                         keep_dataset_in_ram=keep_dataset_in_ram, sample_n_sequences=sample_n_sequences,
                         target_label='status', true_class_label_value='True', verbose=verbose)
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader


def cmv_dataset(dataset_path: str = os.path.join('deeprc_datasets', 'CMV.hdf5'),
                cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                sample_n_sequences: int = int(1e4), verbose: bool = True):
    """Get data loaders for category "real-world immunosequencing data"
     
     Get data loaders for training set and training-, validation-, and test-set in evaluation mode
     (=no random subsampling) for datasets of category "real-world immunosequencing data".
     This is a pre-processed version of the CMV dataset [1]_
     
    Parameters
    ----------
    dataset_path: str
        File path of dataset. If the dataset does not exist, the corresponding hdf5 container will be downloaded.
        Defaults to "deeprc_datasets/CMV.hdf5"
    cross_validation_fold : int
        Specify the fold of the cross-validation the dataloaders should be computed for.
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
    
    References
    -----
    .. [1] Emerson, R. O., DeWitt, W. S., Vignali, M., Gravley, J.,Hu, J. K., Osborne, E. J., Desmarais, C., Klinger,
     M.,Carlson, C. S., Hansen, J. A., et al. Immunosequencingidentifies signatures of cytomegalovirus exposure history
     and hla-mediated effects on the t cell repertoire.Naturegenetics, 49(5):659, 2017
    """
    # Get dataset file
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset to {dataset_path}")
        _url_get_(f"https://ml.jku.at/research/DeepRC/datasets/CMV_data/hdf5/cmv_emerson_2017.hdf5",
                  dataset_path)
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper', 'CMV_splits.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    # Get data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(dataset_path=dataset_path, split_inds=split_inds, cross_validation_fold=cross_validation_fold,
                         n_worker_processes=n_worker_processes, batch_size=batch_size, inputformat=inputformat,
                         keep_dataset_in_ram=keep_dataset_in_ram, sample_n_sequences=sample_n_sequences,
                         target_label='Known CMV status', true_class_label_value='+', verbose=verbose)
    return trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader
