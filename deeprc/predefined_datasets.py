# -*- coding: utf-8 -*-
"""
Pre-defined data loaders for the 4 dataset categories in paper

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
import dill as pkl
from torch.utils.data import DataLoader
from typing import Tuple
import deeprc
from deeprc.utils import url_get, user_confirmation
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling, log_sequence_count_scaling
from deeprc.task_definitions import TaskDefinition, BinaryTarget


def simulated_dataset(dataset_path: str = None, dataset_id: int = 0, task_definition: TaskDefinition = None,
                      cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                      inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                      sample_n_sequences: int = int(1e4), verbose: bool = True) \
        -> Tuple[TaskDefinition, DataLoader, DataLoader, DataLoader, DataLoader]:
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', f'simulated')
    os.makedirs(dataset_path, exist_ok=True)
    metadata_file = os.path.join(dataset_path, f'simulated_{dataset_id:03d}_metadata.tsv')
    repertoiresdata_file = os.path.join(dataset_path, f'simulated_{dataset_id:03d}_repertoiresdata.hdf5')
    
    # Download metadata file
    if not os.path.exists(metadata_file):
        user_confirmation(f"File {metadata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/simulated_immunosequencing_data/metadata/implanted_signals_{dataset_id:03d}.tsv",
                metadata_file)
    
    # Download repertoire file
    if not os.path.exists(repertoiresdata_file):
        user_confirmation(f"File {repertoiresdata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/simulated_immunosequencing_data/hdf5/simulated_{dataset_id:03d}.hdf5",
                repertoiresdata_file)
    
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper',
                              'simulated_immunosequencing.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    
    # Get task_definition
    if task_definition is None:
        task_definition = TaskDefinition(targets=[BinaryTarget(column_name='label', true_class_value='1')])
    
    # Create data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(task_definition=task_definition, metadata_file=metadata_file,
                         repertoiresdata_path=repertoiresdata_file, split_inds=split_inds,
                         cross_validation_fold=cross_validation_fold, n_worker_processes=n_worker_processes,
                         batch_size=batch_size, inputformat=inputformat, keep_dataset_in_ram=keep_dataset_in_ram,
                         sample_n_sequences=sample_n_sequences, sequence_counts_scaling_fn=no_sequence_count_scaling,
                         metadata_file_id_column='Subject ID', verbose=verbose)
    return (task_definition, trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader,
            testset_eval_dataloader)


def lstm_generated_dataset(dataset_path: str = None, dataset_id: int = 0, task_definition: TaskDefinition = None,
                           cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                           inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                           sample_n_sequences: int = int(1e4), verbose: bool = True) \
        -> Tuple[TaskDefinition, DataLoader, DataLoader, DataLoader, DataLoader]:
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', f'LSTM_generated')
    os.makedirs(dataset_path, exist_ok=True)
    metadata_file = os.path.join(dataset_path, f'LSTM_generated_{dataset_id}_metadata.tsv')
    repertoiresdata_file = os.path.join(dataset_path, f'LSTM_generated_{dataset_id}_repertoiresdata.hdf5')
    
    # Download metadata file
    if not os.path.exists(metadata_file):
        user_confirmation(f"File {metadata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/LSTM_generated_data/metadata/lstm_{dataset_id}.tsv",
                metadata_file)
    
    # Download repertoire file
    if not os.path.exists(repertoiresdata_file):
        user_confirmation(f"File {repertoiresdata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/LSTM_generated_data/hdf5/lstm_{dataset_id}.hdf5",
                repertoiresdata_file)
    
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper',
                              'LSTM_generated.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    
    # Get task_definition
    if task_definition is None:
        task_definition = TaskDefinition(targets=[BinaryTarget(column_name='Known CMV status', true_class_value='+')])
    
    # Create data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(task_definition=task_definition, metadata_file=metadata_file,
                         repertoiresdata_path=repertoiresdata_file, split_inds=split_inds,
                         cross_validation_fold=cross_validation_fold, n_worker_processes=n_worker_processes,
                         batch_size=batch_size, inputformat=inputformat, keep_dataset_in_ram=keep_dataset_in_ram,
                         sample_n_sequences=sample_n_sequences, sequence_counts_scaling_fn=no_sequence_count_scaling,
                         metadata_file_id_column='Subject ID', verbose=verbose)
    return (task_definition, trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader,
            testset_eval_dataloader)


def cmv_implanted_dataset(dataset_path: str = None, dataset_id: int = 0, task_definition: TaskDefinition = None,
                          cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                          inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                          sample_n_sequences: int = int(1e4),  verbose: bool = True) \
        -> Tuple[TaskDefinition, DataLoader, DataLoader, DataLoader, DataLoader]:
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', f'CMV_with_implanted_signals')
    os.makedirs(dataset_path, exist_ok=True)
    metadata_file = os.path.join(dataset_path, f'CMV_with_implanted_signals_{dataset_id}_metadata.tsv')
    repertoiresdata_file = os.path.join(dataset_path, f'CMV_with_implanted_signals_{dataset_id}_repertoiresdata.hdf5')
    
    # Download metadata file
    if not os.path.exists(metadata_file):
        user_confirmation(f"File {metadata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        # url_get(f"https://ml.jku.at/research/DeepRC/datasets/CMV_data_with_implanted_signals/metadata/implanted_signals_{dataset_id}.csv",
        #         metadata_file)
        url_get(f"https://cloud.ml.jku.at/s/KQDAdHjHpdn3pzg/download?path=/datasets/CMV_data_with_implanted_signals/metadata&files=implanted_signals_{dataset_id}.tsv",
                metadata_file)
    
    # Download repertoire file
    if not os.path.exists(repertoiresdata_file):
        user_confirmation(f"File {repertoiresdata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/CMV_data_with_implanted_signals/hdf5/implanted_signals_{dataset_id}.hdf5",
                repertoiresdata_file)
    
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper',
                              'CMV_with_implanted_signals.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    
    # Get task_definition
    if task_definition is None:
        task_definition = TaskDefinition(targets=[BinaryTarget(column_name='status', true_class_value='True')])
    
    # Create data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(task_definition=task_definition, metadata_file=metadata_file,
                         repertoiresdata_path=repertoiresdata_file, split_inds=split_inds,
                         cross_validation_fold=cross_validation_fold, n_worker_processes=n_worker_processes,
                         batch_size=batch_size, inputformat=inputformat, keep_dataset_in_ram=keep_dataset_in_ram,
                         sample_n_sequences=sample_n_sequences, sequence_counts_scaling_fn=no_sequence_count_scaling,
                         verbose=verbose)
    return (task_definition, trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader,
            testset_eval_dataloader)


def cmv_dataset(dataset_path: str = None, task_definition: TaskDefinition = None,
                cross_validation_fold: int = 0, n_worker_processes: int = 4, batch_size: int = 4,
                inputformat: str = 'NCL', keep_dataset_in_ram: bool = True,
                sample_n_sequences: int = int(1e4), verbose: bool = True) \
        -> Tuple[TaskDefinition, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get data loaders for category "real-world immunosequencing data"
     
     Get data loaders for training set and training-, validation-, and test-set in evaluation mode
     (=no random subsampling) for datasets of category "real-world immunosequencing data".
     This is a pre-processed version of the CMV dataset [1]_.
     
    Parameters
    ----------
    dataset_path: str
        File path of dataset. If the dataset does not exist, the corresponding hdf5 container will be downloaded.
        Defaults to "deeprc_datasets/CMV.hdf5"
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
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
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', f'CMV')
    os.makedirs(dataset_path, exist_ok=True)
    metadata_file = os.path.join(dataset_path, f'CMV_metadata.tsv')
    repertoiresdata_file = os.path.join(dataset_path, f'CMV_repertoiresdata.hdf5')
    
    # Download metadata file
    if not os.path.exists(metadata_file):
        user_confirmation(f"File {metadata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/CMV_data/metadata/cmv_emerson_2017.tsv",
                metadata_file)
    
    # Download repertoire file
    if not os.path.exists(repertoiresdata_file):
        user_confirmation(f"File {repertoiresdata_file} not found. It will be downloaded now. Continue?", 'y', 'n')
        url_get(f"https://ml.jku.at/research/DeepRC/datasets/CMV_data/hdf5/cmv_emerson_2017.hdf5",
                repertoiresdata_file)
    
    # Get file for dataset splits
    split_file = os.path.join(os.path.dirname(deeprc.__file__), 'datasets', 'splits_used_in_paper', 'CMV_splits.pkl')
    with open(split_file, 'rb') as sfh:
        split_inds = pkl.load(sfh)
    
    # Get task_definition
    if task_definition is None:
        task_definition = TaskDefinition(targets=[BinaryTarget(column_name='Known CMV status', true_class_value='+')])
    
    # Create data loaders
    trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader, testset_eval_dataloader = \
        make_dataloaders(task_definition=task_definition, metadata_file=metadata_file,
                         repertoiresdata_path=repertoiresdata_file, split_inds=split_inds,
                         cross_validation_fold=cross_validation_fold, n_worker_processes=n_worker_processes,
                         batch_size=batch_size, inputformat=inputformat, keep_dataset_in_ram=keep_dataset_in_ram,
                         sample_n_sequences=sample_n_sequences, sequence_counts_scaling_fn=log_sequence_count_scaling,
                         metadata_file_id_column='Subject ID', verbose=verbose)
    return (task_definition, trainingset_dataloader, trainingset_eval_dataloader, validationset_eval_dataloader,
            testset_eval_dataloader)
