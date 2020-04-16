# -*- coding: utf-8 -*-
"""
Training of DeepRC model
"""
import os
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn import metrics
from widis_lstm_tools.utils.collection import TeePrint, SaverLoader, close_all


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, show_progress: bool = True,
             device: torch.device = torch.device('cuda:0')):
    """Compute DeepRC model scores on given dataset for binary classification task
    
    See `deeprc/examples/` for examples.
    
    Parameters
    ----------
    model: torch.nn.Module
         deeprc.architectures.DeepRC or similar model as PyTorch module
    dataloader: torch.utils.data.DataLoader
         Data loader for dataset to calculate scores on
    show_progress: bool
         Show progressbar?
    device: torch.device
         Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.
         Currently, only devices which support 16 bit float are supported.
    
    Returns
    ---------
    roc_auc: float
        Area under the curve for receiver operating characteristic (AUC) score
    bacc: float
        Balanced accuracy score
    f1: float
        F1 score
    scoring_loss: float
        Network loss (including regularization penalty)
    """
    with torch.no_grad():
        model.to(device=device)
        sum_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device=device)
        scoring_loss = 0.
        scoring_predictions = []
        scoring_labels = []
        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model",
                                 disable=not show_progress, position=1):
            
            # Get samples as lists
            labels, inputs, sequence_lengths, counts_per_sequence, sample_ids = scoring_data
            
            # Apply attention-based sequence reduction and create minibatch
            labels, inputs, sequence_lengths, n_sequences = model.reduce_and_stack_minibatch(
                    labels, inputs, sequence_lengths, counts_per_sequence)
            
            # Compute predictions from reduced sequences
            logit_outputs = model(inputs, n_sequences)
            prediction = torch.sigmoid(logit_outputs)
            
            # Compute mean of losses on-the-fly
            scoring_loss += sum_cross_entropy(logit_outputs, labels[..., -1]) / len(dataloader.dataset)
            
            # Store predictions and labels
            scoring_predictions.append(prediction)
            scoring_labels.append(labels[..., -1])
        
        # Compute BACC, F1, and AUC score
        scoring_predictions = torch.cat(scoring_predictions, dim=0).float()
        scoring_predictions_threshold = (scoring_predictions > 0.5).float()
        scoring_labels = torch.cat(scoring_labels).float()
        
        scoring_labels = scoring_labels.cpu().numpy()
        scoring_predictions = scoring_predictions.cpu().numpy()
        scoring_predictions_threshold = scoring_predictions_threshold.cpu().numpy()
        
        roc_auc = metrics.roc_auc_score(scoring_labels, scoring_predictions, average=None)
        bacc = metrics.balanced_accuracy_score(y_true=scoring_labels, y_pred=scoring_predictions_threshold)
        f1 = metrics.f1_score(y_true=scoring_labels, y_pred=scoring_predictions_threshold, average='binary',
                              pos_label=1)
    return roc_auc, bacc, f1, scoring_loss


def train(model: torch.nn.Module,
          trainingset_dataloader: torch.utils.data.DataLoader, trainingset_eval_dataloader: torch.utils.data.DataLoader,
          validationset_eval_dataloader: torch.utils.data.DataLoader,
          results_directory: str = "results", n_updates: int = int(1e5), show_progress: bool = True,
          load_file: str = None, device: torch.device = torch.device('cuda:0'), rnd_seed: int = 0,
          num_torch_threads: int = 3, learning_rate: float = 1e-4, l1_weight_decay: float = 0,
          l2_weight_decay: float = 0, log_training_stats_at: int = int(1e2), evaluate_at: int = int(5e3)):
    """Train a DeepRC model on a given dataset on binary classification task using early stopping
    
    See `deeprc/examples/` for examples.
    
    Parameters
    ----------
    model: torch.nn.Module
         deeprc.architectures.DeepRC or similar model as PyTorch module
    trainingset_dataloader: torch.utils.data.DataLoader
         Data loader for training
    trainingset_eval_dataloader: torch.utils.data.DataLoader
         Data loader for evaluation on training set (=no random subsampling)
    validationset_eval_dataloader: torch.utils.data.DataLoader
         Data loader for evaluation on validation set (=no random subsampling).
         Will be used for early-stopping.
    results_directory: str
         Directory to save checkpoint of best trained model, logfile, and tensorboard files in
    n_updates: int
         Number of updates to train for
    show_progress: bool
         Show progressbar?
    load_file: str
         Path to load checkpoint of previously saved model from
    device: torch.device
         Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.
         Currently, only devices which support 16 bit float are supported.
    rnd_seed: int
         Random seed (will still be non-deterministic due to multiprocessing but weight initialization will be the same)
    num_torch_threads: int
         Number of parallel threads to allow PyTorch
    learning_rate: float
         Learning rate for adam optimizer
    l1_weight_decay: float
         l1 weight decay factor. l1 weight penalty will be added to loss, scaled by `l1_weight_decay`
    l2_weight_decay: float
         l2 weight decay factor. l2 weight penalty will be added to loss, scaled by `l2_weight_decay`
    log_training_stats_at: int
         Write current training statistics to tensorboard every `log_training_stats_at` updates
    evaluate_at: int
         Evaluate model on training and validation set every `evaluate_at` updates.
         This will also check for a new best model for early stopping.
    """
    # Append current timestamp to results directory
    results_directory = os.path.join(results_directory, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(results_directory, exist_ok=True)
    
    # Read config file and set up results folder
    logfile = os.path.join(results_directory, 'log.txt')
    checkpointdir = os.path.join(results_directory, 'checkpoint')
    os.makedirs(checkpointdir, exist_ok=True)
    tensorboarddir = os.path.join(results_directory, 'tensorboard')
    os.makedirs(tensorboarddir, exist_ok=True)
    
    # Prepare tensorboard writer
    writer = SummaryWriter(log_dir=tensorboarddir)
    
    # Print all outputs to logfile and terminal
    tee_print = TeePrint(logfile)
    tprint = tee_print.tee_print
    
    # Set up PyTorch and numpy random seeds
    try:
        torch.set_num_threads(num_torch_threads)
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)
        
        # Send model to device
        model.to(device)
        
        # Define loss function
        mean_cross_entropy = torch.nn.BCEWithLogitsLoss().to(device)
        
        # Get optimizer (eps needs to be at about 1e-4 to be numerically stable with 16 bit float)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight_decay, eps=1e-4)
        
        # Create a checkpoint dictionary with objects we want to have saved and loaded if needed
        state = dict(model=model, optimizer=optimizer, update=0, best_validation_loss=np.inf)
        
        # Setup the SaverLoader class to save/load our checkpoint dictionary to files or to RAM objects
        saver_loader = SaverLoader(save_dict=state, device=device, save_dir=checkpointdir,
                                   n_savefiles=1,  # keep only the latest checkpoint
                                   n_inmem=1  # save checkpoint only in RAM
                                   )
        
        # Load previous checkpoint dictionary, if load_file is specified
        if load_file is not None:
            state.update(saver_loader.load_from_file(loadname=load_file, verbose=True))
            tprint(f"Loaded checkpoint from file {load_file}")
        update, best_validation_loss = state['update'], state['best_validation_loss']
        
        # Save checkpoint dictionary to RAM object
        saver_loader.save_to_ram(savename=str(update))
        
        #
        # Start training
        #
        try:
            tprint("Training model...")
            update_progess_bar = tqdm(total=n_updates, disable=not show_progress, position=0,
                                      desc=f"loss={np.nan:6.4f}")
            while update < n_updates:
                for data in trainingset_dataloader:
                    # Get samples as lists
                    labels, inputs, sequence_lengths, counts_per_sequence, sample_ids = data
                    
                    # Apply attention-based sequence reduction and create minibatch
                    with torch.no_grad():
                        labels, inputs, sequence_lengths, n_sequences = model.reduce_and_stack_minibatch(
                                labels, inputs, sequence_lengths, counts_per_sequence)
                    
                    # Reset gradients
                    optimizer.zero_grad()
                    
                    # Calculate predictions from reduced sequences
                    logit_outputs = model(inputs, n_sequences)
                    
                    # Calculate losses
                    pred_loss = mean_cross_entropy(logit_outputs, labels[..., -1])
                    l1reg_loss = (torch.mean(torch.stack([p.abs().float().mean() for p in model.parameters()])))
                    loss = pred_loss + l1reg_loss * l1_weight_decay
                    
                    # Perform update
                    loss.backward()
                    optimizer.step()
                    
                    update += 1
                    update_progess_bar.update()
                    update_progess_bar.set_description(desc=f"loss={loss.item():6.4f}", refresh=True)
                    
                    # Add to tensorboard
                    if update % log_training_stats_at == 0:
                        tb_group = 'training/'
                        writer.add_scalar(tag=tb_group+'pred_loss', scalar_value=pred_loss, global_step=update)
                        writer.add_scalar(tag=tb_group+'l1reg_loss', scalar_value=l1reg_loss, global_step=update)
                        writer.add_scalar(tag=tb_group+'loss', scalar_value=loss, global_step=update)
                        writer.add_histogram(tag=tb_group+'logit_outputs', values=logit_outputs, global_step=update)
                    
                    # Calculate scores and loss on training set and validation set
                    if update % evaluate_at == 0 or update == n_updates or update == 1:
                        print("  Calculating training score...")
                        roc_auc, bacc, f1, scoring_loss = evaluate(model=model, dataloader=trainingset_eval_dataloader)
                        print(f" ...done!")
                        tprint(f"[training_inference] u: {update:07d}; roc_auc: {roc_auc:6.4f}; bacc: {bacc:6.4f}; "
                               f"f1: {f1:6.4f}; scoring_loss: {scoring_loss:6.4f}")
                        
                        tb_group = 'training_inference/'
                        writer.add_scalar(tag=tb_group+'roc_auc', scalar_value=roc_auc, global_step=update)
                        writer.add_scalar(tag=tb_group+'bacc', scalar_value=bacc, global_step=update)
                        writer.add_scalar(tag=tb_group+'f1', scalar_value=f1, global_step=update)
                        writer.add_scalar(tag=tb_group+'scoring_loss', scalar_value=scoring_loss, global_step=update)
                        
                        print("  Calculating validation score...")
                        roc_auc, bacc, f1, scoring_loss = evaluate(model=model, dataloader=validationset_eval_dataloader)
                        print(f" ...done!")
                        tprint(f"[validation] u: {update:07d}; roc_auc: {roc_auc:6.4f}; bacc: {bacc:6.4f}; "
                               f"f1: {f1:6.4f}; scoring_loss: {scoring_loss:6.4f}")
                        
                        tb_group = 'validation/'
                        writer.add_scalar(tag=tb_group+'roc_auc', scalar_value=roc_auc, global_step=update)
                        writer.add_scalar(tag=tb_group+'bacc', scalar_value=bacc, global_step=update)
                        writer.add_scalar(tag=tb_group+'f1', scalar_value=f1, global_step=update)
                        writer.add_scalar(tag=tb_group+'scoring_loss', scalar_value=scoring_loss, global_step=update)
                        writer.add_histogram(tag=tb_group+'weights',
                                             values=model.sequence_embedding_16bit.conv_aas.weight.cpu().detach(),
                                             global_step=update)
                        writer.add_histogram(tag=tb_group+'biases',
                                             values=model.sequence_embedding_16bit.conv_aas.bias.cpu().detach(),
                                             global_step=update)
                        
                        # If we have a new best loss on the validation set, we save the model as new best model
                        if best_validation_loss > scoring_loss:
                            best_validation_loss = scoring_loss
                            tprint(f"  New best validation loss: {scoring_loss}")
                            # Save current state as RAM object
                            state['update'] = update
                            state['best_validation_loss'] = scoring_loss
                            # Save checkpoint dictionary with currently best model to RAM
                            saver_loader.save_to_ram(savename=str(update))
                            # This would save to disk every time a new best model is found, which can be slow
                            # saver_loader.save_to_file(filename=f'best_so_far_u{update}.tar.gzip')
                    
                    if update >= n_updates:
                        break
            update_progess_bar.close()
        finally:
            # In any case, save the current model and best model to a file
            saver_loader.save_to_file(filename=f'lastsave_failed_u{update}.tar.gzip')
            state.update(saver_loader.load_from_ram())  # load best model so far
            saver_loader.save_to_file(filename=f'best_failed_u{update}.tar.gzip')
            print('Finished Training!')
    except Exception as e:
        with open(logfile, 'a') as lf:
            print(f"Exception: {e}", file=lf)
        raise e
    finally:
        close_all()  # Clean up
