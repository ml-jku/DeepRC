# -*- coding: utf-8 -*-
"""
Training and evaluation of DeepRC model

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from widis_lstm_tools.utils.collection import TeePrint, SaverLoader, close_all
from deeprc.task_definitions import TaskDefinition


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, task_definition: TaskDefinition,
             show_progress: bool = True, device: torch.device = torch.device('cuda:0')) -> dict:
    """Compute DeepRC model scores on given dataset for tasks specified in `task_definition`
    
    Parameters
    ----------
    model: torch.nn.Module
         deeprc.architectures.DeepRC or similar model as PyTorch module
    dataloader: torch.utils.data.DataLoader
         Data loader for dataset to calculate scores on
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
    show_progress: bool
         Show progressbar?
    device: torch.device
         Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.
    
    Returns
    ---------
    scores: dict
        Nested dictionary of format `{task_id: {score_id: score_value}}`, e.g.
        `{"binary_task_1": {"auc": 0.6, "bacc": 0.5, "f1": 0.2, "loss": 0.01}}`. The scores returned are computed using
        the .get_scores() methods of the individual target instances (e.g. `deeprc.task_definitions.BinaryTarget()`).
        See `deeprc/examples/` for examples.
    """
    with torch.no_grad():
        model.to(device=device)
        all_raw_outputs = []
        all_targets = []
        
        for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model", disable=not show_progress):
            
            # Get samples as lists
            targets, inputs, sequence_lengths, counts_per_sequence, sample_ids = scoring_data
            
            # Apply attention-based sequence reduction and create minibatch
            targets, inputs, sequence_lengths, n_sequences = model.reduce_and_stack_minibatch(
                    targets, inputs, sequence_lengths, counts_per_sequence)
            
            # Compute predictions from reduced sequences
            raw_outputs = model(inputs_flat=inputs, sequence_lengths_flat=sequence_lengths,
                                n_sequences_per_bag=n_sequences)
            
            # Store predictions and labels
            all_raw_outputs.append(raw_outputs.detach())
            all_targets.append(targets.detach())
        
        # Compute scores
        all_raw_outputs = torch.cat(all_raw_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        scores = task_definition.get_scores(raw_outputs=all_raw_outputs, targets=all_targets)
    return scores


def train(model: torch.nn.Module, task_definition: TaskDefinition, early_stopping_target_id: str,
          trainingset_dataloader: torch.utils.data.DataLoader, trainingset_eval_dataloader: torch.utils.data.DataLoader,
          validationset_eval_dataloader: torch.utils.data.DataLoader,
          results_directory: str = "results", n_updates: int = int(1e5), show_progress: bool = True,
          load_file: str = None, device: torch.device = torch.device('cuda:0'),
          num_torch_threads: int = 3, learning_rate: float = 1e-4, l1_weight_decay: float = 0,
          l2_weight_decay: float = 0, log_training_stats_at: int = int(1e2), evaluate_at: int = int(5e3),
          ignore_missing_target_values: bool = True):
    """Train a DeepRC model on a given dataset on tasks specified in `task_definition`
     
     Model with lowest validation set loss on target `early_stopping_target_id` will be taken as final model (=early
     stopping). Model performance on validation set will be evaluated every `evaluate_at` updates.
     Trained model, logfile, and tensorboard files will be stored in `results_directory`.
    
    See `deeprc/examples/` for examples.
    
    Parameters
    ----------
    model: torch.nn.Module
         deeprc.architectures.DeepRC or similar model as PyTorch module
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
    early_stopping_target_id: str
        ID of task in TaskDefinition object to use for early stopping.
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
    ignore_missing_target_values: bool
         If True, missing target values will be ignored for training. This can be useful if auxiliary tasks are not
         available for all samples but might increase the computation time per update.
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
    
    try:
        # Set up PyTorch and numpy random seeds
        torch.set_num_threads(num_torch_threads)
        
        # Send model to device
        model.to(device)
        
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
                    
                    # Calculate predictions from reduced sequences,
                    logit_outputs = model(inputs_flat=inputs, sequence_lengths_flat=sequence_lengths,
                                          n_sequences_per_bag=n_sequences)
                    
                    # Calculate losses
                    pred_loss = task_definition.get_loss(raw_outputs=logit_outputs, targets=labels,
                                                         ignore_missing_target_values=ignore_missing_target_values)
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
                        # Loop through tasks and add losses to tensorboard
                        pred_losses = task_definition.get_losses(raw_outputs=logit_outputs, targets=labels)
                        pred_losses = pred_losses.mean(dim=1)  # shape: (n_tasks, n_samples, 1) -> (n_tasks, 1)
                        for task_id, task_loss in zip(task_definition.get_task_ids(), pred_losses):
                            writer.add_scalar(tag=tb_group+f'{task_id}_loss', scalar_value=task_loss,
                                              global_step=update)
                        writer.add_scalar(tag=tb_group+'total_task_loss', scalar_value=pred_loss, global_step=update)
                        writer.add_scalar(tag=tb_group+'l1reg_loss', scalar_value=l1reg_loss, global_step=update)
                        writer.add_scalar(tag=tb_group+'total_loss', scalar_value=loss, global_step=update)
                        writer.add_histogram(tag=tb_group+'logit_outputs', values=logit_outputs, global_step=update)
                    
                    # Calculate scores and loss on training set and validation set
                    if update % evaluate_at == 0 or update == n_updates or update == 1:
                        print("  Calculating training score...")
                        scores = evaluate(model=model, dataloader=trainingset_eval_dataloader,
                                          task_definition=task_definition, device=device)
                        print(f" ...done!")
                        tprint(f"[training_inference] u: {update:07d}; scores: {scores};")
                        
                        tb_group = 'training_inference/'
                        for task_id, task_scores in scores.items():
                            [writer.add_scalar(tag=tb_group+f'{task_id}/{score_name}', scalar_value=score,
                                               global_step=update)
                             for score_name, score in task_scores.items()]
                        
                        print("  Calculating validation score...")
                        scores = evaluate(model=model, dataloader=validationset_eval_dataloader,
                                          task_definition=task_definition, device=device)
                        scoring_loss = scores[early_stopping_target_id]['loss']
                        
                        print(f" ...done!")
                        tprint(f"[validation] u: {update:07d}; scores: {scores};")
                        
                        tb_group = 'validation/'
                        for task_id, task_scores in scores.items():
                            [writer.add_scalar(tag=tb_group+f'{task_id}/{score_name}', scalar_value=score,
                                               global_step=update)
                             for score_name, score in task_scores.items()]
                        
                        # If we have a new best loss on the validation set, we save the model as new best model
                        if best_validation_loss > scoring_loss:
                            best_validation_loss = scoring_loss
                            tprint(f"  New best validation loss for {early_stopping_target_id}: {scoring_loss}")
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
            saver_loader.save_to_file(filename=f'lastsave_u{update}.tar.gzip')
            state.update(saver_loader.load_from_ram())  # load best model so far
            saver_loader.save_to_file(filename=f'best_u{update}.tar.gzip')
            print('Finished Training!')
    except Exception as e:
        with open(logfile, 'a') as lf:
            print(f"Exception: {e}", file=lf)
        raise e
    finally:
        close_all()  # Clean up
