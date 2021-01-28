# -*- coding: utf-8 -*-
"""
Example for training DeepRC with LSTM sequence embedding in a multi-task setting.
IMPORTANT: The used task is a small dummy-task with random data, so DeepRC should over-fit on the main task on the
training set.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import argparse
import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, MulticlassTarget, RegressionTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingLSTM, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate


#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Recommended: int(1e5). Default: int(1e3)',
                    type=int, default=int(1e3))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping. '
                                          'Recommended: int(5e3). Default: int(1e2).',
                    type=int, default=int(1e2))
parser.add_argument('--n_lstm_blocks', help='Number of LSTM blocks per LSTM layer. Default: 32',
                    type=int, default=32)
parser.add_argument('--sample_n_sequences', help='Number of instances to reduce repertoires to during training via'
                                                 'random dropout. This should be less than the number of instances per '
                                                 'repertoire. Only applied during training, not for evaluation. '
                                                 'Default: int(1e4)',
                    type=int, default=int(1e4))
parser.add_argument('--learning_rate', help='Learning rate of DeepRC using Adam optimizer. Default: 1e-4',
                    type=float, default=1e-4)
parser.add_argument('--device', help='Device to use for NN computations, as passed to `torch.device()`. '
                                     'Default: "cuda:0".',
                    type=str, default="cuda:0")
parser.add_argument('--rnd_seed', help='Random seed to use for PyTorch and NumPy. Results will still be '
                                       'non-deterministic due to multiprocessing but weight initialization will be the'
                                       ' same). Default: 0.',
                    type=int, default=0)
args = parser.parse_args()
# Set computation device
device = torch.device(args.device)
# Set random seed (will still be non-deterministic due to multiprocessing but weight initialization will be the same)
torch.manual_seed(args.rnd_seed)
np.random.seed(args.rnd_seed)


#
# Create Task definitions
#
# Assume we want to train on 1 main task and 5 auxiliary tasks. We will set the task-weight of the main task to 1 and
# of the auxiliary tasks to 0.1/5. The tasks-weight is used to compute the training loss as weighted sum of the
# individual tasks losses.
aux_task_weight = 0.1 / 5
# Below we define how the tasks should be extracted from the metadata file. We can choose between combinations of
# binary, regression, and multiclass tasks. The column names have to be found in the metadata file.
task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
    BinaryTarget(  # Add binary classification task with sigmoid output function
            column_name='binary_target_1',  # Column name of task in metadata file
            true_class_value='+',  # Entries with value '+' will be positive class, others will be negative class
            pos_weight=1.,  # We can up- or down-weight the positive class if the classes are imbalanced
            task_weight=1.  # Weight of this task for the total training loss
    ),
    BinaryTarget(  # Add another binary classification task
            column_name='binary_target_2',
            true_class_value='True',  # Entries with value 'True' will be positive class, others will be negative class
            pos_weight=1./3,  # Down-weights the positive class samples (e.g. if the positive class is underrepresented)
            task_weight=aux_task_weight
    ),
    RegressionTarget(  # Add regression task with linear output function
            column_name='regression_target_1',  # Column name of task in metadata file
            normalization_mean=0., normalization_std=275.,  # Normalize targets by ((target_value - mean) / std)
            task_weight=aux_task_weight  # Weight of this task for the total training loss
    ),
    RegressionTarget(  # Add another regression task
            column_name='regression_target_2',
            normalization_mean=0.5, normalization_std=1.,
            task_weight=aux_task_weight
    ),
    MulticlassTarget(  # Add multiclass classification task with softmax output function (=classes mutually exclusive)
            column_name='multiclass_target_1',  # Column name of task in metadata file
            possible_target_values=['class_a', 'class_b', 'class_c'],  # Values in task column to expect
            class_weights=[1., 1., 0.5],  # Weight individual classes (e.g. if class 'class_c' is overrepresented)
            task_weight=aux_task_weight  # Weight of this task for the total training loss
    ),
    MulticlassTarget(  # Add another multiclass classification task
            column_name='multiclass_target_2',
            possible_target_values=['type_1', 'type_2', 'type_3', 'type_4', 'type_5'],
            class_weights=[1., 1., 1., 1., 1.],
            task_weight=aux_task_weight
    )
]).to(device=device)


#
# Get dataset
#
# Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
        task_definition=task_definition,
        metadata_file="../datasets/example_dataset/metadata.tsv",
        repertoiresdata_path="../datasets/example_dataset/repertoires",
        metadata_file_id_column='ID',
        sequence_column='amino_acid',
        sequence_counts_column='templates',
        sample_n_sequences=args.sample_n_sequences,
        sequence_counts_scaling_fn=no_sequence_count_scaling  # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
)


#
# Create DeepRC Network
#
# Create sequence embedding network (for LSTM, n_lstm_blocks is an important hyper-parameter)
sequence_embedding_network = SequenceEmbeddingLSTM(n_input_features=20+3, n_lstm_blocks=args.n_lstm_blocks, n_layers=1)
# Create attention network
attention_network = AttentionNetwork(n_input_features=args.n_lstm_blocks, n_layers=2, n_units=32)
# Create output network
output_network = OutputNetwork(n_input_features=args.n_lstm_blocks, n_output_features=task_definition.get_n_output_features(),
                               n_layers=1, n_units=32)
# Combine networks to DeepRC network
model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
               attention_network=attention_network,
               output_network=output_network,
               consider_seq_counts=False, n_input_features=20, add_positional_information=True,
               sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
               device=device).to(device=device)


#
# Train DeepRC model
#
train(model, task_definition=task_definition, trainingset_dataloader=trainingset,
      trainingset_eval_dataloader=trainingset_eval, learning_rate=args.learning_rate,
      early_stopping_target_id='binary_target_1',  # Get model that performs best for this task
      validationset_eval_dataloader=validationset_eval, n_updates=args.n_updates, evaluate_at=args.evaluate_at,
      device=device, results_directory="results/multitask_lstm"  # Here our results and trained models will be stored
      )
# You can use "tensorboard --logdir [results_directory] --port=6060" and open "http://localhost:6060/" in your
# web-browser to view the progress


#
# Evaluate trained model on testset
#
scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition, device=device)
print(f"Test scores:\n{scores}")
