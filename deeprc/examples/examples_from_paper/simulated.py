# -*- coding: utf-8 -*-
"""
Simple example for training binary DeepRC classifier on datasets of category "simulated immunosequencing data".
Warning: Filesize to download is ~20GB per dataset!

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import argparse
import numpy as np
import torch
from deeprc.predefined_datasets import simulated_dataset
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate
parser = argparse.ArgumentParser()
parser.add_argument('id', help='ID of dataset. Valid IDs: 0...17. See paper for specifications."', type=int)
parser.add_argument('--n_updates', help='Number of updates to train for. Default: int(1e5)', type=int,
                    default=int(1e5))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping.'
                                          ' Default: int(5e3)', type=int,
                    default=int(5e3))
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
# Get dataset
#
# Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
task_definition, trainingset, trainingset_eval, validationset_eval, testset_eval = simulated_dataset(dataset_id=args.id)


#
# Create DeepRC Network
#
# Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20+3, kernel_size=9, n_kernels=32, n_layers=1)
# Create attention network
attention_network = AttentionNetwork(n_input_features=32, n_layers=2, n_units=32)
# Create output network
output_network = OutputNetwork(n_input_features=32, n_output_features=task_definition.get_n_output_features(),
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
      trainingset_eval_dataloader=trainingset_eval,
      early_stopping_target_id='label',  # Get model that performs best for this task
      validationset_eval_dataloader=validationset_eval, n_updates=args.n_updates, evaluate_at=args.evaluate_at,
      device=device, results_directory=f"results/simulated_{args.id}"  # Here our results and trained models will be stored
      )
# You can use "tensorboard --logdir [results_directory] --port=6060" and open "http://localhost:6060/" in your
# web-browser to view the progress

#
# Evaluate trained model on testset
#
scores = evaluate(model=model, dataloader=testset_eval, task_definition=task_definition, device=device)
print(f"Test scores:\n{scores}")
