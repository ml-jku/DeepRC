# -*- coding: utf-8 -*-
"""
Simple example for training binary DeepRC classifier on datasets of category "real-world data"
"""

import argparse
from deeprc.deeprc_binary.predefined_datasets import cmv_dataset
from deeprc.deeprc_binary.architectures import DeepRC
from deeprc.deeprc_binary.training import train, evaluate
parser = argparse.ArgumentParser()
parser.add_argument('--n_updates', help='Number of updates to train for. Default: int(1e5)', type=int,
                    default=int(1e5))
parser.add_argument('--evaluate_at', help='Evaluate model on training and validation set every `evaluate_at` updates. '
                                          'This will also check for a new best model for early stopping.'
                                          ' Default: int(5e3)', type=int,
                    default=int(5e3))
args = parser.parse_args()

# Get data loaders for training set and training-, validation-, and test-set in evaluation mode (=no random subsampling)
trainingset, trainingset_eval, validationset_eval, testset_eval = cmv_dataset()

# Create DeepRC model
model = DeepRC(n_input_features=23, n_output_features=1, max_seq_len=30, kernel_size=9, consider_seq_counts=True)

# Train DeepRC model
train(model, trainingset_dataloader=trainingset, trainingset_eval_dataloader=trainingset_eval,
      validationset_eval_dataloader=validationset_eval, n_updates=args.n_updates, evaluate_at=args.evaluate_at,
      results_directory="results/simple_cmv")
# You can use "tensorboard --logdir [results_directory] --port=6060" and open "http://localhost:6060/" in your
# web-browser to view the progress

# Evaluate trained model
roc_auc, bacc, f1, scoring_loss = evaluate(model=model, dataloader=testset_eval)

print(f"Test scores:\nroc_auc: {roc_auc:6.4f}; bacc: {bacc:6.4f}; f1:{f1:6.4f}; scoring_loss: {scoring_loss:6.4f}")
