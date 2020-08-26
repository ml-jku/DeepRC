import argparse
import json
import numpy as np
import os
import torch

from pathlib import Path
from typing import List, Tuple, Union

from logisticirc import LogisticMIL, LogisticMILDataReader


def tuple_of_int_t(argument: Union[str, List[str]]) -> Tuple[int, ...]:
    argument = argument[0] if type(argument) == list else argument
    try:
        return tuple(int(_.strip()) for _ in argument.split(r','))
    except ValueError:
        raise argparse.ArgumentError(r'Argument is not a valid tuple of integers!')


def tuple_of_float_t(argument: Union[str, List[str]]) -> Tuple[float, ...]:
    argument = argument[0] if type(argument) == list else argument
    try:
        return tuple(float(_.strip()) for _ in argument.split(r','))
    except ValueError:
        raise argparse.ArgumentError(r'Argument is not a valid tuple of floating point numbers!')


def console_entry():
    # Initialise argument parsers for ...
    arg_parser = argparse.ArgumentParser(description=r'Logistic MIL model for immune repertoire classification')
    arg_sub_parsers = arg_parser.add_subparsers(dest=r'mode', required=True)
    # ... data set adaption.
    adapt_parser = arg_sub_parsers.add_parser(
        name=r'adapt', help=r'adapt data set to be compatible with logistic MIL')
    adapt_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data file (h5py) to use', required=True)
    adapt_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to resulting data file (h5py)', required=True)
    adapt_parser.add_argument(
        r'-z', r'--kmer_size', type=int, help=r'size <k> of a k-mer', default=4)
    adapt_parser.add_argument(
        r'-k', r'--worker', type=int, help=r'number of worker processes (data reading)', default=-1)
    # ... hyperparameter optimisation.
    hyper_parser = arg_sub_parsers.add_parser(name=r'optim', help=r'perform hyperparameter optimisation')
    hyper_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data file (h5py) to use', required=True)
    hyper_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to store best hyperparameters', required=True)
    hyper_parser.add_argument(
        r'-g', r'--log_dir', type=str, help=r'directory to store TensorBoard logs', default=None)
    hyper_parser.add_argument(
        r'-n', r'--normalise', help=r'normalise features', action=r'store_true')
    hyper_parser.add_argument(
        r'-a', r'--normalise_abundance', help=r'normalise relative abundance term', action=r'store_true')
    hyper_parser.add_argument(
        r'-c', r'--abundance', type=LogisticMILDataReader.RelativeAbundance, help=r'type of abundance', required=True)
    hyper_parser_folds = hyper_parser.add_mutually_exclusive_group(required=False)
    hyper_parser_folds.add_argument(
        r'-f', r'--folds', type=int, help=r'number of folds')
    hyper_parser_pickle = hyper_parser_folds.add_argument_group()
    hyper_parser_pickle.add_argument(
        r'-z', r'--pickle', type=str, help=r'fold definitions (pickle-file)')
    hyper_parser_pickle.add_argument(
        r'--offset', type=int, help=r'offset defining the folds for training/evaluation/test splits', default=0)
    hyper_parser.add_argument(
        r'-e', r'--epochs', type=int, help=r'maximum number of epochs to optimise', default=1)
    hyper_parser.add_argument(
        r'-b', r'--batch_size', type=str, help=r'list of batch sizes', nargs=1, required=True)
    hyper_parser.add_argument(
        r'-q', r'--top_n', type=str, help=r'tuple of top <n> entities considered per sample', nargs=1, default=r'1')
    hyper_parser.add_argument(
        r'-r', r'--randomise', help=r'randomise batches between epochs', action=r'store_true')
    hyper_parser.add_argument(
        r'-l', r'--learning_rate', type=str, help=r'tuple of learning rates (Adam)', nargs=1, required=True)
    hyper_parser.add_argument(
        r'-x', r'--beta_one', type=str, help=r'tuple of beta 1 (Adam)', nargs=1, default=r'0.9')
    hyper_parser.add_argument(
        r'-y', r'--beta_two', type=str, help=r'tuple of beta 2 (Adam)', nargs=1, default=r'0.999')
    hyper_parser.add_argument(
        r'-w', r'--weight_decay', type=str, help=r'tuple of weight decay terms (Adam)', nargs=1, default=r'0.0')
    hyper_parser.add_argument(
        r'-v', r'--amsgrad', help=r'use AMSGrad version of Adam', action=r'store_true')
    hyper_parser.add_argument(
        r'-p', r'--epsilon', type=float, help=r'epsilon to use for numerical stability', default=1e-8)
    hyper_parser.add_argument(
        r'-s', r'--seed', type=int, help=r'seed to be used for reproducibility', default=42)
    hyper_parser.add_argument(
        r'-k', r'--worker', type=int, help=r'number of worker processes (data reading)', default=0)
    hyper_parser.add_argument(
        r'-d', r'--device', type=str, help=r'device to use for heavy computations', default=r'cpu')
    # ... training.
    train_parser = arg_sub_parsers.add_parser(
        name=r'train', help=r'train logistic regression model')
    train_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data set (h5py) to use', required=True)
    train_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to store resulting model', required=True)
    train_parser.add_argument(
        r'-n', r'--normalise', help=r'normalise features', action=r'store_true')
    train_parser.add_argument(
        r'-a', r'--normalise_abundance', help=r'normalise relative abundance term', action=r'store_true')
    train_parser.add_argument(
        r'-c', r'--abundance', type=LogisticMILDataReader.RelativeAbundance, help=r'type of abundance', required=True)
    train_parser.add_argument(
        r'-r', r'--randomise', help=r'randomise batches between epochs', action=r'store_true')
    train_parser.add_argument(
        r'-s', r'--seed', type=int, help=r'seed to be used for reproducibility', default=42)
    train_parser.add_argument(
        r'-k', r'--worker', type=int, help=r'number of worker processes (data reading)', default=0)
    train_parser.add_argument(
        r'-d', r'--device', type=str, help=r'device to use for heavy computations', default=r'cpu')
    train_parser_main_group = train_parser.add_mutually_exclusive_group(required=False)
    train_parser_main_group.add_argument(
        r'-j', r'--json', type=str, help=r'hyperparameters to use (json)')
    train_cmd = train_parser_main_group.add_argument_group()
    train_cmd.add_argument(
        r'-e', r'--epochs', type=int, help=r'number of epochs to train', default=1)
    train_cmd.add_argument(
        r'-b', r'--batch_size', type=int, help=r'batch size to be applied', default=1)
    train_cmd.add_argument(
        r'-q', r'--top_n', type=int, help=r'consider top <n> entities per sample', default=1)
    train_cmd.add_argument(
        r'-l', r'--learning_rate', type=float, help=r'learning rate (Adam)', default=1e-3)
    train_cmd.add_argument(
        r'-x', r'--beta_one', type=float, help=r'beta 1 of Adam optimiser', default=0.9)
    train_cmd.add_argument(
        r'-y', r'--beta_two', type=float, help=r'beta 2 of Adam optimiser', default=0.999)
    train_cmd.add_argument(
        r'-w', r'--weight_decay', type=float, help=r'weight decay (Adam)', default=0.0)
    train_cmd.add_argument(
        r'-v', r'--amsgrad', help=r'use AMSGrad version of Adam', action=r'store_true')
    train_cmd.add_argument(
        r'-p', r'--epsilon', type=float, help=r'epsilon to use for numerical stability', default=1e-8)
    # ...prediction.
    predict_parser = arg_sub_parsers.add_parser(
        name=r'predict', help=r'predict cancer using pre-trained model')
    predict_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data set (h5py) to use', required=True)
    predict_parser.add_argument(
        r'-o', r'--output_dir', type=str, help=r'directory to store predictions (and ROC AUC)', default=None)
    predict_parser.add_argument(
        r'-a', r'--activations', action=r'store_true', help=r'compute activations instead of discrete predictions')
    predict_parser.add_argument(
        r'-m', r'--model', type=str, help=r'model to be used for prediction', required=True)
    predict_parser.add_argument(
        r'-z', r'--pickle', type=str, help=r'fold definitions (pickle-file)', default=None)
    predict_parser.add_argument(
        r'-k', r'--worker', type=int, help=r'number of worker processes (data reading)', default=0)
    predict_parser.add_argument(
        r'-d', r'--device', type=str, help=r'device to use for heavy computations', default=r'cpu')
    predict_parser.add_argument(
        r'-l', r'--offset', type=int, help=r'offset defining the folds for training/evaluation/test splits', default=0)
    # Parse arguments.
    args = arg_parser.parse_args()

    # Execute logistic MIL
    if args.mode == r'adapt':

        # Adapt data set (compute auxiliary features).
        LogisticMILDataReader.adapt(file_path=Path(args.input), store_path=args.output, kmer_size=args.kmer_size,
                                    num_workers=args.worker, dtype=np.float16)

    elif args.mode == r'optim':

        # Manually parse custom arguments.
        batch_sizes = tuple_of_int_t(argument=args.batch_size)
        top_n_samples = tuple_of_int_t(argument=args.top_n)
        learning_rates = tuple_of_float_t(argument=args.learning_rate)
        betas_one = tuple_of_float_t(argument=args.beta_one)
        betas_two = tuple_of_float_t(argument=args.beta_two)
        weight_decays = tuple_of_float_t(argument=args.weight_decay)

        # Create and optimise logistic MIL
        logistic_mil = LogisticMIL(file_path=Path(args.input), relative_abundance=args.abundance,
                                   fold_info=args.folds if (args.pickle is None) else Path(args.pickle),
                                   num_workers=args.worker, device=args.device, dtype=torch.float32,
                                   test_mode=False, offset=args.offset)
        hyperparameters = logistic_mil.optimise(epochs=args.epochs, batch_sizes=batch_sizes, top_n=top_n_samples,
                                                learning_rates=learning_rates, betas_one=betas_one, betas_two=betas_two,
                                                weight_decays=weight_decays, amsgrad=args.amsgrad, epsilon=args.epsilon,
                                                normalise=args.normalise, normalise_abundance=args.normalise_abundance,
                                                randomise=args.randomise, repetitions=0, seed=args.seed,
                                                log_dir=None if args.log_dir is None else Path(args.log_dir))

        # Store best hyperparameters as obtained by grid search.
        output_directory = os.path.dirname(args.output)
        if (len(output_directory) > 0) and (not os.path.exists(output_directory)):
            os.makedirs(output_directory)
        with open(args.output, r'w') as hyperparameters_json:
            json.dump(hyperparameters, hyperparameters_json)

    elif args.mode == r'train':

        # Process data file according to logistic MIL.
        logistic_mil = LogisticMIL(file_path=Path(args.input), relative_abundance=args.abundance,
                                   fold_info=None, num_workers=args.worker, device=args.device, dtype=torch.float32,
                                   test_mode=False)

        # Fetch hyperparameters to be used.
        if args.json is not None:
            with open(args.json, r'r') as hyperparameters_json:
                hyperparameters = json.load(hyperparameters_json)
        else:
            hyperparameters = {r'epochs': args.epochs, r'batch_size': args.batch_size, r'top_n': args.top_n,
                               r'learning_rate': args.learning_rate, r'beta_one': args.beta_one,
                               r'beta_two': args.beta_two, r'weight_decay': args.weight_decay,
                               r'amsgrad': args.amsgrad, r'epsilon': args.epsilon}

        # Train logistic MIL model.
        logistic_mil.train(file_path_output=Path(args.output), epochs=hyperparameters[r'epochs'],
                           batch_size=hyperparameters[r'batch_size'], top_n=hyperparameters[r'top_n'],
                           learning_rate=hyperparameters[r'learning_rate'], beta_one=hyperparameters[r'beta_one'],
                           beta_two=hyperparameters[r'beta_two'], weight_decay=hyperparameters[r'weight_decay'],
                           amsgrad=hyperparameters[r'amsgrad'], epsilon=hyperparameters[r'epsilon'],
                           normalise=args.normalise, normalise_abundance=args.normalise_abundance,
                           randomise=args.randomise, seed=args.seed)

    elif args.mode == r'predict':

        # Check output directory.
        output_result, output_roc_auc = None, None
        output_directory = None if ((args.output_dir is None) or (len(args.output_dir) == 0)) else Path(args.output_dir)
        if output_directory is not None:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_result = open(str(output_directory / r'predictions.txt'), mode=r'w')
            if args.pickle is not None:
                output_roc_auc = open(str(output_directory / r'roc_auc.txt'), mode=r'w')

        # Fetch auxiliary information from trained logistic MIL model.
        state_dict = torch.load(args.model)
        relative_abundance = LogisticMILDataReader.RelativeAbundance[state_dict[r'abundance'].strip().upper()]
        del state_dict

        # Predict using pre-trained logistic MIL. model.
        logistic_mil = LogisticMIL(file_path=Path(args.input), relative_abundance=relative_abundance,
                                   fold_info=None if (args.pickle is None) else Path(args.pickle),
                                   num_workers=args.worker, device=args.device, dtype=torch.float32,
                                   test_mode=True, offset=args.offset)
        result = logistic_mil.predict_from_path(file_path_model=Path(args.model), activations=args.activations)

        # Print (or store) results.
        if output_result is None:
            print(f'[ROC AUC]\n{result[1]}', end='\n')
        elif result[1] is not None:
            print(result[1], end='\n', file=output_roc_auc)
        if output_directory is None:
            print(r'[PREDICTIONS]', end='\n')
        for prediction in result[0]:
            print(prediction, end='\n', file=output_result)

    else:
        raise ValueError(r'Invalid <mode> specified! Aborting...')
