import argparse
import json
import numpy as np
import os

from joblib import load
from pathlib import Path
from svmirc import SVMBaseline, SVMDataReader


def console_entry():
    # Initialise argument parsers for ...
    arg_parser = argparse.ArgumentParser(
        description=r'SVM model for immune repertoire classification')
    arg_sub_parsers = arg_parser.add_subparsers(
        dest=r'mode', required=True)
    # ... data set adaption.
    adapt_parser = arg_sub_parsers.add_parser(
        name=r'adapt', help=r'adapt data set to be compatible with SVM baseline')
    adapt_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data file (h5py) to use', required=True)
    adapt_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to resulting data file (h5py)', required=True)
    adapt_parser.add_argument(
        r'-z', r'--kmer_size', type=int, help=r'size <k> of a k-mer', default=4)
    adapt_parser.add_argument(
        r'-w', r'--worker', type=int, help=r'number of worker proc. (data reading)', default=-1)
    # ... sequence analysis.
    analysis_parser = arg_sub_parsers.add_parser(
        name=r'analyse', help=r'analyse data set with respect to sequence counts per sample')
    analysis_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data file (h5py) to use', required=True)
    analysis_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to resulting data file (h5py)', required=True)
    analysis_parser.add_argument(
        r'-z', r'--kmer_size', type=int, help=r'size <k> of a k-mer', default=4)
    analysis_parser.add_argument(
        r'-w', r'--worker', type=int, help=r'number of worker proc. (data reading)', default=-1)
    # ... hyperparameter optimisation.
    hyper_parser = arg_sub_parsers.add_parser(
        name=r'optim', help=r'perform hyperparameter optimisation')
    hyper_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data file (h5py) to use', required=True)
    hyper_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to store best hyperparameters', required=True)
    hyper_parser.add_argument(
        r'-g', r'--log_dir', type=str, help=r'directory to store TensorBoard logs', default=None)
    hyper_parser.add_argument(
        r'-k', r'--kernel', type=SVMDataReader.Kernel, help=r'type of kernel', required=True)
    hyper_parser_folds = hyper_parser.add_mutually_exclusive_group(required=False)
    hyper_parser_folds.add_argument(
        r'-f', r'--folds', type=int, help=r'number of folds')
    hyper_parser_pickle = hyper_parser_folds.add_argument_group()
    hyper_parser_pickle.add_argument(
        r'-z', r'--pickle', type=str, help=r'fold definitions (pickle-file)')
    hyper_parser_pickle.add_argument(
        r'-l', r'--offset', type=int, help=r'offset defining the folds for training/evaluation/test splits', default=0)
    hyper_parser.add_argument(
        r'-t', r'--trials', type=int, help=r'number of trials', required=True)
    hyper_parser.add_argument(
        r'-c', r'--penalty', type=float, help=r'range of penalty parameter of the C-SVM', nargs=2, required=True)
    hyper_parser.add_argument(
        r'-p', r'--epsilon', type=float, help=r'tolerance for stopping criterion', default=1e-3)
    hyper_parser.add_argument(
        r'-s', r'--seed', type=int, help=r'seed to be used for reproducibility', default=42)
    # ... training.
    train_parser = arg_sub_parsers.add_parser(name=r'train', help=r'train SVM baseline model')
    train_parser.add_argument(
        r'-i', r'--input', type=str, help=r'data set (h5py) to use', required=True)
    train_parser.add_argument(
        r'-o', r'--output', type=str, help=r'path to store resulting model', required=True)
    train_parser.add_argument(
        r'-k', r'--kernel', type=SVMDataReader.Kernel, help=r'type of kernel', required=True)
    train_parser.add_argument(
        r'-s', r'--seed', type=int, help=r'seed to be used for reproducibility', default=42)
    train_parser_main_group = train_parser.add_mutually_exclusive_group(required=False)
    train_parser_main_group.add_argument(
        r'-j', r'--json', type=str, help=r'hyperparameters to use (json)')
    train_cmd = train_parser_main_group.add_argument_group()
    train_cmd.add_argument(
        r'-c', r'--penalty', type=float, help=r'penalty parameter of the C-SVM')
    train_cmd.add_argument(
        r'-p', r'--epsilon', type=float, help=r'tolerance for stopping criterion', default=1e-3)
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
        r'-w', r'--worker', type=int, help=r'number of worker proc. (data reading)', default=-1)
    predict_parser.add_argument(
        r'-l', r'--offset', type=int, help=r'offset defining the folds for training/evaluation/test splits', default=0)
    # Parse arguments.
    args = arg_parser.parse_args()

    # Execute SVM baseline.
    if args.mode == r'adapt':

        # Adapt data set (compute auxiliary features).
        SVMDataReader.adapt(file_path=Path(args.input), store_path=Path(args.output), kmer_size=args.kmer_size,
                            num_workers=args.worker, dtype=np.float32)

    elif args.mode == r'analyse':

        # Analyse data set with respect to sequence counts.
        SVMDataReader.analyse(
            file_path=Path(args.input), store_path=Path(args.output), kmer_size=args.kmer_size, num_workers=args.worker)

    elif args.mode == r'optim':

        # Create and optimise SVM baseline.
        svm_baseline = SVMBaseline(file_path=Path(args.input), kernel=args.kernel,
                                   fold_info=args.folds if (args.pickle is None) else Path(args.pickle),
                                   load_metadata=True, dtype=np.float32, test_mode=False, offset=args.offset)
        hyperparameters = svm_baseline.optimise(trials=args.trials, penalty=args.penalty,
                                                epsilon=args.epsilon, seed=args.seed,
                                                log_dir=None if args.log_dir is None else Path(args.log_dir))

        # Store best hyperparameters as obtained by grid search.
        output_directory = os.path.dirname(args.output)
        if (len(output_directory) > 0) and (not os.path.exists(output_directory)):
            os.makedirs(output_directory)
        with open(args.output, r'w') as hyperparameters_json:
            json.dump(hyperparameters, hyperparameters_json)

    elif args.mode == r'train':

        # Process data file according to SVM baseline.
        svm_baseline = SVMBaseline(
            file_path=Path(args.input), kernel=args.kernel, fold_info=None,
            load_metadata=True, dtype=np.float32, test_mode=False)

        # Fetch hyperparameters to be used.
        if args.json is not None:
            with open(args.json, r'r') as hyperparameters_json:
                hyperparameters = json.load(hyperparameters_json)
        else:
            hyperparameters = {r'penalty': args.penalty, r'epsilon': args.epsilon}

        # Train SVM baseline.
        svm_baseline.train(
            file_path_output=Path(args.output), penalty=hyperparameters[r'penalty'],
            epsilon=hyperparameters[r'epsilon'], seed=args.seed)

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

        # Fetch auxiliary information from trained SVM baseline model.
        svm_module = load(filename=args.model)
        kernel_type = SVMDataReader.Kernel[svm_module.__dict__[r'kernel_type'].strip().upper()]
        del svm_module

        # Predict using pre-trained SVM baseline model.
        svm_baseline = SVMBaseline(
            file_path=Path(args.input), kernel=kernel_type,
            fold_info=None if (args.pickle is None) else Path(args.pickle),
            load_metadata=True, dtype=np.float32, test_mode=True, offset=args.offset)
        result = svm_baseline.predict_from_path(
            file_path_model=Path(args.model), activations=args.activations, num_workers=args.worker)

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
