# Installation

The k-nearest neighbour (KNN) baseline can be installed via `pip/pip3`:

```bash
$ pip3 install git+https://github.com/ml-jku/DeepRC/tree/master/compared_methods/knn
```

# Basic usage

The k-nearest neighbour (KNN) baseline has four modes of operation:

- `predict` for predicting disease status using an already fitted KNN baseline model
- `adapt` for adapting a raw data set to be processable by the KNN baseline
- `optim` for performing hyperparameter optimization using exhaustive grid search
- `train` for fitting a new KNN baseline to a data set

More information regarding the arguments is accessible via the `-h` flag.

```bash
$ knnirc -h
```

```bash
$ knnirc MODE -h
```

Or, alternatively, by specifying `--help`.

```bash
$ knnirc --help
```

```bash
$ knnirc MODE --help
```

# Predicting the disease status

Prediction of the disease status of all repertoires in a data set `data_set.h5py` using the model `model.knn` can be
achieved by:

```bash
$ knnirc predict --input data_set.h5py --model model.knn
```

The data set specified via `--input` is temporarily adapted on the fly and the resulting predictions of the applied KNN model are
printed to the standard output. The predictions (one for each repertoire) can be interpreted as:
- `0` &rarr; negative disease status (no disease predicted)
- `1` &rarr; positive disease status

```bash
[0, 1, 1, 1, ..., 1, 1, 0, 1, 1]
```

A comprehensive listing of the `predict` mode of operation is accessible via the `-h` flag.

```bash
usage: knnirc predict [-h] -i INPUT [-o OUTPUT_DIR] [-a] -m MODEL [-z PICKLE]
                      [-w WORKER] [-l OFFSET]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data set (h5py) to use
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        directory to store predictions (and ROC AUC)
  -a, --activations     compute activations instead of discrete predictions
  -m MODEL, --model MODEL
                        model to be used for prediction
  -z PICKLE, --pickle PICKLE
                        fold definitions (pickle-file)
  -w WORKER, --worker WORKER
                        number of worker proc. (data reading)
  -l OFFSET, --offset OFFSET
                        offset defining the folds for training/evaluation/test splits
```

# Adapting a raw data set

Adaption of a raw data set `raw_data_set.h5py` with a k-mer size of `4` storing to `data_set.h5py` can be achieved by:

```bash
$ knnirc adapt --input raw_data_set.h5py --output data_set.h5py --kmer_size 4
```

A comprehensive listing of the `adapt` mode of operation is accessible via the `-h` flag.

```bash
usage: knnirc adapt [-h] -i INPUT -o OUTPUT [-z KMER_SIZE] [-w WORKER]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data file (h5py) to use
  -o OUTPUT, --output OUTPUT
                        path to resulting data file (h5py)
  -z KMER_SIZE, --kmer_size KMER_SIZE
                        size <k> of a k-mer
  -w WORKER, --worker WORKER
                        number of worker proc. (data reading)
```

# Selecting an appropriate model

Model selection on the basis of a `5`-fold cross-validation using a data set `data_set.h5py` and the `min_max`
kernel, applying exhaustive grid search with respect to the amount of neighbours in the range of `[1;10]` and storing
the hyperparameter values `settings.json` of the final model, can be achieved by:

```bash
$ knnirc optim --input data_set.h5py --output settings.json --kernel "min_max" --folds 5 --neighbours 1 10
```

A comprehensive listing of the `optim` mode of operation is accessible via the `-h` flag.

```bash
usage: knnirc optim [-h] -i INPUT -o OUTPUT [-g LOG_DIR] -k KERNEL [-f FOLDS]
                    [-z PICKLE] [-l OFFSET] -n NEIGHBOURS NEIGHBOURS
                    [-s SEED]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data file (h5py) to use
  -o OUTPUT, --output OUTPUT
                        path to store best hyperparameters
  -g LOG_DIR, --log_dir LOG_DIR
                        directory to store TensorBoard logs
  -k KERNEL, --kernel KERNEL
                        type of kernel
  -f FOLDS, --folds FOLDS
                        number of folds
  -n NEIGHBOURS NEIGHBOURS, --neighbours NEIGHBOURS NEIGHBOURS
                        range of neighbours parameter of the KNN
  -s SEED, --seed SEED  seed to be used for reproducibility
```

# Fitting a KNN model

Fitting of a model `model.knn` using a data set `data_set.h5py` and the `min_max` kernel, considering the `1` nearest
neighbour, can be achieved by:

```bash
$ knnirc train --input data_set.h5py --output model.knn --kernel "min_max" --neighbours 1
```

Alternatively, one can supply the resulting settings file created by `--output` of the `optim` mode to fit a KNN model,
instead of manually specifying the values of the hyperparameters.

```bash
$ knnirc train --input data_set.h5py --output model.knn --kernel "min_max" --json settings.json
```

A comprehensive listing of the `train` mode of operation is accessible via the `-h` flag.

```bash
usage: knnirc train [-h] -i INPUT -o OUTPUT -k KERNEL [-s SEED] (-j JSON | -n NEIGHBOURS)

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data set (h5py) to use
  -o OUTPUT, --output OUTPUT
                        path to store resulting model
  -k KERNEL, --kernel KERNEL
                        type of kernel
  -s SEED, --seed SEED  seed to be used for reproducibility
  -j JSON, --json JSON  hyperparameters to use (json)
  -n NEIGHBOURS, --neighbours NEIGHBOURS
                        neighbours parameter of the KNN
```

# Pickle format

When [selecting an appropriate model](#selecting-an-appropriate-model), two different approaches are possible:

- `--folds` defines the amount of folds of the cross-validation
- `--pickle` specifies an additional file defining training, validation and testing subsets, controlled by `--offset`

In case of the second approach, the corresponding file specified via `--pickle` needs to be a pickled Python dictionary
`dict` with at least the following field:

- `'inds': List[np.ndarray]` &rarr; one entry for each fold, specifying the indices of the respective samples

Exemplarily, the `'inds'` field may look like the following:

```bash
[
    array([551,  26, 206, ..., 401, 421, 287]),
    array([765, 486, 133, ..., 513, 404, 645]),
    array([209, 484, 371, ...,  16, 781,  59]),
    array([229, 378, 143, ..., 233, 759, 738]),
    array([727, 577, 648, ...,  81, 258,  41])
]
```

The accompanying `--offset` argument controls the assignment of the `'inds'` entries to the respective training,
validation and testing subsets according to the following rules:

- `validation subset` &rarr; `--offset`
- `testing subset` &rarr; `validation subset + 1`
- `training subset` &rarr; all remaining entries

# References (excerpt)

- Ralaivola, L., Swamidass, S. J., Saigo, H., and Baldi, P. Graph kernels for chemical informatics.
Neural networks,18(8):1093–1110, 2005.

- Levandowsky, M. and Winter, D.  Distance between sets. Nature, 234(5323):34–35, 1971
