# Installation

The logistic MIL baseline can be installed via `pip/pip3`:

```bash
$ pip3 install git+https://github.com/ml-jku/DeepRC/tree/master/compared_methods/logistic_mil
```

# Basic usage

The logistic MIL baseline has four modes of operation:

- `predict` for predicting disease status using an already fitted logistic MIL baseline model
- `adapt` for adapting a raw data set to be processable by the logistic MIL baseline
- `optim` for performing hyperparameter optimization using exhaustive grid search
- `train` for fitting a new logistic MIL baseline to a data set

More information regarding the arguments is accessible via the `-h` flag.

```bash
$ logisticirc -h
```

```bash
$ logisticirc MODE -h
```

Or, alternatively, by specifying `--help`.

```bash
$ logisticirc --help
```

```bash
$ logisticirc MODE --help
```

# Predicting the disease status

Prediction of the disease status of all repertoires in a data set `data_set.h5py` using the model `model.pth` can be
achieved by:

```bash
$ logisticirc predict --input data_set.h5py --model model.pth
```

The data set specified via `--input` is temporarily adapted on the fly and the resulting predictions of the applied
logistic MIL model are printed to the standard output. The predictions (one for each repertoire) can be interpreted as:
- `0` &rarr; negative disease status (no disease predicted)
- `1` &rarr; positive disease status

```bash
[0, 1, 1, 1, ..., 1, 1, 0, 1, 1]
```

A comprehensive listing of the `predict` mode of operation is accessible via the `-h` flag.

```bash
usage: logisticirc predict [-h] -i INPUT [-o OUTPUT_DIR] [-a] -m MODEL [-z PICKLE]
                           [-k WORKER] [-d DEVICE] [-l OFFSET]

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
  -k WORKER, --worker WORKER
                        number of worker processes (data reading)
  -d DEVICE, --device DEVICE
                        device to use for heavy computations
  -l OFFSET, --offset OFFSET
                        offset defining the folds for training/evaluation/test splits
```

# Adapting a raw data set

Adaption of a raw data set `raw_data_set.h5py` with a k-mer size of `4` storing to `data_set.h5py` can be achieved by:

```bash
$ logisticirc adapt --input raw_data_set.h5py --output data_set.h5py --kmer_size 4
```

A comprehensive listing of the `adapt` mode of operation is accessible via the `-h` flag.

```bash
usage: logisticirc adapt [-h] -i INPUT -o OUTPUT [-z KMER_SIZE] [-k WORKER]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data file (h5py) to use
  -o OUTPUT, --output OUTPUT
                        path to resulting data file (h5py)
  -z KMER_SIZE, --kmer_size KMER_SIZE
                        size <k> of a k-mer
  -k WORKER, --worker WORKER
                        number of worker processes (data reading)
```

# Selecting an appropriate model

Model selection on the basis of a `5`-fold cross-validation using a data set `data_set.h5py` and the `kmer`
abundance, applying `4` grid search trials for `12` epochs each with respect to the learning rates `0.1` and `0.01` as
well as the batch sizes `1` and `32`, and storing the hyperparameter values `settings.json` of the final model,
can be achieved by:

```bash
$ logisticirc optim --input data_set.h5py --output settings.json --folds 5 --abundance "kmer" --batch_size 1,32 --learning_rate 0.1,0.01 --epochs 12 
```

A comprehensive listing of the `optim` mode of operation is accessible via the `-h` flag.

```bash
usage: logisticirc optim [-h] -i INPUT -o OUTPUT [-g LOG_DIR] [-n] [-a] -c
                         ABUNDANCE [-f FOLDS] [-z PICKLE] [--offset OFFSET]
                         [-e EPOCHS] -b BATCH_SIZE [-q TOP_N] [-r] -l
                         LEARNING_RATE [-x BETA_ONE] [-y BETA_TWO]
                         [-w WEIGHT_DECAY] [-v] [-p EPSILON] [-s SEED] [-k WORKER]
                         [-d DEVICE] [-u]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data file (h5py) to use
  -o OUTPUT, --output OUTPUT
                        path to store best hyperparameters
  -g LOG_DIR, --log_dir LOG_DIR
                        directory to store TensorBoard logs
  -n, --normalise       normalise features
  -a, --normalise_abundance
                        normalise relative abundance term
  -c ABUNDANCE, --abundance ABUNDANCE
                        type of abundance
  -f FOLDS, --folds FOLDS
                        number of folds
  -e EPOCHS, --epochs EPOCHS
                        maximum number of epochs to optimise
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        list of batch sizes
  -q TOP_N, --top_n TOP_N
                        tuple of top <n> entities considered per sample
  -r, --randomise       randomise batches between epochs
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        tuple of learning rates (Adam)
  -x BETA_ONE, --beta_one BETA_ONE
                        tuple of beta 1 (Adam)
  -y BETA_TWO, --beta_two BETA_TWO
                        tuple of beta 2 (Adam)
  -w WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        tuple of weight decay terms (Adam)
  -v, --amsgrad         use AMSGrad version of Adam
  -p EPSILON, --epsilon EPSILON
                        epsilon to use for numerical stability
  -s SEED, --seed SEED  seed to be used for reproducibility
  -k WORKER, --worker WORKER
                        number of worker processes (data reading)
  -d DEVICE, --device DEVICE
                        device to use for heavy computations
  -u, --debug           use debugging module
```

# Fitting a logistic MIL model

Fitting of a model `model.pth` for `12` epochs using a data set `data_set.h5py`, a batch size of `16` and the `kmer`
abundance, can be achieved by:

```bash
$ logisticirc train --input data_set.h5py --output model.pth --abundance "kmer" --epochs 12 --batch_size 16
```

Alternatively, one can supply the resulting settings file created by `--output` of the `optim` mode to fit a
logistic MIL model, instead of manually specifying the values of the hyperparameters.

```bash
$ logisticirc train --input data_set.h5py --output model.pth --kernel "min_max" --json settings.json
```

A comprehensive listing of the `train` mode of operation is accessible via the `-h` flag.

```bash
usage: logisticirc train [-h] -i INPUT -o OUTPUT [-n] [-a] -c ABUNDANCE [-r]
                         [-s SEED] [-k WORKER] [-d DEVICE] [-j JSON] [-e EPOCHS]
                         [-b BATCH_SIZE] [-q TOP_N] [-l LEARNING_RATE]
                         [-x BETA_ONE] [-y BETA_TWO] [-w WEIGHT_DECAY] [-v]
                         [-p EPSILON]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        data set (h5py) to use
  -o OUTPUT, --output OUTPUT
                        path to store resulting model
  -n, --normalise       normalise features
  -a, --normalise_abundance
                        normalise relative abundance term
  -c ABUNDANCE, --abundance ABUNDANCE
                        type of abundance
  -r, --randomise       randomise batches between epochs
  -s SEED, --seed SEED  seed to be used for reproducibility
  -k WORKER, --worker WORKER
                        number of worker processes (data reading)
  -d DEVICE, --device DEVICE
                        device to use for heavy computations
  -j JSON, --json JSON  hyperparameters to use (json)
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

- Ostmeyer, J., Christley, S., Toby, I. T., and Cowell, L. G. Biophysicochemical motifs in t-cell receptor sequences
distinguish repertoires from tumor-infiltrating lymphocyte and adjacent healthy tissue.Cancer research,
79(7):1671–1680, 2019.