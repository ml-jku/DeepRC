# DeepRC: Immune repertoire classification with attention-based deep massive multiple instance learning

Michael Widrich<sup>1 2</sup>, 
Bernhard Schäfl<sup>1 2</sup>, 
Milena Pavlović<sup>3 4</sup>, 
Geir Kjetil Sandve<sup>4</sup>, 
Sepp Hochreiter<sup>2 1 5</sup>, 
Victor Greiff<sup>3</sup>, 
Günter Klambauer<sup>2 1</sup>

(1) Institute for Machine Learning, Johannes Kepler University Linz, Austria\
(2) LIT AI Lab, Johannes Kepler University Linz, Austria\
(3) Department of Immunology, University of Oslo, Oslo, Norway\
(4) Department of Informatics, University of Oslo, Oslo, Norway\
(5) Institute of Advanced Research in Artificial Intelligence (IARAI)

Paper: https://www.biorxiv.org/content/10.1101/2020.04.12.038158v2

## Quickstart

You can directly install the package from GitHub using the commands below:

```bash
pip install --no-dependencies git+https://github.com/widmi/widis-lstm-tools
pip install git+https://github.com/ml-jku/DeepRC
```

To update your installation with dependencies, you can use:

```bash
pip install --no-dependencies git+https://github.com/widmi/widis-lstm-tools
pip install --upgrade git+https://github.com/ml-jku/DeepRC
```

To update your installation without dependencies, you can use:

```bash
pip install --no-dependencies git+https://github.com/widmi/widis-lstm-tools
pip install --no-dependencies --upgrade git+https://github.com/ml-jku/DeepRC
```

Train a binary DeepRC classifier on dataset "0" of category "real-world data with implanted signals":

```bash
python3 -m deeprc.examples.simple_cmv_with_implanted_signals 0 --n_updates 10000 --evaluate_at 2000
```

## Usage
### Training DeepRC on pre-defined datasets
You can train a DeepRC model on the pre-defined datasets of the DeepRC paper 
using one of the Python files in folder `deeprc/examples`.
The datasets will be downloaded automatically.

You can use `tensorboard --logdir [results_directory] --port=6060` and 
open `http://localhost:6060/` in your web-browser to view the performance.

##### Real-world data with implanted signals
This is category has the smallest dataset files and is a good starting point.
Training a binary DeepRC classifier on dataset "0" of category "real-world data with implanted signals":
```bash
python3 -m deeprc.examples.simple_cmv_with_implanted_signals 0 --n_updates 10000 --evaluate_at 2000
```

To get more information, you can use the help function:
```bash
python3 -m deeprc.examples.simple_cmv_with_implanted_signals -h
```

##### LSTM-generated data
Training a binary DeepRC classifier on dataset "0" of category "LSTM-generated data":
```bash
python3 -m deeprc.examples.simple_lstm_generated 0
```

##### Real-world data
Training a binary DeepRC classifier on dataset "real-world data":
```bash
python3 -m deeprc.examples.simple_cmv
```

### Training DeepRC on a custom dataset
You can train DeepRC on custom text-based datasets,
which will be automatically converted to hdf5 containers.
Specifications of the supported formats are give here: `deeprc/datasets/README.md`
```python
from deeprc.deeprc_binary.dataset_readers import make_dataloaders
from deeprc.deeprc_binary.architectures import DeepRC
from deeprc.deeprc_binary.training import train, evaluate

# Let's assume this is your dataset metadata file
metadatafile = 'custom_dataset/metadata.tsv'

# Get data loaders from text-based dataset (see `deeprc/datasets/README.md` for format)
trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders(
    metadatafile, target_label='status', true_class_label_value='+', id_column='ID', 
    single_class_label_columns=('status',), sequence_column='amino_acid',
    sequence_counts_column='templates', column_sep='\t', filename_extension='.tsv')

# Train a DeepRC model
model = DeepRC(n_input_features=23, n_output_features=1, max_seq_len=30)
train(model, trainingset_dataloader=trainingset, trainingset_eval_dataloader=trainingset_eval,
      validationset_eval_dataloader=validationset_eval, results_directory='results')

# Evaluate on test set
roc_auc, bacc, f1, scoring_loss = evaluate(model=model, dataloader=testset_eval)

print(f"Test scores:\nroc_auc: {roc_auc:6.4f}; bacc: {bacc:6.4f}; f1:{f1:6.4f}; scoring_loss: {scoring_loss:6.4f}")
```

Note that `make_dataloaders()` will automatically create a hdf5 container of your dataset.
Later, you can simply load this hdf5 container instead of the text-based dataset:
```python
from deeprc.deeprc_binary.dataset_readers import make_dataloaders
# Get data loaders from hdf5 container
trainingset, trainingset_eval, validationset_eval, testset_eval = make_dataloaders('dataset.hdf5')
```

You can use `tensorboard --logdir [results_directory] --port=6060` and 
open 'http://localhost:6060/' in your web-browser to view the performance.

## Structure
```text
deeprc
      |--datasets : stores datasets
      |   |--README.md : Information on supported dataset formats
      |--deeprc_binary : DeepRC implementation for binary classification
      |   |--architectures.py : DeepRC network architecture
      |   |--dataset_converters.py : Converter for text-based datasets
      |   |--dataset_readers.py : Tools for reading datasets
      |   |--predefined_datasets.py : Pre-defined datasets from paper
      |   |--training.py : Tools for training DeepRC model
      |--examples : DeepRC examples
```

## Note
We are currently cleaning up and uploading the code for the paper.

## Requirements
- [Python3.6.9](https://www.python.org/) or higher
- Python packages:
   - [Pytorch](https://pytorch.org/) (tested with version 1.3.1)
   - [numpy](https://www.numpy.org/) (tested with version 1.18.2)
   - [h5py](https://www.h5py.org/) (tested with version 2.9.0)
   - [dill](https://pypi.org/project/dill/) (tested with version 0.3.0)
   - [pandas](https://pandas.pydata.org/) (tested with version 0.24.2)
   - [tqdm](https://tqdm.github.io/) (tested with version 0.24.2)
   - [scikit-learn](https://scikit-learn.org/) (tested with version 0.22.2.post1)
   - [requests](https://requests.readthedocs.io/en/master/) (tested with version 2.21.0)
   - [tensorboard](https://www.tensorflow.org/tensorboard) (tested with version 1.14.0)
   - [widis-lstm-tools](https://github.com/widmi/widis-lstm-tools) (tested with version 0.4)
