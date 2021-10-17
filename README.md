# Modern Hopfield Networks and Attention for Immune Repertoire Classification

Michael Widrich<sup>1</sup>,
Bernhard Schäfl<sup>1</sup>, 
Milena Pavlović<sup>3 4</sup>,
Hubert Ramsauer<sup>1</sup>,
Lukas Gruber<sup>1</sup>,
Markus Holzleitner<sup>1</sup>,
Johannes Brandstetter<sup>1</sup>,
Geir Kjetil Sandve<sup>4</sup>, 
Victor Greiff<sup>3</sup>, 
Sepp Hochreiter<sup>1 2</sup>,
Günter Klambauer<sup>1</sup>

(1) ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria\
(2) Institute of Advanced Research in Artificial Intelligence (IARAI)\
(3) Department of Immunology, University of Oslo, Oslo, Norway\
(4) Department of Informatics, University of Oslo, Oslo, Norway

- Paper: https://arxiv.org/abs/2007.13505
- Poster: [neurips_poster.pdf](neurips_poster.pdf)

**This package provides:**
- modular and customizable DeepRC implementation for massive multiple instance learning problems, such as immune repertoire classification,
- CNN and LSTM sequence embedding,
- single- or multi-task settings (simple building-block principle),
- support for custom datasets,
- examples that you can quickly adapt to your problem settings.

**Will be added:**
- multiple attention heads/queries,
- Integrated Gradients analysis (write me an email (widrich at ml.jku.at) if you urgently need a preliminary version).

## Installation
### pip
You can install this package via pip:
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

## Usage
To run the examples,
download the github repo as .zip file, extract the .zip file,
and navigate into the extracted directory (you should see a `deeprc` folder and the `README.md` there).

Can't wait? Examples are here: ```deeprc/examples/```.

### Training DeepRC on pre-defined datasets
You can train a DeepRC model on the pre-defined datasets of the DeepRC paper 
using one of the Python files in folder `deeprc/examples/examples_from_paper`.
The datasets will be downloaded automatically (please only download them once and then reuse the downloaded versions).

You can use `tensorboard --logdir [results_directory] --port=6060` and 
open `http://localhost:6060/` in your web-browser to view the performance.

##### Real-world data with implanted signals
This is category has the smallest dataset files and is a **good starting point**.
Training a binary DeepRC classifier on dataset "0" of category "real-world data with implanted signals":
```bash
python3 -m deeprc.examples.examples_from_paper.cmv_with_implanted_signals 0 --n_updates 10000 --evaluate_at 2000
```

To get more information, you can use the help function:
```bash
python3 -m deeprc.examples.examples_from_paper.cmv_with_implanted_signals -h
```

##### LSTM-generated data
Training a binary DeepRC classifier on dataset "0" of category "LSTM-generated data":
```bash
python3 -m deeprc.examples.examples_from_paper.lstm_generated 0
```

##### Simulated immunosequencing data
Training a binary DeepRC classifier on dataset "0" of category "simulated immunosequencing data":
```bash
python3 -m deeprc.examples.examples_from_paper.simulated 0
```
Warning: Filesize to download is ~20GB per dataset!

##### Real-world data
Training a binary DeepRC classifier on dataset "real-world data":
```bash
python3 -m deeprc.examples.examples_from_paper.cmv
```

### Training DeepRC on a custom dataset
You can train DeepRC on custom text-based datasets,
such as the small example dataset `deeprc/datasets/example_dataset`.
Specifications of the supported dataset formats are give here: `deeprc/datasets/README.md`.

You can change the dataset directory and task description in the examples listed below and start training a DeepRC model on your task:

##### Training a binary DeepRC classifier on a small random example dataset using 1D CNN sequence embedding:
```bash
python3 -m deeprc.examples.example_single_task_cnn.py
```

##### Training DeepRC in a multi-task setting on a small random example dataset using 1D CNN sequence embedding:
```bash
python3 -m deeprc.examples.example_multitask_cnn.py
```

##### Training DeepRC in a multi-task setting on a small random example dataset using LSTM sequence embedding:
```bash
python3 -m deeprc.examples.example_multitask_lstm.py
```

## Datasets
The datasets will be automatically downloaded when running the examples from section "Training DeepRC on pre-defined datasets".
You can also manually download the datasets here: https://ml.jku.at/research/DeepRC/datasets/.
Please see our paper for descriptions of the datasets.

## Structure
```text
deeprc
      |--datasets : stores datasets
      |   |--example_dataset : Small example dataset
      |   |--README.md : Information on supported dataset formats
      |   |--splits_used_in_paper : Dataset splits as used in paper
      |--deeprc : DeepRC implementation
      |   |--architectures.py : DeepRC network architecture
      |   |--dataset_converters.py : Converter for text-based datasets
      |   |--dataset_readers.py : Tools for reading datasets
      |   |--predefined_datasets.py : Pre-defined datasets from paper
      |   |--task_definitions.py : Tools for defining the task to train DeepRC on
      |   |--training.py : Tools for training DeepRC model
      |--examples : DeepRC examples
      |   |--examples_from_paper : Examples on datasets used in paper
      |--neurips_poster.pdf : Poster from NeurIPS2020 poster session
```

## Note
I'm currently cleaning up and uploading the code for the paper.
There might be (and probably are) some bugs which will be fixed soon.
If you need help with running DeepRC in the meantime,
feel free to write me an email (widrich at ml.jku.at).

Best wishes,

Michael

## Requirements
I relaxed the package versions to untested versions now.
Please see the list below for the tested package versions and let me know if some higher package version fails.
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
   - [tensorboard](https://www.tensorflow.org/tensorboard) (tested with version 1.15.0)
   - [widis-lstm-tools](https://github.com/widmi/widis-lstm-tools) (tested with version 0.4)
