## Dataset formats
This code supports hdf5 containers and text based (`.tsv` or `.csv`) datasets,
which will be automatically converted to hdf5 containers.

The pre-defined datasets from our paper are provided via convenience functions in 
`deeprc_binary/predefined_datasets.py`.

A convenience function for reading custom datasets is provided via function `make_dataloaders()` in 
`deeprc_binary/dataset_readers.py`.

### Text-based dataset format
This code expects the following structure of dataset:

- All files should be tab-, comma-, or semicolon-separated files with `.tsv` or `.csv` file name extension.
Default: tab-separated `.tsv` files.
Names of columns can be set in function ---.
- 1 file `metadata.tsv` that contains
  - 1 column holding the repertoire names (Default column name: `ID`)
  - 1 column holding the the labels (Default column name: `Status`)
- n repertoire files representing the n repertoires in the dataset
  - Filenames must be equal to the repertoire names in the `metadata.tsv` file
  - Files must be located in the same directory or subdirectories of `metadata.tsv` file
  - Each repertoire file must include
    - 1 column holding the amino acid sequences (as string) (Default column name: `amino_acid`)
    - 1 column holding the count of each sequence in the repertoire (Default column name: `templates`)

Dataset example using default column names is given in folder `datasets/example_dataset_format`.

Example `metadata.tsv`:
```text
ID	status
HIP00110	-
HIP00169	-
Keck0035	+
```
Example `HIP00110.tsv`:
```text
amino_acid	templates
CASSLGPNTEAFF	6
CASSYEGDSSYEQYF	1839
CASSGGQGSFSGANVLTF	756
```
