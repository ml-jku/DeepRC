## Dataset formats
This code supports hdf5 containers and text based (`.tsv` or `.csv`) datasets,
which will be automatically converted to hdf5 containers.

The pre-defined datasets from our paper are provided via convenience functions in 
`deeprc/predefined_datasets.py`.

A convenience function for reading custom datasets is provided via function `make_dataloaders()` in 
`deeprc/dataset_readers.py`.
This function will also automatically convert your textbased dataset to a hdf5 file.
See `deeprc/examples/example_single_task_cnn.py` for an example.

### Text-based dataset format
This code expects the following structure of dataset:

- All files should be tab-, comma-, or semicolon-separated files with `.tsv` or `.csv` file name extension.
Default: tab-separated `.tsv` files.
- 1 file `metadata.tsv` that contains
  - 1 column holding the repertoire names/IDs
  - 1+ columns holding the target values/labels of the repertoires
- n repertoire files representing the n repertoires in the dataset
  - Filenames must be equal to the repertoire names in the `metadata.tsv` file
  - All repertoire files must be located in the same directory or subdirectories.
  - Each repertoire file must include
    - 1 column holding the amino acid sequences (as string) (Default column name: `amino_acid`)
    - 1 column holding the number of occurrences (=`count`) of each sequence in the repertoire (Default column name: `templates`)

A dataset example is given in folder `deeprc/datasets/example_dataset`, with example usages shown in
in `deeprc/examples`, e.g. `deeprc/examples/example_single_task_cnn.py`.

Example metadata file `metadata.tsv`:
```text
ID	status
HIP00110	-
HIP00169	-
Keck0035	+
```
Example repertoire file `HIP00110.tsv`:
```text
amino_acid	templates
CASSLGPNTEAFF	6
CASSYEGDSSYEQYF	1839
CASSGGQGSFSGANVLTF	756
```
Example folder structure:
```text
some_folder/
      |--metadata.tsv : The metadata file
      |--repertoires/ : Folder containing the repertoire files
      |   |--HIP00110.tsv
      |   |--HIP00111.tsv
      |   |--HIP00112.tsv
      |   |--HIP00115.tsv
      |   |--HIP00118.tsv
      |   |--HIP00119.tsv
```