# -*- coding: utf-8 -*-
from setuptools import find_packages, setup
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
requirements = {"install": ["torch>=1.3.1", "numpy>=1.18.2", "h5py>=2.9.0", "dill>=0.3.0", "pandas>=0.24.2",
                            "tqdm>=0.24.2", "tensorboard==1.15.0", "scikit-learn>=0.22.2.post1",
                            "requests>=2.21.0"]}
install_requires = requirements["install"]

setup(
        # Metadata
        name="deeprc",
        version=get_version("deeprc/__init__.py"),
        author="Michael Widrich",
        author_email="widrich@ml.jku.at",
        url="https://github.com/ml-jku/DeepRC",
        description=(
            "DeepRC: Immune repertoire classification with attention-based deep massive multiple instance learning"
        ),
        long_description=readme,
        long_description_content_type="text/markdown",
        # Package info
        packages=find_packages(),
        zip_safe=True,
        install_requires=install_requires,
        include_package_data=True,
)
