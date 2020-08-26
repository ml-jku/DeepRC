import setuptools

with open(r'README.md', mode=r'r') as readme_handle:
    long_description = readme_handle.read()

setuptools.setup(
    name=r'svmirc',
    version=r'1.0.0',
    author=r'Bernhard Schäfl',
    author_email=r'schaefl@ml.jku.at',
    url=r'https://github.com/ml-jku/DeepRC/tree/master/compared_methods/svm',
    description=r'SVM model for immune repertoire classification',
    long_description=long_description,
    long_description_content_type=r'text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        r'torch>=1.3.1',
        r'numpy>=1.18.2',
        r'h5py>=2.9.0',
        r'tqdm>=0.24.2',
        r'tensorboard>=1.14.0',
        r'scikit-learn>=0.22.1',
        r'joblib>=0.16.0'
    ],
    zip_safe=True,
    entry_points={
        r'console_scripts': [
            r'svmirc = svmirc.interactive:console_entry'
        ]
    }
)
