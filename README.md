# Supply Chain Model

This model allows to build any supply chain tree. Example code can be found in a notebook in the repository.

## Installation
Running this code will require a python installation along with conda. To start, import this repository on your machine. Navigate to the main folder and use the line below to create an environment called supply_chain_model:
```
conda env create -f environment.yml
```

If you already have the environment, you can use update and --prune instead:
```
conda env update -f environment.yml --prune
```

Finally, while still in the same folder, activate the environment and install the package supply_chain_model:
```
conda activate supply_chain_model
pip install -e .
```

If you want VSCode and pylance to recognise the functions, either update ```python.analysis.extraPaths``` in VSCode's settings.json or install using:
```
pip install -e . --config-settings editable_mode=compat --no-cache-dir
```

That's it! The package will now work with the latest functions that are pulled from Github without having to run the pip install again.
