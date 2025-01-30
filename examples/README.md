# [`ml4gw`](https://github.com/ML4GW/ml4gw) Tutorial

This repo hosts the Jupyter notebook and the environment files for the `ml4gw` tutorial.

If you use `poetry`, the environment can be installed and a Jupyter kernel created with
```bash
poetry install
poetry run python -m ipykernel install --user --name ml4gw_tutorial
```

If you use a different environment manager, all of the packages listed in the `pyproject.toml` file can be `pip install`ed into whatever environment you desire.

The background data files used in this tutorial can be copied from `/home/william.benoit/ML4GW/ml4gw_tutorial/background_data` on the Hanford computing cluster.
