## Getting started

Create a new conda environment and install dependencies:

```
conda create -n risk python=3.10
conda activate risk
pip install -r requirements.txt
```

Optional: Configure environment variables for [Toshi Hazard Store](https://github.com/GNS-Science/toshi-hazard-store)

## Other commands

Run the tests:

```
pytest test
```

Run jupyter notebook:

```
jupyter notebook
```

## Docker

You can also run the development environment using docker. See the [docker README](docker/README.md).
