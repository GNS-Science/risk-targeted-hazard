# risk-targeted-hazard
Python library for generating risk-targeted hazard seismic design intensities

Based on the framework in:

Horspool, N., Hulsey, A., Elwood., K., and Gerstenberger, M. 2021. Risk-targeted hazard spectra for seismic design in New Zealand. Proceedings of the New Zealand Society of Earthquake Engineering Conference, Christchurch, April 2021. http://13.237.132.70/handle/nzsee/2324 

### Getting Started

1. Create and activate a new conda environment:
    ```
    conda create -n risk python=3.10
    conda activate risk
    ```

2. Install the library dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Optional: Configure environment variables for [Toshi Hazard Store](https://github.com/GNS-Science/toshi-hazard-store)

### Development

Install the development dependencies:

```
pip install -r requirements-dev.txt
```

Run the tests:

```
pytest test
```

Run jupyter notebook:

```
jupyter notebook
```

### Docker

You can also run the development environment using docker. See the [docker README](docker/README.md).
