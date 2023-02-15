# risk-targeted-hazard
Python library for generating risk-targeted hazard seismic design intensities

Based on the framework in:

Horspool, N., Hulsey, A., Elwood., K., and Gerstenberger, M. 2021. Risk-targeted hazard spectra for seismic design in New Zealand. Proceedings of the New Zealand Society of Earthquake Engineering Conference, Christchurch, April 2021. http://13.237.132.70/handle/nzsee/2324 

### Getting Started

1. Create and activate a new Python virtual env:
    ```
    virtualenv rth_venv
    rth_venv\Scripts\activate.bat
    ```
    *NOTE*: On Linux / Mac activate the virtual env with `source rth_venv/bin/activate`

2. Install the library dependencies:
    ```
    pip3 install -r requirements.txt
    ```

### Development

Install the development dependencies:

```
pip3 install -r requirements-dev.txt
```

### Docker

You can also run the development environment using docker. See the [docker README](docker/README.md).
