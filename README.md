# risk-targeted-hazard
Python library for generating risk-targeted hazard seismic design intensities

Based on the framework in:

Horspool, N., Hulsey, A., Elwood., K., and Gerstenberger, M. 2021. Risk-targeted hazard spectra for seismic design in New Zealand. Proceedings of the New Zealand Society of Earthquake Engineering Conference, Christchurch, April 2021. http://13.237.132.70/handle/nzsee/2324 

## Developer instructions

The development environment is a docker container that runs jupyter notebook with the library dependencies needed by the library installed.

### Setup

To run the developer environment you will need `make` and docker with the [docker compose plugin](https://docs.docker.com/compose/install/linux/):

TODO: instructions for setting UID

TODO: instructions for toshi_hazard_store

Create the developer environment with:

`make developer`

Start development environment and display the login URL of the jupyter notebook:

`make start`

Stop and remove the container:

`make stop`

Create a shell in the container:

`make shell`

Run tests:

`make test`

Linting:

`make lint`

Update python packages. Set the shell variable `POETRY_PACKAGES` to update a specific package(s):

`make venv-update`

### Visual studio code

If you are using [vscode](https://code.visualstudio.com/) as your editor you will need python 3.10 on your host machine for it to successfully detect and use the `.venv` as your virtual environment.
