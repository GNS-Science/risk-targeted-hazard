## Docker Development Environment

This a docker image that runs jupyter notebook with the dependencies needed by the library installed.

When running it mounts the main `risk-targeted-hazard` directory so you can edit files in your preferred editor and see changes immediately.

### Getting Started

#### Configuration

Make a copy of the `.env.example` file and name it `.env`.

Set `DOCKER_AWS_CREDENTIALS_DIR` to the directory containing your AWS credentials.

You can also configure the environment variables for [Toshi Hazard Store](https://github.com/GNS-Science/toshi-hazard-store) such as `NZSHM22_HAZARD_STORE_STAGE` and `NZSHM22_HAZARD_STORE_REGION`.

On Linux and Mac you should set `DOCKER_USER_UID` to your user id. On Windows this has no effect so you can leave the default value.

#### Windows Instructions

To run the developer environment you need to install and run [Docker Desktop](https://www.docker.com/products/docker-desktop/).

To build the docker image:

`docker_build.bat`

To run the image:

`docker_start.bat`

To create a `bash` shell inside the running docker container:

`docker_shell.bat`

#### Linix / Mac Instructions

To run the developer environment you will need `make` and docker with the [docker compose plugin](https://docs.docker.com/compose/install/linux/).

##### Makefile commands

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
