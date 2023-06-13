FROM ubuntu:22.04 as base

WORKDIR /tmp

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -qyy -o APT::Install-Recommends=false -o APT::Install-Suggests=false \
    python3 \
    curl \
    ca-certificates \
    less

WORKDIR /src

ARG DOCKER_USER_UID=9999
RUN useradd --system --uid "$DOCKER_USER_UID" --shell /bin/bash --create-home user

USER user

# Install poetry
ENV POETRY_HOME=/home/user/.poetry
ARG POETRY_VERSION=1.5.1

RUN mkdir "$POETRY_HOME"
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 - --version "$POETRY_VERSION"

ENV PATH="${POETRY_HOME}/bin/:${PATH}"

ENV SHELL=bash

# Set the default command to start Jupyter
ENTRYPOINT ["poetry", "run", "jupyter-notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
