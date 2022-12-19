FROM ubuntu:22.04 as dev

WORKDIR /tmp

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -qyy -o APT::Install-Recommends=false -o APT::Install-Suggests=false \
    python3 \
    curl \
    ca-certificates \
    less

WORKDIR /src

# HACK: using my uid
ARG DOCKER_USER_UID=10006
RUN useradd --system --uid "$DOCKER_USER_UID" --shell /bin/bash --create-home user

USER user

# Install poetry
ENV POETRY_HOME=/home/user/.poetry
ARG POETRY_VERSION=1.3.0

RUN mkdir "$POETRY_HOME"
RUN curl -sSL https://install.python-poetry.org | python3 - --version "$POETRY_VERSION"

ENV PATH="${POETRY_HOME}/bin/:/src/.local/bin/:${PATH}"

# TODO: set this in an .env file the user configures themselves?
ENV HOME=/src
ENV NZSHM22_HAZARD_STORE_STAGE=PROD
ENV NZSHM22_HAZARD_STORE_REGION=ap-southeast-2

# TODO: possibly set AWS_SHARED_CREDENTIALS_FILE location?
# https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html

# Set the default command to start Jupyter
ENTRYPOINT ["poetry", "run", "jupyter-notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]