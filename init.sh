#!/usr/bin/env bash

# Set up pyenv (for python version management)
export PYENV_ROOT="/home/software/users/james/pyenv"
export PYTHON_CONFIGURE_OPTS="--enable-shared"
export PATH="${PATH}:${PYENV_ROOT}/bin"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv local 3.9.12

# Get command line option to determine whether we need to install the virtual environment, or just enter it
for i in "$@"; do
  case $i in
    --install)
      INSTALL=TRUE
      shift
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done
if [ ! -z ${INSTALL} ]; then
    echo
    echo "Remove existing virtual environment..."
    rm -r .venv
    echo
    echo "Create new virtual environment..."
    pdm install
fi

# Initialize python virtual environment for package management
source .venv/bin/activate
