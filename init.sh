#!/usr/bin/env bash

# On hiccup CPU, we use pyenv -- on hiccup GPU, we will use system python
if ! lspci | grep -i 'nvidia' > /dev/null; then
    # Set up pyenv (for python version management)
    export PYENV_ROOT="/home/software/users/james/pyenv"
    export PYTHON_CONFIGURE_OPTS="--enable-shared"
    export PATH="${PATH}:${PYENV_ROOT}/bin"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    pyenv local 3.9.12
fi

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

# Install separate venv for GPU and CPU (to allow to work on either machine, using same home directory)
if [ ! -z ${INSTALL} ]; then
    if lspci | grep -i 'nvidia' > /dev/null; then
        VENV_DIR=".venv_gpu"
    else
        VENV_DIR=".venv_cpu"
    fi

    if [ -d $VENV_DIR ]; then
        echo
        echo "Remove existing virtual environment..."
        rm -r $VENV_DIR
    fi
    echo
    echo "Create new virtual environment..."
    python -m venv $VENV_DIR
    pdm install
fi

# Initialize python virtual environment for package management
source $VENV_DIR/bin/activate

