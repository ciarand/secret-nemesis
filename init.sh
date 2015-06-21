#!/usr/bin/env bash

# stop immediately if anything fails
set -eo pipefail

main() {
    echo "Checking preconditions..."
    python --version 2&>1 || (echo "You need python in your \$PATH" && false)
    pip --version 2&>1 || (echo "You need pip in your \$PATH" && false)
    virtualenv --version 2&>1 || pip install virtualenv

    echo "Setting up virtualenv..."
    virtualenv venv

    echo "Installing deap..."
    venv/bin/pip install deap

    echo "Installing numpy..."
    venv/bin/pip install numpy
}

main
