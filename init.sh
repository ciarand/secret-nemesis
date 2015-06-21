#!/usr/bin/env bash

# stop immediately if anything fails
set -eo pipefail

main() {
    echo "Checking requirements..."
    python --version > /dev/null 2>&1     || (echo "You need python in your \$PATH" && false)
    pip --version > /dev/null 2>&1        || (echo "You need pip in your \$PATH" && false)
    virtualenv --version > /dev/null 2>&1 || pip install virtualenv

    echo "Setting up virtualenv..."
    virtualenv venv

    echo "Installing deap..."
    venv/bin/pip install deap

    echo "Installing numpy..."
    venv/bin/pip install numpy
}

main
