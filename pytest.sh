#!/bin/bash

# Get the current directory
CURRENT_DIR="$(pwd)"

# Export the current directory as PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$CURRENT_DIR"

# Run pytest with coverage on the src folder
pytest --cov=src tests/
