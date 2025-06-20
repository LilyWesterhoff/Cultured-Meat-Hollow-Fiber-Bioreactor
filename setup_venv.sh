#!/bin/bash
# setup.sh - Setup script
# This script sets up a Python virtual environment and installs required packages
# Assumes Python 3.8+ is installed

# Define venv name
VENV_NAME="bioprocess_venv"
PYTHON_VERSION="python3"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    # Check if Python is installed
    if ! command -v $PYTHON_VERSION &> /dev/null; then
        echo "Error: Python 3 is not installed or not found. Please install Python 3.8+ and try again."
        exit 1
    fi

    echo "Creating virtual environment: $VENV_NAME"
    $PYTHON_VERSION -m venv $VENV_NAME
    $VENV_NAME/bin/pip install --upgrade pip
    $VENV_NAME/bin/pip install -r requirements.txt
else
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
fi

echo "Setup complete! Now running the app."
$VENV_NAME/bin/streamlit run solve_pmm_model.py

