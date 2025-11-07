#!/bin/bash

# Create a virtual environment.
python3.11 -m venv venv

# Activate the virtual environment.
source venv/bin/activate

# Install the packages without building.
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt --only-binary :all: