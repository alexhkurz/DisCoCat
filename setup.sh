#!/bin/bash

# Create the virtual environment
python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
