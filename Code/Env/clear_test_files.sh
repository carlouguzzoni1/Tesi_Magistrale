#!/bin/bash

# Remove notebooks.
if compgen -G "*.ipynb" > /dev/null; then
    rm *.ipynb
fi

# Remove python files.
if compgen -G "*.py" > /dev/null; then
    rm *.py
fi