# !/bin/bash

# Copy python files (if any) in their respective directories.
if [ -f PHI.py ]; then
    cp PHI* ../PHI/
fi

if [ -f R_metric.py ]; then
    cp R_metric.py ../R_metric/R_metric.py
fi

if compgen -G "test*" > /dev/null; then
    cp test* ../Tests/
fi

if compgen -G "Ellipsoid*" > /dev/null; then
    cp Ellipsoid* ../Ellipsoid/
fi

if compgen -G "experiment*" > /dev/null; then
    cp experiment* ../Experiments/
fi