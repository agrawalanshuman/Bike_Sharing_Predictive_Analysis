"""Historical bike-sharing forecasting and decision-analysis components."""
import os
import tempfile
from pathlib import Path

# Set numeric parallelism before importing NumPy/scikit-learn.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'bike-mpl'))
