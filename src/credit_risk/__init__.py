"""Credit Decision & Risk Intelligence System (regulation-grounded prototype)."""
import warnings

# LightGBM >= 4.6 records generated feature names even when fitted on the numpy
# output of a ColumnTransformer, so scikit-learn warns on every predict. The
# column order is fixed by the pipeline, so the warning is noise.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

__version__ = "0.1.0"
