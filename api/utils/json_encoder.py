"""
Custom JSON Encoder for Flask
Handles numpy and pandas types that aren't JSON serializable by default
"""
import json
import numpy as np
import pandas as pd
from flask.json.provider import DefaultJSONProvider


class CustomJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles numpy and pandas types"""
    
    def default(self, obj):
        """Convert numpy/pandas types to native Python types"""
        # Check for numpy scalar types (NumPy 2.0 compatible)
        # Use generic checks instead of specific deprecated types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and Infinity
            try:
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            except (TypeError, ValueError):
                return None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Check for numpy arrays
        elif isinstance(obj, np.ndarray):
            return [self.default(item) for item in obj.tolist()]
        # Check for pandas types
        elif isinstance(obj, (pd.Series, pd.Index)):
            return [self.default(item) for item in obj.tolist()]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        # Check for pandas NA
        elif pd.isna(obj):
            return None
        # Handle dictionaries and lists recursively
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        return super().default(obj)

