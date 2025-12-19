"""
Response Cleaner - Clean numpy/pandas types from response dictionaries
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List


def clean_value(value: Any) -> Any:
    """Recursively clean numpy/pandas types from a value"""
    # Check for numpy scalar types (NumPy 2.0 compatible)
    # Use generic checks instead of specific deprecated types
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return [clean_value(item) for item in value.tolist()]
    elif isinstance(value, (pd.Series, pd.Index)):
        return [clean_value(item) for item in value.tolist()]
    elif isinstance(value, pd.DataFrame):
        return value.to_dict(orient='records')
    elif isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    elif pd.isna(value):
        return None
    return value


def clean_response(response: Dict) -> Dict:
    """Clean a response dictionary of all numpy/pandas types"""
    return clean_value(response)

