"""
Data Repository - Data Access Layer
Handles all data loading and storage operations
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from api.config import DATASET_PATH


class DataRepository:
    """Repository for data access operations"""
    
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._df_clean: Optional[pd.DataFrame] = None
        self._pca_results: Optional[dict] = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset from CSV file"""
        if self._df is None:
            self._df = pd.read_csv(DATASET_PATH, sep=';')
        return self._df
    
    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Get raw dataset"""
        return self._df
    
    def set_cleaned_data(self, df: pd.DataFrame) -> None:
        """Store cleaned dataset"""
        self._df_clean = df.copy()
    
    def get_cleaned_data(self) -> Optional[pd.DataFrame]:
        """Get cleaned dataset"""
        return self._df_clean
    
    def set_pca_results(self, results: dict) -> None:
        """Store PCA results"""
        self._pca_results = results
    
    def get_pca_results(self) -> Optional[dict]:
        """Get PCA results"""
        return self._pca_results
    
    def is_data_loaded(self) -> bool:
        """Check if data is loaded"""
        return self._df_clean is not None

