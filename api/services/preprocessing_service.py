"""
Preprocessing Service - Data Cleaning and Transformation
Implements all data preprocessing steps from the analysis notebook
"""
import pandas as pd
import numpy as np
from typing import Tuple
from api.repositories.data_repository import DataRepository


class PreprocessingService:
    """Service for data preprocessing operations"""
    
    def __init__(self, repository: DataRepository):
        self.repository = repository
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline
        Returns cleaned and preprocessed dataframe
        """
        # Load raw data
        df = self.repository.load_raw_data()
        
        # Create working copy
        df_clean = df.copy()
        
        # Step 1: Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace('.', '_').str.replace('-', '_')
        
        # Step 2: Standardize categorical values to lowercase
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].str.lower()
        
        # Step 3: Replace "unknown" with NaN
        df_clean.replace("unknown", np.nan, inplace=True)
        
        # Step 4: Handle pdays (999 = not previously contacted)
        if 'pdays' in df_clean.columns:
            df_clean['pdays'] = np.where(df_clean['pdays'] == 999, 0, 1)
        
        # Step 5: Fill missing values
        # Numerical columns: fill with median
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
        
        # Categorical columns: fill with mode
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'unknown'
                df_clean[col].fillna(mode_value, inplace=True)
        
        # Step 6: Remove duplicates
        df_clean.drop_duplicates(inplace=True)
        
        # Step 7: Encode target variable
        if 'y' in df_clean.columns:
            df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0})
        
        # Step 8: Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Step 9: Create derived features
        df_clean = self._create_derived_features(df_clean)
        
        # Store cleaned data
        self.repository.set_cleaned_data(df_clean)
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method and log transformation"""
        df = df.copy()
        
        # Log transformation for duration (highly skewed)
        if 'duration' in df.columns:
            df['duration_log'] = np.log1p(df['duration'])
        
        # IQR capping for other numerical features (excluding target and duration_log)
        num_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['y', 'duration_log', 'pdays']  # Don't cap these
        
        for col in num_cols:
            if col in exclude_cols:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features like age groups and duration groups"""
        df = df.copy()
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 30, 50, 100],
                labels=['young', 'adult', 'senior']
            )
        
        # Duration groups
        if 'duration' in df.columns:
            df['duration_group'] = pd.cut(
                df['duration'],
                bins=[0, 100, 300, 1000],
                labels=['short', 'medium', 'long']
            )
        
        return df

