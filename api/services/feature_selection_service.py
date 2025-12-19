"""
Feature Selection Service
Handles feature selection using different methods
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, LassoCV
from typing import Dict, List, Optional
from api.repositories.data_repository import DataRepository


class FeatureSelectionService:
    """Service for feature selection operations"""
    
    def __init__(self, repository: DataRepository):
        self.repository = repository
    
    def get_feature_selection_results(self, method: Optional[str] = None, top_k: int = 5) -> Dict:
        """Get feature selection results using different methods"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        results = {}
        
        # Filter method (correlation)
        if method is None or method == 'filter':
            results['filter_method'] = self._filter_method(df, top_k)
        
        # Lasso method
        if method is None or method == 'lasso':
            results['lasso_method'] = self._lasso_method(df)
        
        # RFE method
        if method is None or method == 'rfe':
            results['rfe_method'] = self._rfe_method(df, top_k)
        
        return results
    
    def _filter_method(self, df: pd.DataFrame, top_k: int) -> Dict:
        """Filter method using correlation"""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if 'y' not in num_cols:
            return {'features': [], 'method': 'correlation', 'error': 'Target not numeric'}
        
        corr_series = df[num_cols].corr()['y'].abs().sort_values(ascending=False)
        corr_series = corr_series.drop('y').head(top_k)
        
        correlations_dict = {}
        for feature in corr_series.index:
            corr_val = corr_series[feature]
            # Handle NaN/Inf values
            if pd.isna(corr_val) or np.isinf(corr_val):
                continue
            correlations_dict[str(feature)] = float(corr_val)
        
        return {
            'features': [str(f) for f in corr_series.index.tolist()],
            'method': 'correlation',
            'correlations': correlations_dict
        }
    
    def _lasso_method(self, df: pd.DataFrame) -> Dict:
        """Lasso method for feature selection"""
        num_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['y', 'duration_log', 'age_group', 'duration_group']
        feature_cols = [col for col in num_cols if col not in exclude_cols]
        
        if 'y' not in df.columns:
            return {'features': [], 'method': 'lasso', 'error': 'Target not found'}
        
        X = df[feature_cols].values
        y = df['y'].values
        
        # Fit Lasso
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        
        # Get non-zero coefficients
        selected_features = []
        coefficients = {}
        for i in range(len(feature_cols)):
            coef_val = lasso.coef_[i]
            if abs(coef_val) > 1e-5:
                feature_name = str(feature_cols[i])
                selected_features.append(feature_name)
                # Handle NaN/Inf values
                if not (pd.isna(coef_val) or np.isinf(coef_val)):
                    coefficients[feature_name] = float(coef_val)
        
        alpha_val = float(lasso.alpha_)
        if pd.isna(alpha_val) or np.isinf(alpha_val):
            alpha_val = 0.0
        
        return {
            'features': selected_features,
            'method': 'lasso',
            'alpha': alpha_val,
            'coefficients': coefficients
        }
    
    def _rfe_method(self, df: pd.DataFrame, top_k: int) -> Dict:
        """RFE method for feature selection"""
        num_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['y', 'duration_log', 'age_group', 'duration_group']
        feature_cols = [col for col in num_cols if col not in exclude_cols]
        
        if 'y' not in df.columns:
            return {'features': [], 'method': 'rfe', 'error': 'Target not found'}
        
        X = df[feature_cols].values
        y = df['y'].values
        
        # Fit RFE
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=top_k)
        rfe.fit(X, y)
        
        selected_features = [
            str(feature_cols[i])
            for i in range(len(feature_cols))
            if rfe.support_[i]
        ]
        
        rankings = {
            str(feature_cols[i]): int(rfe.ranking_[i])
            for i in range(len(feature_cols))
        }
        
        return {
            'features': selected_features,
            'method': 'rfe',
            'base_model': 'LogisticRegression',
            'rankings': rankings
        }

