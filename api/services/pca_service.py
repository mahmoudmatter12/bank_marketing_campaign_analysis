"""
PCA Service - Dimensionality Reduction
Handles Principal Component Analysis operations
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
from api.repositories.data_repository import DataRepository


class PCAService:
    """Service for PCA operations"""
    
    def __init__(self, repository: DataRepository):
        self.repository = repository
        self._pca_model: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
    
    def compute_pca(self, n_components: int = 2) -> Dict:
        """Compute PCA transformation"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        # Select numerical features (exclude target and derived features)
        exclude_cols = ['y', 'duration_log', 'age_group', 'duration_group']
        num_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in num_cols if col not in exclude_cols]
        
        # Prepare data
        X = df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Store models
        self._pca_model = pca
        self._scaler = scaler
        
        # Create results dictionary
        results = {
            'n_components': n_components,
            'explained_variance': {
                f'PC{i+1}': float(pca.explained_variance_ratio_[i])
                for i in range(n_components)
            },
            'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
            'components': [
                {
                    'name': f'PC{i+1}',
                    'loadings': {
                        feature_cols[j]: float(pca.components_[i, j])
                        for j in range(len(feature_cols))
                    }
                }
                for i in range(n_components)
            ],
            'feature_names': feature_cols
        }
        
        # Store results
        self.repository.set_pca_results(results)
        
        return results
    
    def get_pca_results(self, n_components: int = 2) -> Dict:
        """Get PCA results (compute if not already computed)"""
        existing_results = self.repository.get_pca_results()
        
        if existing_results and existing_results.get('n_components') == n_components:
            return existing_results
        
        return self.compute_pca(n_components)
    
    def get_transformed_data(self, n_components: int = 2, limit: Optional[int] = None) -> Dict:
        """Get PCA transformed data points"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        # Compute PCA if needed
        self.get_pca_results(n_components)
        
        # Select numerical features
        exclude_cols = ['y', 'duration_log', 'age_group', 'duration_group']
        num_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in num_cols if col not in exclude_cols]
        
        # Prepare data
        X = df[feature_cols].values
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca_model.transform(X_scaled)
        
        # Create data points
        data_points = []
        df_subset = df.head(limit) if limit else df
        
        for idx, row in df_subset.iterrows():
            point = {
                'PC1': float(X_pca[idx, 0]),
                'PC2': float(X_pca[idx, 1]) if n_components >= 2 else None,
                'y': int(row['y']) if 'y' in row else None
            }
            data_points.append(point)
        
        return {
            'data_points': data_points,
            'count': len(data_points)
        }

