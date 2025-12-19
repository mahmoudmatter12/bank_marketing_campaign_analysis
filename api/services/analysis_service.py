"""
Analysis Service - Statistical Analysis and Feature Analysis
Handles correlations, hypothesis tests, probabilities, etc.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from typing import Dict, List, Optional
from api.repositories.data_repository import DataRepository


class AnalysisService:
    """Service for statistical analysis operations"""
    
    def __init__(self, repository: DataRepository):
        self.repository = repository
    
    def get_dataset_info(self) -> Dict:
        """Get dataset information"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        return {
            'shape': {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1])
            },
            'columns': [
                {
                    'name': col,
                    'type': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'unique_values': int(df[col].nunique())
                }
                for col in df.columns
            ],
            'target_variable': {
                'name': 'y',
                'type': str(df['y'].dtype) if 'y' in df.columns else 'unknown',
                'values': df['y'].unique().tolist() if 'y' in df.columns else []
            }
        }
    
    def get_statistics(self, column: Optional[str] = None) -> Dict:
        """Get summary statistics"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        if column:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
            
            stats_data = df[column].describe().to_dict()
            return {
                'feature': column,
                'statistics': stats_data
            }
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            stats_dict = {}
            for col in num_cols:
                stats_dict[col] = df[col].describe().to_dict()
            
            return stats_dict
    
    def get_target_distribution(self) -> Dict:
        """Get target variable distribution"""
        df = self.repository.get_cleaned_data()
        if df is None or 'y' not in df.columns:
            raise ValueError("Target variable not found")
        
        value_counts = df['y'].value_counts()
        total = len(df)
        
        subscribed_count = int(value_counts.get(1, 0))
        not_subscribed_count = int(value_counts.get(0, 0))
        
        return {
            'total': int(total),
            'subscribed': {
                'count': subscribed_count,
                'percentage': round((subscribed_count / total) * 100, 2)
            },
            'not_subscribed': {
                'count': not_subscribed_count,
                'percentage': round((not_subscribed_count / total) * 100, 2)
            },
            'imbalance_ratio': round(not_subscribed_count / subscribed_count, 2) if subscribed_count > 0 else 0
        }
    
    def get_correlations(self, top_n: int = 10) -> Dict:
        """Get feature correlations with target"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if 'y' not in num_cols:
            raise ValueError("Target variable not numeric")
        
        # Calculate correlations
        corr_series = df[num_cols].corr()['y'].abs().sort_values(ascending=False)
        corr_series = corr_series.drop('y').head(top_n)
        
        top_features = []
        for feature in corr_series.index:
            corr_value = df[num_cols].corr()['y'][feature]
            # Handle NaN/Inf values
            if pd.isna(corr_value) or np.isinf(corr_value):
                continue
            top_features.append({
                'feature': str(feature),
                'correlation': float(corr_value),
                'direction': 'positive' if corr_value > 0 else 'negative'
            })
        
        return {
            'top_features': top_features,
            'count': int(len(top_features))
        }
    
    def get_categorical_analysis(self, feature_name: Optional[str] = None) -> Dict:
        """Get categorical feature analysis"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if feature_name:
            if feature_name not in cat_cols:
                raise ValueError(f"Categorical feature {feature_name} not found")
            
            # Calculate subscription rates by category
            crosstab = pd.crosstab(df[feature_name], df['y'], normalize='index') * 100
            
            categories = []
            for category in crosstab.index:
                categories.append({
                    'category': str(category),
                    'subscription_rate': float(crosstab.loc[category, 1]) if 1 in crosstab.columns else 0.0,
                    'count': int((df[feature_name] == category).sum())
                })
            
            return {
                'feature': feature_name,
                'categories': categories
            }
        else:
            return {
                'categorical_features': cat_cols.tolist(),
                'count': len(cat_cols)
            }
    
    def get_numerical_analysis(self, feature_name: Optional[str] = None) -> Dict:
        """Get numerical feature analysis"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if feature_name:
            if feature_name not in num_cols:
                raise ValueError(f"Numerical feature {feature_name} not found")
            
            # Calculate statistics by target
            stats_by_target = {}
            for target_val in [0, 1]:
                subset = df[df['y'] == target_val][feature_name]
                stats_by_target[f"target_{target_val}"] = {
                    'mean': float(subset.mean()),
                    'std': float(subset.std()),
                    'median': float(subset.median()),
                    'count': int(len(subset))
                }
            
            # Overall statistics
            overall_stats = df[feature_name].describe().to_dict()
            
            return {
                'feature': feature_name,
                'overall_statistics': overall_stats,
                'statistics_by_target': stats_by_target
            }
        else:
            return {
                'numerical_features': num_cols.tolist(),
                'count': len(num_cols)
            }
    
    def get_hypothesis_tests(self, test_type: Optional[str] = None) -> Dict:
        """Get hypothesis test results"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        tests = []
        
        # T-tests for numerical features
        if test_type is None or test_type == 't_test':
            # Duration test
            if 'duration' in df.columns:
                group_yes = df[df['y'] == 1]['duration']
                group_no = df[df['y'] == 0]['duration']
                t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)
                
                tests.append({
                    'test_name': 'duration_difference',
                    'test_type': 't_test',
                    'hypothesis': {
                        'null': 'No difference in mean duration between subscribers and non-subscribers',
                        'alternative': 'Significant difference exists'
                    },
                    'results': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significance': self._get_significance_level(p_value),
                        'interpretation': 'REJECT NULL HYPOTHESIS' if p_value < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'
                    },
                    'group_statistics': {
                        'subscribers': {
                            'mean': float(group_yes.mean()),
                            'std': float(group_yes.std()),
                            'count': int(len(group_yes))
                        },
                        'non_subscribers': {
                            'mean': float(group_no.mean()),
                            'std': float(group_no.std()),
                            'count': int(len(group_no))
                        }
                    }
                })
            
            # Age test
            if 'age' in df.columns:
                group_yes = df[df['y'] == 1]['age']
                group_no = df[df['y'] == 0]['age']
                t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)
                
                tests.append({
                    'test_name': 'age_difference',
                    'test_type': 't_test',
                    'hypothesis': {
                        'null': 'No difference in mean age between subscribers and non-subscribers',
                        'alternative': 'Significant difference exists'
                    },
                    'results': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significance': self._get_significance_level(p_value),
                        'interpretation': 'REJECT NULL HYPOTHESIS' if p_value < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'
                    },
                    'group_statistics': {
                        'subscribers': {
                            'mean': float(group_yes.mean()),
                            'std': float(group_yes.std()),
                            'count': int(len(group_yes))
                        },
                        'non_subscribers': {
                            'mean': float(group_no.mean()),
                            'std': float(group_no.std()),
                            'count': int(len(group_no))
                        }
                    }
                })
        
        # Chi-square tests for categorical features
        if test_type is None or test_type == 'chi_square':
            # Housing loan test
            if 'housing' in df.columns:
                table = pd.crosstab(df['housing'], df['y'])
                chi2, p_value, dof, expected = chi2_contingency(table)
                
                tests.append({
                    'test_name': 'housing_loan_association',
                    'test_type': 'chi_square',
                    'hypothesis': {
                        'null': 'Housing loan and subscription are independent',
                        'alternative': 'Housing loan and subscription are dependent'
                    },
                    'results': {
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significance': self._get_significance_level(p_value),
                        'interpretation': 'REJECT NULL HYPOTHESIS' if p_value < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'
                    }
                })
            
            # Personal loan test
            if 'loan' in df.columns:
                table = pd.crosstab(df['loan'], df['y'])
                chi2, p_value, dof, expected = chi2_contingency(table)
                
                tests.append({
                    'test_name': 'personal_loan_association',
                    'test_type': 'chi_square',
                    'hypothesis': {
                        'null': 'Personal loan and subscription are independent',
                        'alternative': 'Personal loan and subscription are dependent'
                    },
                    'results': {
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significance': self._get_significance_level(p_value),
                        'interpretation': 'REJECT NULL HYPOTHESIS' if p_value < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'
                    }
                })
        
        return {
            'tests': tests,
            'count': len(tests)
        }
    
    def get_probabilities(self, event: Optional[str] = None) -> Dict:
        """Get probability calculations"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        probabilities = {}
        
        # Overall subscription probability
        prob_subscribe = (df['y'] == 1).mean()
        probabilities['overall_subscription'] = {
            'probability': float(prob_subscribe),
            'percentage': float(prob_subscribe * 100),
            'description': 'Probability that a customer subscribes'
        }
        
        # Conditional probabilities
        if event == 'long_call' or event is None:
            if 'duration_log' in df.columns:
                threshold = 5  # log-transformed duration > 5
                prob_long = (df['duration_log'] > threshold).mean()
                prob_subscribe_given_long = df[df['duration_log'] > threshold]['y'].mean()
                
                probabilities['long_call'] = {
                    'probability': float(prob_long),
                    'percentage': float(prob_long * 100),
                    'description': 'Probability of long call duration'
                }
                
                probabilities['subscribe_given_long_call'] = {
                    'probability': float(prob_subscribe_given_long),
                    'percentage': float(prob_subscribe_given_long * 100),
                    'description': 'Probability of subscription given long call'
                }
        
        return probabilities
    
    def get_correlation_heatmap(self) -> Dict:
        """Get full correlation matrix for numerical features"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[num_cols].corr()
        
        # Convert to list format for frontend
        features = corr_matrix.columns.tolist()
        correlations = []
        
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                corr_value = corr_matrix.loc[feature1, feature2]
                if pd.isna(corr_value) or np.isinf(corr_value):
                    corr_value = 0.0
                correlations.append({
                    'feature1': str(feature1),
                    'feature2': str(feature2),
                    'correlation': float(corr_value)
                })
        
        return {
            'features': features,
            'correlations': correlations,
            'matrix': corr_matrix.to_dict()
        }
    
    def get_numerical_distributions(self) -> Dict:
        """Get distribution data for all numerical features (for histograms)"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [col for col in num_cols if col != 'y']  # Exclude target
        
        distributions = []
        for col in num_cols:
            values = df[col].dropna().tolist()
            distributions.append({
                'feature': str(col),
                'values': [float(v) for v in values if not (pd.isna(v) or np.isinf(v))],
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            })
        
        return {
            'features': [str(col) for col in num_cols],
            'distributions': distributions,
            'count': len(distributions)
        }
    
    def get_categorical_distributions(self) -> Dict:
        """Get distribution data for all categorical features (for count plots)"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        
        distributions = []
        for col in cat_cols:
            value_counts = df[col].value_counts()
            categories = []
            for category, count in value_counts.items():
                categories.append({
                    'category': str(category),
                    'count': int(count),
                    'percentage': float((count / len(df)) * 100)
                })
            
            distributions.append({
                'feature': str(col),
                'categories': categories,
                'total_categories': len(categories)
            })
        
        return {
            'features': [str(col) for col in cat_cols],
            'distributions': distributions,
            'count': len(distributions)
        }
    
    def get_top_features_vs_target(self, top_n: int = 5) -> Dict:
        """Get boxplot data for top numerical features vs target"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        if 'y' not in num_cols:
            raise ValueError("Target variable not numeric")
        
        # Get top correlated features
        corr_series = df[num_cols].corr()['y'].abs().sort_values(ascending=False)
        top_features = corr_series.drop('y').head(top_n).index.tolist()
        
        boxplot_data = []
        for feature in top_features:
            # Get data for each target group
            group_0 = df[df['y'] == 0][feature].dropna().tolist()
            group_1 = df[df['y'] == 1][feature].dropna().tolist()
            
            boxplot_data.append({
                'feature': str(feature),
                'correlation': float(df[num_cols].corr()['y'][feature]),
                'target_0': {
                    'values': [float(v) for v in group_0 if not (pd.isna(v) or np.isinf(v))],
                    'mean': float(df[df['y'] == 0][feature].mean()),
                    'median': float(df[df['y'] == 0][feature].median()),
                    'count': int(len(group_0))
                },
                'target_1': {
                    'values': [float(v) for v in group_1 if not (pd.isna(v) or np.isinf(v))],
                    'mean': float(df[df['y'] == 1][feature].mean()),
                    'median': float(df[df['y'] == 1][feature].median()),
                    'count': int(len(group_1))
                }
            })
        
        return {
            'features': [str(f) for f in top_features],
            'boxplot_data': boxplot_data,
            'count': len(boxplot_data)
        }
    
    def get_top_categorical_vs_target(self, top_n: int = 5) -> Dict:
        """Get grouped count plot data for top categorical features vs target"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        
        # Calculate subscription rates for each categorical feature
        feature_scores = []
        for col in cat_cols:
            crosstab = pd.crosstab(df[col], df['y'], normalize='index')
            if 1 in crosstab.columns:
                avg_subscription_rate = crosstab[1].mean()
                feature_scores.append({
                    'feature': col,
                    'avg_subscription_rate': float(avg_subscription_rate)
                })
        
        # Sort by subscription rate and get top N
        feature_scores.sort(key=lambda x: x['avg_subscription_rate'], reverse=True)
        top_features = [item['feature'] for item in feature_scores[:top_n]]
        
        grouped_data = []
        for feature in top_features:
            crosstab = pd.crosstab(df[feature], df['y'])
            categories = []
            
            for category in crosstab.index:
                count_0 = int(crosstab.loc[category, 0]) if 0 in crosstab.columns else 0
                count_1 = int(crosstab.loc[category, 1]) if 1 in crosstab.columns else 0
                total = count_0 + count_1
                
                categories.append({
                    'category': str(category),
                    'target_0': count_0,
                    'target_1': count_1,
                    'total': total,
                    'subscription_rate': float((count_1 / total * 100) if total > 0 else 0)
                })
            
            grouped_data.append({
                'feature': str(feature),
                'categories': categories
            })
        
        return {
            'features': [str(f) for f in top_features],
            'grouped_data': grouped_data,
            'count': len(grouped_data)
        }
    
    def get_duration_comparison(self) -> Dict:
        """Get comparison data for original vs log-transformed duration"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        if 'duration' not in df.columns:
            raise ValueError("Duration feature not found")
        
        # Ensure duration_log exists
        if 'duration_log' not in df.columns:
            df['duration_log'] = np.log1p(df['duration'])
        
        original_values = df['duration'].dropna().tolist()
        log_values = df['duration_log'].dropna().tolist()
        
        return {
            'original': {
                'values': [float(v) for v in original_values if not (pd.isna(v) or np.isinf(v))],
                'mean': float(df['duration'].mean()),
                'std': float(df['duration'].std()),
                'skewness': float(df['duration'].skew()),
                'min': float(df['duration'].min()),
                'max': float(df['duration'].max())
            },
            'log_transformed': {
                'values': [float(v) for v in log_values if not (pd.isna(v) or np.isinf(v))],
                'mean': float(df['duration_log'].mean()),
                'std': float(df['duration_log'].std()),
                'skewness': float(df['duration_log'].skew()),
                'min': float(df['duration_log'].min()),
                'max': float(df['duration_log'].max())
            },
            'threshold': {
                'log_value': 5.0,
                'original_value': float(np.exp(5) - 1)
            }
        }
    
    def get_all_categorical_tests(self) -> Dict:
        """Get chi-square tests for all categorical features"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        tests = []
        
        for col in cat_cols:
            try:
                table = pd.crosstab(df[col], df['y'])
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue
                
                chi2, p_value, dof, expected = chi2_contingency(table)
                
                # Calculate subscription rates by category
                rates = pd.crosstab(df[col], df['y'], normalize='index') * 100
                categories = []
                if 1 in rates.columns:
                    for category in rates.index:
                        rate = float(rates.loc[category, 1])
                        count_0 = int(table.loc[category, 0]) if 0 in table.columns else 0
                        count_1 = int(table.loc[category, 1]) if 1 in table.columns else 0
                        categories.append({
                            'category': str(category),
                            'subscription_rate': rate,
                            'count_0': count_0,
                            'count_1': count_1,
                            'total': count_0 + count_1
                        })
                
                tests.append({
                    'feature': str(col),
                    'test_type': 'chi_square',
                    'results': {
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significance': self._get_significance_level(p_value),
                        'interpretation': 'REJECT NULL HYPOTHESIS' if p_value < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'
                    },
                    'categories': categories,
                    'contingency_table': {
                        'target_0': {str(k): int(v) for k, v in table[0].to_dict().items()} if 0 in table.columns else {},
                        'target_1': {str(k): int(v) for k, v in table[1].to_dict().items()} if 1 in table.columns else {}
                    }
                })
            except Exception as e:
                continue
        
        # Sort by p-value (most significant first)
        tests.sort(key=lambda x: x['results']['p_value'])
        
        return {
            'tests': tests,
            'count': len(tests),
            'significant_count': sum(1 for t in tests if t['results']['p_value'] < 0.05)
        }
    
    def get_all_numerical_tests(self) -> Dict:
        """Get t-tests for all numerical features"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [col for col in num_cols if col != 'y']  # Exclude target
        
        tests = []
        
        for col in num_cols:
            try:
                group_yes = df[df['y'] == 1][col].dropna()
                group_no = df[df['y'] == 0][col].dropna()
                
                if len(group_yes) < 2 or len(group_no) < 2:
                    continue
                
                t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)
                
                tests.append({
                    'feature': str(col),
                    'test_type': 't_test',
                    'results': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significance': self._get_significance_level(p_value),
                        'interpretation': 'REJECT NULL HYPOTHESIS' if p_value < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'
                    },
                    'group_statistics': {
                        'subscribers': {
                            'mean': float(group_yes.mean()),
                            'std': float(group_yes.std()),
                            'median': float(group_yes.median()),
                            'count': int(len(group_yes))
                        },
                        'non_subscribers': {
                            'mean': float(group_no.mean()),
                            'std': float(group_no.std()),
                            'median': float(group_no.median()),
                            'count': int(len(group_no))
                        }
                    },
                    'difference': {
                        'mean_diff': float(group_yes.mean() - group_no.mean()),
                        'percent_diff': float(((group_yes.mean() - group_no.mean()) / group_no.mean() * 100) if group_no.mean() != 0 else 0)
                    }
                })
            except Exception as e:
                continue
        
        # Sort by p-value (most significant first)
        tests.sort(key=lambda x: x['results']['p_value'])
        
        return {
            'tests': tests,
            'count': len(tests),
            'significant_count': sum(1 for t in tests if t['results']['p_value'] < 0.05)
        }
    
    def get_statistical_significance_ranking(self) -> Dict:
        """Get ranking of all features by statistical significance"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        rankings = []
        
        # Get numerical tests
        num_tests = self.get_all_numerical_tests()
        for test in num_tests['tests']:
            rankings.append({
                'feature': test['feature'],
                'feature_type': 'numerical',
                'test_type': 't_test',
                'p_value': test['results']['p_value'],
                'statistic': test['results']['t_statistic'],
                'significance': test['results']['significance']
            })
        
        # Get categorical tests
        cat_tests = self.get_all_categorical_tests()
        for test in cat_tests['tests']:
            rankings.append({
                'feature': test['feature'],
                'feature_type': 'categorical',
                'test_type': 'chi_square',
                'p_value': test['results']['p_value'],
                'statistic': test['results']['chi2_statistic'],
                'significance': test['results']['significance']
            })
        
        # Sort by p-value
        rankings.sort(key=lambda x: x['p_value'])
        
        # Add rank
        for i, ranking in enumerate(rankings, 1):
            ranking['rank'] = i
        
        return {
            'rankings': rankings,
            'count': len(rankings),
            'top_significant': rankings[:10]
        }
    
    def get_contingency_table(self, feature: str) -> Dict:
        """Get contingency table for a categorical feature"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found")
        
        if df[feature].dtype == 'object':
            table = pd.crosstab(df[feature], df['y'], margins=True)
            
            # Calculate percentages
            table_percent = pd.crosstab(df[feature], df['y'], normalize='index') * 100
            
            categories = []
            for category in table.index:
                if category == 'All':
                    continue
                count_0 = int(table.loc[category, 0]) if 0 in table.columns else 0
                count_1 = int(table.loc[category, 1]) if 1 in table.columns else 0
                total = count_0 + count_1
                
                percent_0 = float(table_percent.loc[category, 0]) if 0 in table_percent.columns else 0
                percent_1 = float(table_percent.loc[category, 1]) if 1 in table_percent.columns else 0
                
                categories.append({
                    'category': str(category),
                    'target_0': {
                        'count': count_0,
                        'percentage': percent_0
                    },
                    'target_1': {
                        'count': count_1,
                        'percentage': percent_1
                    },
                    'total': total
                })
            
            return {
                'feature': feature,
                'categories': categories,
                'total': {
                    'target_0': int(table.loc['All', 0]) if 0 in table.columns else 0,
                    'target_1': int(table.loc['All', 1]) if 1 in table.columns else 0,
                    'all': int(table.loc['All', 'All'])
                }
            }
        else:
            raise ValueError(f"Feature {feature} is not categorical")
    
    def get_feature_comparison(self, feature: str) -> Dict:
        """Get detailed comparison for a feature between target groups"""
        df = self.repository.get_cleaned_data()
        if df is None:
            raise ValueError("Dataset not loaded")
        
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} not found")
        
        if df[feature].dtype == 'object':
            # Categorical feature
            table = pd.crosstab(df[feature], df['y'])
            chi2, p_value, dof, expected = chi2_contingency(table)
            
            return {
                'feature': feature,
                'feature_type': 'categorical',
                'test': {
                    'type': 'chi_square',
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'dof': int(dof),
                    'significance': self._get_significance_level(p_value)
                },
                'groups': {
                    'target_0': {
                        'count': int((df['y'] == 0).sum()),
                        'distribution': {str(k): int(v) for k, v in table[0].to_dict().items()} if 0 in table.columns else {}
                    },
                    'target_1': {
                        'count': int((df['y'] == 1).sum()),
                        'distribution': {str(k): int(v) for k, v in table[1].to_dict().items()} if 1 in table.columns else {}
                    }
                }
            }
        else:
            # Numerical feature
            group_yes = df[df['y'] == 1][feature].dropna()
            group_no = df[df['y'] == 0][feature].dropna()
            
            t_stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)
            
            return {
                'feature': feature,
                'feature_type': 'numerical',
                'test': {
                    'type': 't_test',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significance': self._get_significance_level(p_value)
                },
                'groups': {
                    'target_0': {
                        'count': int(len(group_no)),
                        'mean': float(group_no.mean()),
                        'std': float(group_no.std()),
                        'median': float(group_no.median()),
                        'min': float(group_no.min()),
                        'max': float(group_no.max())
                    },
                    'target_1': {
                        'count': int(len(group_yes)),
                        'mean': float(group_yes.mean()),
                        'std': float(group_yes.std()),
                        'median': float(group_yes.median()),
                        'min': float(group_yes.min()),
                        'max': float(group_yes.max())
                    }
                },
                'difference': {
                    'mean_diff': float(group_yes.mean() - group_no.mean()),
                    'percent_diff': float(((group_yes.mean() - group_no.mean()) / group_no.mean() * 100) if group_no.mean() != 0 else 0)
                }
            }
    
    def _get_significance_level(self, p_value: float) -> str:
        """Get significance level description"""
        if p_value < 0.001:
            return 'highly significant'
        elif p_value < 0.01:
            return 'very significant'
        elif p_value < 0.05:
            return 'significant'
        else:
            return 'not significant'

