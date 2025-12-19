# Dependency Analysis - API Requirements

## ✅ All Dependencies in requirements.txt ARE NEEDED

### 1. **pandas** ✅ REQUIRED
**Used in:**
- `api/repositories/data_repository.py` - Loading CSV, DataFrame operations
- `api/services/preprocessing_service.py` - Data cleaning, transformations
- `api/services/analysis_service.py` - Statistical analysis, correlations
- `api/services/pca_service.py` - Data preparation for PCA
- `api/services/feature_selection_service.py` - Feature selection operations
- `api/utils/json_encoder.py` - Handling pandas types in JSON
- `api/utils/response_cleaner.py` - Cleaning pandas types from responses

**Usage examples:**
- `pd.read_csv()` - Loading dataset
- `pd.DataFrame` - Data manipulation
- `pd.crosstab()` - Contingency tables
- `pd.cut()` - Creating bins/groups
- `pd.Series`, `pd.Index` - Data structures

### 2. **numpy** ✅ REQUIRED
**Used in:**
- `api/repositories/data_repository.py` - NumPy arrays
- `api/services/preprocessing_service.py` - `np.nan`, `np.where()`, `np.log1p()`, `np.clip()`
- `api/services/analysis_service.py` - Statistical calculations
- `api/services/pca_service.py` - Array operations
- `api/services/feature_selection_service.py` - Numerical operations
- `api/utils/json_encoder.py` - Handling numpy types in JSON
- `api/utils/response_cleaner.py` - Cleaning numpy types from responses

**Usage examples:**
- `np.nan` - Missing values
- `np.where()` - Conditional operations
- `np.log1p()` - Log transformation
- `np.clip()` - Outlier capping
- `np.isnan()`, `np.isinf()` - Value checks
- `np.number` - Type checking

### 3. **scipy** ✅ REQUIRED
**Used in:**
- `api/services/analysis_service.py` - Statistical tests

**Usage examples:**
- `scipy.stats.ttest_ind()` - T-tests for numerical features
- `scipy.stats.chi2_contingency()` - Chi-square tests for categorical features

**Critical for:**
- Hypothesis testing endpoints
- Statistical significance calculations

### 4. **scikit-learn** ✅ REQUIRED
**Used in:**
- `api/services/pca_service.py` - PCA decomposition
- `api/services/feature_selection_service.py` - Feature selection methods

**Usage examples:**
- `sklearn.decomposition.PCA` - Principal Component Analysis
- `sklearn.preprocessing.StandardScaler` - Feature scaling
- `sklearn.feature_selection.SelectKBest` - Filter method
- `sklearn.feature_selection.RFE` - Recursive Feature Elimination
- `sklearn.linear_model.LogisticRegression` - RFE base model
- `sklearn.linear_model.LassoCV` - Lasso feature selection

**Critical for:**
- PCA endpoints (`/api/pca/*`)
- Feature selection endpoints (`/api/features/selection`)

### 5. **flask** ✅ REQUIRED
**Used in:**
- `api/__init__.py` - Flask app creation
- All route files - Blueprint, jsonify, request handling

**Usage examples:**
- `Flask(__name__)` - App initialization
- `Blueprint()` - Route organization
- `jsonify()` - JSON responses
- `@app.route()` - Endpoint decorators

### 6. **flask-cors** ✅ REQUIRED
**Used in:**
- `api/__init__.py` - CORS configuration

**Usage examples:**
- `CORS(app, origins=CORS_ORIGINS)` - Enabling CORS for frontend

## ❌ Dependencies NOT in requirements.txt (and not needed)

- **matplotlib** - ❌ NOT USED (removed)
- **seaborn** - ❌ NOT USED (removed)
- **gunicorn** - ❌ NOT NEEDED (serverless doesn't use WSGI server)

## 📊 Summary

| Package | Status | Size Impact | Can Remove? |
|---------|--------|-------------|-------------|
| pandas | ✅ Required | ~50-80MB | ❌ No |
| numpy | ✅ Required | ~20-30MB | ❌ No |
| scipy | ✅ Required | ~50-70MB | ❌ No |
| scikit-learn | ✅ Required | ~30-50MB | ❌ No |
| flask | ✅ Required | ~1-2MB | ❌ No |
| flask-cors | ✅ Required | ~0.5MB | ❌ No |

**Total estimated size: ~150-230MB** (close to Vercel's 250MB limit)

## 🔍 Conclusion

**ALL dependencies in requirements.txt are essential and cannot be removed.**

The size issue is due to the inherent size of data science libraries (pandas, numpy, scipy, scikit-learn), not unused dependencies.

### Recommendations:
1. ✅ Keep current requirements.txt as-is
2. Consider Vercel Pro plan for higher limits (1GB)
3. Consider using `requirements-optimized.txt` with pinned versions
4. Monitor actual deployment size in Vercel dashboard

