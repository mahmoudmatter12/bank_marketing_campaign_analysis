# Vercel Deployment Guide

## Size Optimization

This Flask application uses large data science libraries (pandas, numpy, scipy, scikit-learn) which can exceed Vercel's 250MB unzipped limit.

### Optimizations Applied

1. ✅ **Removed unused dependencies**:
   - Removed `matplotlib` (not used in API)
   - Removed `seaborn` (not used in API)
   - Removed `gunicorn` (not needed for serverless)

2. ✅ **Updated `.vercelignore`**:
   - Excludes virtual environments, cache files, IDE files, and documentation

3. ✅ **Updated `vercel.json`**:
   - Added `excludeFiles` to prevent unnecessary files from being bundled
   - Configured function memory and timeout

### If Still Exceeding 250MB

If the deployment still fails, consider these options:

#### Option 1: Use Vercel Pro Plan

- Pro plan has higher limits (up to 1GB for serverless functions)
- Upgrade at: <https://vercel.com/pricing>

#### Option 2: Optimize Package Versions

Try using slightly older, potentially smaller versions:

```txt
pandas==1.5.3
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0
```

#### Option 3: Split into Multiple Functions

- Create separate serverless functions for different endpoints
- Reduces individual function size

#### Option 4: Use External Data Storage

- Move the CSV file to external storage (S3, etc.)
- Load data on-demand instead of bundling it

#### Option 5: Use Lighter Alternatives

- Consider using `polars` instead of `pandas` (smaller, faster)
- Use `numba` for numerical computations instead of full scipy

### Current Package Sizes (Approximate)

- pandas: ~50-80MB
- numpy: ~20-30MB
- scipy: ~50-70MB
- scikit-learn: ~30-50MB
- flask + dependencies: ~5-10MB
- CSV file: ~5.6MB
- **Total: ~160-245MB** (close to limit)

### Deployment Command

```bash
vercel deploy
```

### Monitoring

Check function size in Vercel dashboard after deployment.
