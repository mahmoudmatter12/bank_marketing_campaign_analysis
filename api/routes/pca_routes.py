"""
PCA Routes - API endpoints for PCA analysis
"""
from flask import Blueprint, jsonify, request
from api.services.pca_service import PCAService
from api.utils.error_handler import handle_error

pca_bp = Blueprint('pca', __name__)


def init_pca_routes(pca_service: PCAService):
    """Initialize PCA routes with service"""
    
    @pca_bp.route('/results', methods=['GET'])
    def get_pca_results():
        """Get PCA results"""
        try:
            from api.utils.response_cleaner import clean_response
            n_components = int(request.args.get('n', 2))
            results = pca_service.get_pca_results(n_components)
            cleaned = clean_response(results)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @pca_bp.route('/components', methods=['GET'])
    def get_pca_components():
        """Get PCA components"""
        try:
            from api.utils.response_cleaner import clean_response
            n_components = int(request.args.get('n', 2))
            results = pca_service.get_pca_results(n_components)
            response_data = {
                'components': results['components'],
                'explained_variance': results['explained_variance']
            }
            cleaned = clean_response(response_data)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @pca_bp.route('/transformed', methods=['GET'])
    def get_transformed_data():
        """Get PCA transformed data points"""
        try:
            from api.utils.response_cleaner import clean_response
            n_components = int(request.args.get('n', 2))
            limit = request.args.get('limit', None)
            limit = int(limit) if limit else None
            data = pca_service.get_transformed_data(n_components, limit)
            cleaned = clean_response(data)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    return pca_bp

