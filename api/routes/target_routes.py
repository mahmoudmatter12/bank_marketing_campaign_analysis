"""
Target Routes - API endpoints for target variable analysis
"""
from flask import Blueprint, jsonify
from api.services.analysis_service import AnalysisService
from api.utils.error_handler import handle_error

target_bp = Blueprint('target', __name__)


def init_target_routes(analysis_service: AnalysisService):
    """Initialize target routes with service"""
    
    @target_bp.route('/distribution', methods=['GET'])
    def get_target_distribution():
        """Get target variable distribution"""
        try:
            from api.utils.response_cleaner import clean_response
            distribution = analysis_service.get_target_distribution()
            cleaned = clean_response(distribution)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    return target_bp

