"""
Dataset Routes - API endpoints for dataset information
"""
from flask import Blueprint, jsonify, request
from api.services.analysis_service import AnalysisService
from api.utils.error_handler import handle_error

dataset_bp = Blueprint('dataset', __name__)


def init_dataset_routes(analysis_service: AnalysisService):
    """Initialize dataset routes with service"""
    
    @dataset_bp.route('/info', methods=['GET'])
    def get_dataset_info():
        """Get dataset information"""
        try:
            from api.utils.response_cleaner import clean_response
            info = analysis_service.get_dataset_info()
            cleaned = clean_response(info)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @dataset_bp.route('/statistics', methods=['GET'])
    def get_statistics():
        """Get summary statistics"""
        try:
            from api.utils.response_cleaner import clean_response
            column = request.args.get('column', None)
            stats = analysis_service.get_statistics(column)
            cleaned = clean_response(stats)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    return dataset_bp

