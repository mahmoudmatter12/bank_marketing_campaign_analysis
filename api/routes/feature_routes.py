"""
Feature Routes - API endpoints for feature analysis
"""
from flask import Blueprint, jsonify, request
from api.services.analysis_service import AnalysisService
from api.services.feature_selection_service import FeatureSelectionService
from api.utils.error_handler import handle_error

feature_bp = Blueprint('features', __name__)


def init_feature_routes(analysis_service: AnalysisService, feature_selection_service: FeatureSelectionService):
    """Initialize feature routes with services"""
    
    @feature_bp.route('/correlations', methods=['GET'])
    def get_correlations():
        """Get feature correlations with target"""
        try:
            top_n = int(request.args.get('top', 10))
            correlations = analysis_service.get_correlations(top_n)
            # Clean response to ensure all values are JSON serializable
            from api.utils.response_cleaner import clean_response
            cleaned = clean_response(correlations)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/categorical', methods=['GET'])
    def get_categorical_analysis():
        """Get categorical feature analysis"""
        try:
            from api.utils.response_cleaner import clean_response
            feature_name = request.args.get('feature', None)
            analysis = analysis_service.get_categorical_analysis(feature_name)
            cleaned = clean_response(analysis)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/numerical', methods=['GET'])
    def get_numerical_analysis():
        """Get numerical feature analysis"""
        try:
            from api.utils.response_cleaner import clean_response
            feature_name = request.args.get('feature', None)
            analysis = analysis_service.get_numerical_analysis(feature_name)
            cleaned = clean_response(analysis)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/selection', methods=['GET'])
    def get_feature_selection():
        """Get feature selection results"""
        try:
            method = request.args.get('method', None)
            top_k = int(request.args.get('top_k', 5))
            results = feature_selection_service.get_feature_selection_results(method, top_k)
            # Clean response to ensure all values are JSON serializable
            from api.utils.response_cleaner import clean_response
            cleaned = clean_response(results)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/heatmap', methods=['GET'])
    def get_correlation_heatmap():
        """Get full correlation heatmap matrix"""
        try:
            from api.utils.response_cleaner import clean_response
            heatmap = analysis_service.get_correlation_heatmap()
            cleaned = clean_response(heatmap)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/numerical-distributions', methods=['GET'])
    def get_numerical_distributions():
        """Get numerical feature distributions"""
        try:
            from api.utils.response_cleaner import clean_response
            distributions = analysis_service.get_numerical_distributions()
            cleaned = clean_response(distributions)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/categorical-distributions', methods=['GET'])
    def get_categorical_distributions():
        """Get categorical feature distributions"""
        try:
            from api.utils.response_cleaner import clean_response
            distributions = analysis_service.get_categorical_distributions()
            cleaned = clean_response(distributions)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/top-vs-target', methods=['GET'])
    def get_top_features_vs_target():
        """Get top features vs target boxplot data"""
        try:
            from api.utils.response_cleaner import clean_response
            top_n = int(request.args.get('top_n', 5))
            data = analysis_service.get_top_features_vs_target(top_n)
            cleaned = clean_response(data)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/top-categorical-vs-target', methods=['GET'])
    def get_top_categorical_vs_target():
        """Get top categorical features vs target grouped count plot data"""
        try:
            from api.utils.response_cleaner import clean_response
            top_n = int(request.args.get('top_n', 5))
            data = analysis_service.get_top_categorical_vs_target(top_n)
            cleaned = clean_response(data)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @feature_bp.route('/duration-comparison', methods=['GET'])
    def get_duration_comparison():
        """Get duration comparison (original vs log-transformed)"""
        try:
            from api.utils.response_cleaner import clean_response
            comparison = analysis_service.get_duration_comparison()
            cleaned = clean_response(comparison)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    return feature_bp

