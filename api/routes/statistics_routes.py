"""
Statistics Routes - API endpoints for statistical analysis
"""
from flask import Blueprint, jsonify, request
from api.services.analysis_service import AnalysisService
from api.utils.error_handler import handle_error

statistics_bp = Blueprint('statistics', __name__)


def init_statistics_routes(analysis_service: AnalysisService):
    """Initialize statistics routes with service"""
    
    @statistics_bp.route('/tests', methods=['GET'])
    def get_hypothesis_tests():
        """Get hypothesis test results"""
        try:
            from api.utils.response_cleaner import clean_response
            test_type = request.args.get('test_type', None)
            tests = analysis_service.get_hypothesis_tests(test_type)
            cleaned = clean_response(tests)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @statistics_bp.route('/probabilities', methods=['GET'])
    def get_probabilities():
        """Get probability calculations"""
        try:
            from api.utils.response_cleaner import clean_response
            event = request.args.get('event', None)
            probabilities = analysis_service.get_probabilities(event)
            cleaned = clean_response(probabilities)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @statistics_bp.route('/categorical-tests', methods=['GET'])
    def get_all_categorical_tests():
        """Get chi-square tests for all categorical features"""
        try:
            from api.utils.response_cleaner import clean_response
            tests = analysis_service.get_all_categorical_tests()
            cleaned = clean_response(tests)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @statistics_bp.route('/numerical-tests', methods=['GET'])
    def get_all_numerical_tests():
        """Get t-tests for all numerical features"""
        try:
            from api.utils.response_cleaner import clean_response
            tests = analysis_service.get_all_numerical_tests()
            cleaned = clean_response(tests)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @statistics_bp.route('/significance-ranking', methods=['GET'])
    def get_significance_ranking():
        """Get ranking of features by statistical significance"""
        try:
            from api.utils.response_cleaner import clean_response
            ranking = analysis_service.get_statistical_significance_ranking()
            cleaned = clean_response(ranking)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @statistics_bp.route('/contingency-table', methods=['GET'])
    def get_contingency_table():
        """Get contingency table for a categorical feature"""
        try:
            from api.utils.response_cleaner import clean_response
            feature = request.args.get('feature', None)
            if not feature:
                return jsonify({'error': 'Feature parameter is required'}), 400
            table = analysis_service.get_contingency_table(feature)
            cleaned = clean_response(table)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    @statistics_bp.route('/feature-comparison', methods=['GET'])
    def get_feature_comparison():
        """Get detailed comparison for a feature"""
        try:
            from api.utils.response_cleaner import clean_response
            feature = request.args.get('feature', None)
            if not feature:
                return jsonify({'error': 'Feature parameter is required'}), 400
            comparison = analysis_service.get_feature_comparison(feature)
            cleaned = clean_response(comparison)
            return jsonify(cleaned), 200
        except Exception as e:
            return handle_error(e)
    
    return statistics_bp

