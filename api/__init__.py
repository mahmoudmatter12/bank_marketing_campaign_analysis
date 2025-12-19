"""
Flask Application Factory
"""
from flask import Flask, jsonify
from flask_cors import CORS

from api.config import CORS_ORIGINS
from api.repositories.data_repository import DataRepository
from api.routes.dataset_routes import init_dataset_routes
from api.routes.feature_routes import init_feature_routes
from api.routes.pca_routes import init_pca_routes
from api.routes.statistics_routes import init_statistics_routes
from api.routes.target_routes import init_target_routes
from api.services.analysis_service import AnalysisService
from api.services.feature_selection_service import FeatureSelectionService
from api.services.pca_service import PCAService
from api.services.preprocessing_service import PreprocessingService
from api.utils.json_encoder import CustomJSONProvider


def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Set custom JSON provider to handle numpy/pandas types
    app.json = CustomJSONProvider(app)
    
    # Enable CORS
    CORS(app, origins=CORS_ORIGINS)
    
    # Initialize repository
    repository = DataRepository()
    
    # Initialize services
    preprocessing_service = PreprocessingService(repository)
    analysis_service = AnalysisService(repository)
    pca_service = PCAService(repository)
    feature_selection_service = FeatureSelectionService(repository)
    
    # Preprocess data on startup
    try:
        print("Loading and preprocessing data...")
        preprocessing_service.preprocess_data()
        print("Data preprocessing completed successfully!")
    except Exception as e:
        print(f"Warning: Error during data preprocessing: {e}")
        print("Some endpoints may not work until data is loaded.")
    
    # Register blueprints
    app.register_blueprint(
        init_dataset_routes(analysis_service),
        url_prefix='/api/dataset'
    )
    
    app.register_blueprint(
        init_target_routes(analysis_service),
        url_prefix='/api/target'
    )
    
    app.register_blueprint(
        init_feature_routes(analysis_service, feature_selection_service),
        url_prefix='/api/features'
    )
    
    app.register_blueprint(
        init_statistics_routes(analysis_service),
        url_prefix='/api/statistics'
    )
    
    app.register_blueprint(
        init_pca_routes(pca_service),
        url_prefix='/api/pca'
    )
    
    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'message': 'Bank Marketing Analysis API is running',
            'data_loaded': repository.is_data_loaded()
        }), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint"""
        return jsonify({
            'message': 'Bank Marketing Campaign Analysis API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/api/health',
                'dataset_info': '/api/dataset/info',
                'dataset_statistics': '/api/dataset/statistics',
                'target_distribution': '/api/target/distribution',
                'correlations': '/api/features/correlations',
                'categorical_analysis': '/api/features/categorical',
                'numerical_analysis': '/api/features/numerical',
                'feature_selection': '/api/features/selection',
                'hypothesis_tests': '/api/statistics/tests',
                'probabilities': '/api/statistics/probabilities',
                'pca_results': '/api/pca/results',
                'pca_components': '/api/pca/components',
                'pca_transformed': '/api/pca/transformed'
            }
        }), 200
    
    return app

