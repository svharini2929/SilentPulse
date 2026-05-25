import os
import sys

# Inject project root into sys.path to support absolute package imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, jsonify
from flask_cors import CORS
from backend.routes.ecg_routes import ecg_bp

def create_app():
    """Factory function to create and configure the Flask application."""
    frontend_dir = os.path.join(BASE_DIR, "frontend")
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    
    # Enable CORS for communication with the frontend (port 5173 / localhost)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register API blueprint
    app.register_blueprint(ecg_bp, url_prefix="/api")
    
    # Root route to serve the frontend
    @app.route("/")
    def index():
        return app.send_static_file("index.html")
        
    return app

if __name__ == "__main__":
    app = create_app()
    # Run the server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
