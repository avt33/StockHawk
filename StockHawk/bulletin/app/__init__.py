from flask import Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail

# Initialize database and mail
db = SQLAlchemy()
login_manager = LoginManager()
mail = Mail()

def create_bulletin_blueprint():
    bulletin_bp = Blueprint(
        'bulletin',
        __name__,
        template_folder='templates',
        static_folder='static'
    )

    # Register Blueprints for views and auth
    from .views import views
    from .auth import auth
    bulletin_bp.register_blueprint(views, url_prefix='/')
    bulletin_bp.register_blueprint(auth, url_prefix='/auth')

    return bulletin_bp