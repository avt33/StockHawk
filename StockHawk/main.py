from flask import Flask, redirect, url_for
from bulletin.app import create_bulletin_blueprint, db, login_manager, mail
from bulletin.app.models import User
from portfolio.home import portfolio_bp  # Import the portfolio blueprint
from flask_login import LoginManager

def create_app():
    app = Flask(__name__, static_folder='bulletin/app/static')

    # Load configuration
    app.config.from_mapping(
        SECRET_KEY='HFX@5971&',  
        SQLALCHEMY_DATABASE_URI='sqlite:///database.db',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,

        # Flask-Mail Configuration
        MAIL_SERVER='smtp.gmail.com',
        MAIL_PORT=587,
        MAIL_USE_TLS=True,
        MAIL_USERNAME='your_email@gmail.com',  # Use environment variables for security
        MAIL_PASSWORD='your_email_password',  # Use environment variables for security
        MAIL_DEFAULT_SENDER='your_email@gmail.com'
    )

    # Initialize Flask extensions
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)  # Initialize Flask-Mail

    # Set up the login_view for Flask-Login
    login_manager.login_view = 'auth.login'  # Ensure this matches the blueprint name

    # Register blueprints
    app.register_blueprint(create_bulletin_blueprint())
    app.register_blueprint(portfolio_bp)

    @app.route('/')
    def home():
        return redirect(url_for('portfolio.home'))  # Redirect to the portfolio home route

    return app

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create and run the app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
