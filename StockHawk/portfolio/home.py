# portfolio/home.py
from flask import Blueprint, render_template
from .DCF import DCF_bp
from .monte_carlo import monte_carlo_bp

portfolio_bp = Blueprint('portfolio', __name__, template_folder='templates', static_folder='static')

# Register blueprints for DCF and Monte Carlo
portfolio_bp.register_blueprint(DCF_bp, url_prefix='/DCF')
portfolio_bp.register_blueprint(monte_carlo_bp, url_prefix='/monte_carlo')

@portfolio_bp.route("/portfolio")
def home():
    return render_template("index.html")
