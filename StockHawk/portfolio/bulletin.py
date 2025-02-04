from flask import Blueprint, render_template, request

bulletin_bp = Blueprint('bulletin', __name__)

@bulletin_bp.route('/bulletin')
def bulletin():
    return render_template('bulletin.html')