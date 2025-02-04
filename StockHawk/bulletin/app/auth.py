from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Message
from .models import User
from . import db, mail  # Assuming Flask-Mail is set up

auth = Blueprint('auth', __name__)

SECRET_KEY = "HFX@5971&"  # Store securely (env variable)

# Serializer for email verification tokens
serializer = URLSafeTimedSerializer(SECRET_KEY)

def send_verification_email(user_email):
    token = serializer.dumps(user_email, salt='email-confirmation')
    confirm_url = url_for('bulletin.auth.confirm_email', token=token, _external=True)
    
    msg = Message("Confirm Your Email", sender="noreply@yourapp.com", recipients=[user_email])
    msg.body = f"Click the link to verify your email: {confirm_url}"
    
    mail.send(msg)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        password = request.form.get('password').strip()
        first_name = request.form.get('first_name').strip()
        username = request.form.get('username').strip()
        secret_key = request.form.get('secret_key', '').strip()

        # Validation
        if not email or not password or not first_name or not username:
            flash('All fields are required.', 'danger')
            return redirect(url_for('bulletin.auth.register'))
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('bulletin.auth.register'))
        if User.query.filter_by(email=email).first():
            flash('Email is already registered.', 'danger')
            return redirect(url_for('bulletin.auth.register'))
        if User.query.filter_by(username=username).first():
            flash('Username is already taken.', 'danger')
            return redirect(url_for('bulletin.auth.register'))

        is_admin = secret_key == SECRET_KEY  # Admin check

        # Store hashed password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(email=email, password=hashed_password, first_name=first_name, username=username, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()

        send_verification_email(email)  # Send email verification

        flash('Registration successful! Check your email to verify your account.', 'success')
        return redirect(url_for('bulletin.auth.login'))

    return render_template('register.html')

@auth.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = serializer.loads(token, salt='email-confirmation', max_age=3600)  # 1-hour expiry
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Invalid token.', 'danger')
            return redirect(url_for('bulletin.auth.register'))

        flash('Email verified! You can now log in.', 'success')
        return redirect(url_for('bulletin.auth.login'))
    except:
        flash('The confirmation link has expired or is invalid.', 'danger')
        return redirect(url_for('bulletin.auth.register'))

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        password = request.form.get('password').strip()
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('bulletin.views.home'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('bulletin.auth.login'))

    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('bulletin.auth.login'))
