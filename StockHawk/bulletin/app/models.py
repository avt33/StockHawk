# bulletin/app/models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy.sql import func
from . import db  # Import db instance correctly here

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    first_name = db.Column(db.String(150))
    is_admin = db.Column(db.Boolean, default=False)
    wants_email_notifications = db.Column(db.Boolean, default=True)

class Folder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    body = db.Column(db.Text, nullable=False)
    tags = db.Column(db.String(250))  # Tags as comma-separated values
    folder_id = db.Column(db.Integer, db.ForeignKey('folder.id'))
    folder = db.relationship('Folder', backref='posts')
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    author = db.relationship('User', backref='posts')
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())
