from main import create_app, db  # Adjust the import path if needed
from bulletin.app.models import User  # Import your User model

app = create_app()

with app.app_context():
    db.create_all()
    print("Database tables created.")
