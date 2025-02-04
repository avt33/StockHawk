from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user, logout_user
from .models import User, Folder, Post, db
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import os

views = Blueprint('views', __name__)
mail = Mail()  # Add Flask-Mail setup in your app initialization

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app', 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@views.route('/bulletin-home', methods=['GET'])
@login_required
def home():
    search_query = request.args.get('search', '')
    folder_filter = request.args.get('folder', '')

    posts = Post.query
    if search_query:
        posts = posts.filter(Post.body.contains(search_query) | Post.tags.contains(search_query))
    if folder_filter:
        posts = posts.filter_by(folder_id=folder_filter)

    posts = posts.order_by(Post.date_created.desc()).all()
    folders = Folder.query.all()

    return render_template('home.html', user=current_user, posts=posts, folders=folders)
    
@views.route('/admin/manage_folders', methods=['GET', 'POST'])
@login_required
def manage_folders():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', category='error')
        return redirect(url_for('bulletin.views.home'))

    if request.method == 'POST':
        folder_name = request.form.get('folder_name')
        if Folder.query.filter_by(name=folder_name).first():
            flash('Folder already exists.', category='error')
        else:
            folder = Folder(name=folder_name)
            db.session.add(folder)
            db.session.commit()
            flash('Folder created!', category='success')

    folders = Folder.query.all()
    return render_template('manage_folders.html', user=current_user, folders=folders)

@views.route('/post/<int:post_id>', methods=['GET'])
@login_required
def view_post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('view_post.html', post=post)

@views.route('/admin/create_post', methods=['GET', 'POST'])
@login_required
def create_post():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', category='error')
        return redirect(url_for('bulletin.views.home'))
    
    folders = Folder.query.all()

    if request.method == 'POST':
        title = request.form.get('title')
        tags = request.form.get('tags')
        body = request.form.get('body')
        folder_id = request.form.get('folder')

        if not title or not body or not folder_id:
            flash('Title, body, & folder are required.', category='error')
        else:
            post = Post(title=title, body=body, tags=tags, folder_id=folder_id, author_id=current_user.id)
            db.session.add(post)
            db.session.commit()

            flash('Post created successfully!', category='success')
            return redirect(url_for('bulletin.views.home'))

    return render_template('create_post.html', user=current_user, folders=folders)

@views.route('/admin/delete_folder/<int:folder_id>', methods=['POST'])
@login_required
def delete_folder(folder_id):
    if not current_user.is_admin:
        flash("You do not have permission to do this.", category='error')
        return redirect(url_for('bulletin.views.home'))
    
    folder = Folder.query.get_or_404(folder_id)
    if folder.posts:
        flash('Cannot delete with posts. Move or delete posts first.', category='error')
    else:
        db.session.delete(folder)
        db.session.commit()
        flash('Folder deleted successfully!', category='success')
    
    return redirect(url_for('bulletin.views.manage_folders'))

@views.route('/admin/delete_post/<int:post_id>', methods=['POST'])
@login_required
def delete_post(post_id):
    if not current_user.is_admin:
        flash("You don't have permission to do this.", category='error')
        return redirect(url_for('bulletin.views.home'))
    
    post = Post.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    flash('Post deleted successfully!', category='success')
    return redirect(url_for('bulletin.views.home'))

@views.route('/logout')
def logout():
    logout_user()
    return redirect('portfolio')

@views.route('/upload-image', methods=['POST'])
@login_required
def upload_image():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        if 'upload' not in request.files:
            return jsonify({
                'error': {
                    'message': 'No file uploaded'
                }
            }), 400
            
        file = request.files['upload']
        
        if file.filename == '':
            return jsonify({
                'error': {
                    'message': 'No file selected'
                }
            }), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Create upload directory if it doesn't exist
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
                
            # Save the file
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Return the URL in the format CKEditor expects
            url = url_for('static', filename=f'uploads/{filename}')
            print("Image uploaded successfully.")

            return jsonify({
                'url': url,
                'uploaded': True
            })
        
        return jsonify({
            'error': {
                'message': 'Invalid file type'
            }
        }), 400
        
    except Exception as e:
        print(f"Upload error: {str(e)}")  # For debugging
        return jsonify({
            'error': {
                'message': 'Server error during upload'
            }
        }), 500