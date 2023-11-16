from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,send_from_directory,send_file,Response
from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length,ValidationError,InputRequired
from flask_bcrypt import Bcrypt
from sqlalchemy import inspect
from ultralytics import YOLO

import inference
import tempfile
import numpy as np
import base64
import os
import cv2
from ultralytics import YOLO
import supervision as sv
from roboflow import Roboflow
import MySQLdb
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import cv2
import requests
import supervision as sv
import pickle
import os
from typing import List

rf = Roboflow(api_key="06Fw6PZkF1NWPjvmWIDK")
project = rf.workspace().project("live-road-detection")
model = project.version(6).model

app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:#1ntelligent1901@localhost:3306/users'
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a secret key for session security
app.config['UPLOAD_FOLDER'] = "C:\\Users\\amirt\\Downloads\\app\\annotated"
db = SQLAlchemy(app)

app.add_url_rule('/annotated_images/<filename>', 'annotated_image', build_only=True)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

def create_database():
    with app.app_context():
        inspector = inspect(db.engine)
        if not inspector.has_table(User.__tablename__):
            db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model,UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

@app.route('/')
def index():
    form = RegisterForm()
    return render_template('index.html',form=form)

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Username"})
    email = StringField(validators=[InputRequired(),Email()],render_kw={"placeholder":"Email"})
    password = PasswordField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Password"})
    submit = SubmitField("Register")

    def validate_username(self,username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                "That username already exists. Please choose a different one."
            )
        
class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Username"})
    password = PasswordField(validators=[InputRequired(),Length(min=4,max=20)],render_kw={"placeholder":"Password"})
    submit = SubmitField("Login")

@app.route('/signup', methods=["GET", "POST"])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("index"))
    elif request.method == "GET":
        # Render the registration form for GET requests
        return render_template("index.html", form=form)
    # For POST requests with form validation errors, the form will be rendered again
    return render_template("index.html", form=form)

@app.route('/signin', methods=["GET", "POST"])
def signin():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for("upload"))
    return render_template("login.html", form=form)


@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html")

@app.route('/selection', methods=['POST'])
def selection():
    # Add logic based on the button clicked (not implemented in this example)
    button_clicked = request.form.get('button_clicked')
    
    # Redirect to the corresponding page
    if button_clicked == 'image':
        return render_template('image.html')
    elif button_clicked == 'video':
        return render_template('video.html')
    elif button_clicked == 'camera':
        # Add logic for camera
        pass
    else:
        # Handle other cases or redirect to the home page
        return redirect('/')

@app.route('/predict', methods=["POST"])
def predict_and_annotate():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    else:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = model.predict(file_path, confidence=40, overlap=30).json()
        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_roboflow(result)

        image = cv2.imread(file_path)
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoundingBoxAnnotator()

        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
        cv2.imwrite(annotated_image_path, annotated_image)

        print("Original Image Path:", file_path)
        print("Annotated Image Path:", annotated_image_path)
        with open(annotated_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    return jsonify({'success': True, 'annotated_image': base64_image})

@app.route("/annotate_upload", methods=["POST"])
def process_video():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    model = YOLO("best.pt")
    results = model.predict(file_path,save=True, project="runs", name="detect")
    annotated_video_path = "runs\detect\" + file_path
    return send_file(annotated_video_path, mimetype='video/mp4', as_attachment=True)


if __name__ == '__main__':
    create_database()
    app.run(debug=True)

