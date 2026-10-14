from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import urllib.parse
import requests

app = Flask(__name__)
app.secret_key = 'hanisha_vit_secure_key_2026'

password = "Krishnahani@1"
encoded_password = urllib.parse.quote_plus(password)
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres:{encoded_password}@localhost:5432/academic_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)


class PredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    burnout = db.Column(db.String(50))
    mental_health = db.Column(db.String(50))
    productivity = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


with app.app_context():
    db.create_all()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


model = load_model("models/multitask_model.h5", compile=False)
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")
burnout_le = encoders['burnout']
mental_le = encoders['mental']


@app.route("/")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials! Try again.")
    return render_template("login.html", register_mode=False)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("User already exists!")
            return redirect(url_for("login"))
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("Account created! Please Login.")
        return redirect(url_for("login"))
    return render_template("login.html", register_mode=True)


@app.route("/dashboard")
@login_required
def dashboard():
    latest = PredictionRecord.query.filter_by(user_id=session['user_id']).order_by(
        PredictionRecord.timestamp.desc()).first()
    level_map = {"Low": 25, "Medium": 60, "High": 85}
    burnout = latest.burnout if latest else "Not Calculated"
    mental = latest.mental_health if latest else "Not Calculated"
    productivity = latest.productivity if latest else 0.0
    return render_template(
        "dashboard.html",
        username=session.get('username'),
        burnout=burnout,
        mental=mental,
        productivity=productivity,
        burnout_value=level_map.get(burnout, 0),
        mental_value=level_map.get(mental, 0)
    )


@app.route("/prediction")
@login_required
def prediction_page():
    return render_template("prediction.html")


@app.route("/learning_hub")
@login_required
def learning_hub():
    return render_template("learning_hub.html")


@app.route("/tech_news")
@login_required
def tech_news():
    category = request.args.get('category', 'artificial intelligence')

    mock_articles = [
        {
            "title": "OpenAI Sora: The Future of Text-to-Video is Here",
            "description": "Discover how Sora is transforming the creative industry with high-fidelity video generation.",
            "source": {"name": "AI Insights"},
            "url": "https://openai.com/sora",
            "publishedAt": "2026-04-04"
        },
        {
            "title": "NVIDIA Blackwell: The World's Most Powerful AI Chip",
            "description": "NVIDIA reveals the next generation of GPU architecture designed for trillion-parameter LLMs.",
            "source": {"name": "Hardware Zone"},
            "url": "#",
            "publishedAt": "2026-04-03"
        },
        {
            "title": "Python 3.13: No-GIL and Performance Boosts",
            "description": "The latest Python update brings significant speed improvements for data science workflows.",
            "source": {"name": "Dev Weekly"},
            "url": "#",
            "publishedAt": "2026-04-02"
        },
        {
            "title": "Apple M4 Chip Leaks: Enhanced Neural Engine for AI",
            "description": "Rumors suggest the upcoming M4 chips will focus heavily on local AI processing power.",
            "source": {"name": "Mac Rumors"},
            "url": "#",
            "publishedAt": "2026-04-01"
        },
        {
            "title": "The Rise of Agentic AI in Enterprise",
            "description": "How autonomous AI agents are becoming the new standard for business automation.",
            "source": {"name": "Tech Crunch"},
            "url": "#",
            "publishedAt": "2026-04-04"
        }
    ]

    api_key = "675eb36452cd45fc8d93ab676b6b65f5"
    url = f"https://newsapi.org/v2/everything?q={category}&sortBy=publishedAt&language=en&apiKey={api_key}"

    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            articles = response.json().get("articles", [])[:12]
            if not articles:
                articles = mock_articles
        else:
            articles = mock_articles
    except:
        articles = mock_articles

    return render_template("tech_news.html", articles=articles, current_category=category)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        attendance = float(request.form.get("Attendance_Rate", 75))
        courses = float(request.form.get("Online_Courses_Completed", 0))
        study = float(request.form.get("Study_Hours", 5))
        sleep = float(request.form.get("Sleep_Hours", 7))
        part = 1 if request.form.get("Participation") == "Yes" else 0

        data_values = [21, study, courses, part, 85, 75, attendance, 1, 5, sleep, 2, 10, 60]
        cols = [
            'Age', 'Study_Hours_per_Week', 'Online_Courses_Completed',
            'Participation_in_Discussions', 'Assignment_Completion_Rate (%)',
            'Exam_Score (%)', 'Attendance_Rate (%)', 'Use_of_Educational_Tech',
            'Time_Spent_on_Social_Media (hours/week)', 'Sleep_Hours_per_Night',
            'Stress_Num', 'Burnout_Score', 'Productivity_Score'
        ]

        df = pd.DataFrame([data_values], columns=cols)
        scaled = scaler.transform(df)
        preds = model.predict(scaled)

        burnout_res = burnout_le.inverse_transform([np.argmax(preds[0])])[0]
        mental_res = mental_le.inverse_transform([np.argmax(preds[1])])[0]
        prod_score = max(0, min(100, round(float(preds[2][0][0]), 2)))

        new_record = PredictionRecord(
            user_id=session['user_id'],
            burnout=burnout_res,
            mental_health=mental_res,
            productivity=prod_score
        )
        db.session.add(new_record)
        db.session.commit()

        return redirect(url_for("dashboard"))

    except Exception as e:
        return f"Prediction Error: {e}"


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)