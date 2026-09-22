"""
Fake News Detection Flask Web Application.
Academic SDG Project (SDG 16: Peace, Justice, and Strong Institutions).

Individual Portal Authentication System:
- Dedicated Login Portals:
  1. User Portal ('/login/user'): General citizens & researchers analyzing news.
  2. Content Writer Portal ('/login/writer'): Editorial staff moderating low-confidence (<60%) claims.
  3. Administrator Portal ('/login/admin'): System admins monitoring model performance and analytics.
- Pre-seeded Demo Accounts for immediate evaluation:
  - Admin:  admin  / admin123
  - Writer: writer / writer123
  - User:   user   / user123
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ml_model.predict import predict_news
from services.rss_service import get_live_news_feed

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sdg-fake-news-detector-secret-key-2026-auth'

DB_PATH = os.path.join(CURRENT_DIR, "app_data.db")
METRICS_PATH = os.path.join(CURRENT_DIR, "ml_model", "saved_models", "model_metrics.json")


def get_db_connection():
    """Returns a connection to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite tables for users, predictions, and moderation queue."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            title TEXT,
            text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            raw_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            real_probability REAL NOT NULL,
            fake_probability REAL NOT NULL,
            is_needs_review INTEGER NOT NULL,
            writer_status TEXT DEFAULT 'PENDING',
            human_verdict TEXT DEFAULT NULL,
            verdict_display TEXT NOT NULL,
            badge_class TEXT NOT NULL,
            explanation TEXT,
            source_type TEXT DEFAULT 'User Submission',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Column migration safety check
    cursor.execute("PRAGMA table_info(predictions)")
    existing_cols = {row['name'] for row in cursor.fetchall()}
    if 'user_id' not in existing_cols:
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
        except Exception:
            pass
    if 'username' not in existing_cols:
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN username TEXT")
        except Exception:
            pass
    if 'reviewed_by' not in existing_cols:
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN reviewed_by TEXT")
        except Exception:
            pass
    if 'reviewed_at' not in existing_cols:
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN reviewed_at TIMESTAMP")
        except Exception:
            pass
    if 'editorial_notes' not in existing_cols:
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN editorial_notes TEXT")
        except Exception:
            pass

    # Seed Default Accounts (Admin, Writer, User)
    default_users = [
        ("admin", "admin123", "admin"),
        ("writer", "writer123", "writer"),
        ("user", "user123", "user")
    ]

    for uname, pword, role in default_users:
        cursor.execute("SELECT id FROM users WHERE username = ?", (uname,))
        if not cursor.fetchone():
            p_hash = generate_password_hash(pword)
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (uname, p_hash, role)
            )

    # Ensure only FAKE news has writer_status='PENDING'
    cursor.execute("""
        UPDATE predictions 
        SET writer_status = 'NOT_REQUIRED', is_needs_review = 0 
        WHERE (raw_label = 'REAL' OR prediction = 'REAL') AND human_verdict IS NULL AND writer_status = 'PENDING'
    """)

    # Populate seed predictions if table is empty
    cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
    count = cursor.fetchone()['cnt']
    if count == 0:
        seed_samples = [
            (
                1, "admin",
                "ISRO prepares Gaganyaan crew escape module for next high-altitude test",
                "ISRO confirmed all propulsion thrusters for the crew escape test vehicle met performance targets at Sriharikota.",
                "REAL", "REAL", 89.5, 0.895, 0.105, 0, "RESOLVED", "REAL", "Likely Real News", "success",
                "The phrasing matches verified scientific and space reporting patterns.", "System Verification"
            ),
            (
                2, "writer",
                "WhatsApp forward claims boiling raw betel leaves cures all virus mutations in 3 hours",
                "Suppressed medical secret! Forward this message to all family members immediately to prevent hospital visits.",
                "FAKE", "FAKE", 91.2, 0.088, 0.912, 0, "RESOLVED", "FAKE", "Potentially Fake News", "danger",
                "Sensational phrasing, urgent forward commands, and unsubstantiated health claims detected.", "System Verification"
            ),
            (
                3, "user",
                "Municipal council reviews coastal drainage proposal for upcoming monsoon season",
                "The municipal drainage committee met yesterday at the town hall to review canal desilting tenders.",
                "NEEDS_REVIEW", "REAL", 53.4, 0.534, 0.466, 1, "PENDING", None, "Needs Human Review", "warning",
                "Confidence is 53.4% (<60%). Flagged for editorial fact-checker verification.", "Live Feed Ingestion"
            ),
            (
                1, "admin",
                "Viral post alleges new currency notes contain satellite tracking GPS microchips",
                "Leaked memo says central bank embedded microscopic radio transmitters to detect currency from space.",
                "FAKE", "FAKE", 88.0, 0.120, 0.880, 0, "RESOLVED", "FAKE", "Potentially Fake News", "danger",
                "Contains well-documented viral financial hoax markers.", "System Verification"
            ),
            (
                3, "user",
                "State electricity board announces planned maintenance shutdown in select city zones",
                "Substation maintenance work will temporarily disrupt power in northern sectors between 9 AM and 1 PM tomorrow.",
                "NEEDS_REVIEW", "REAL", 56.1, 0.561, 0.439, 1, "PENDING", None, "Needs Human Review", "warning",
                "Confidence is 56.1% (<60%). Requires local bureau fact-checking confirmation.", "User Submission"
            )
        ]

        for s in seed_samples:
            cursor.execute("""
                INSERT INTO predictions 
                (user_id, username, title, text, prediction, raw_label, confidence, real_probability, fake_probability,
                 is_needs_review, writer_status, human_verdict, verdict_display, badge_class, explanation, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, s)

    conn.commit()
    conn.close()


# Initialize database and tables
init_db()


def load_model_metrics():
    """Loads model training metadata and tuning results for Admin Dashboard."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "model_type": "Logistic Regression (L2 Regularized)",
        "optimal_C": 1.0,
        "max_features": 5000,
        "best_metrics": {
            "train_accuracy": 1.0,
            "test_accuracy": 1.0,
            "generalization_gap": 0.0,
            "test_f1_fake": 1.0
        }
    }


# ==============================================================================
# AUTHENTICATION HELPERS & DECORATORS
# ==============================================================================
def login_required(f):
    """Requires user to be logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login_portal_view', role='user', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def role_required(allowed_roles):
    """Restricts view to specific roles (e.g. ['admin', 'writer'])."""
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                target_portal = allowed_roles[0] if allowed_roles else 'user'
                flash("Please log in with appropriate credentials to access this page.", "warning")
                return redirect(url_for('login_portal_view', role=target_portal, next=request.url))
            
            user_role = session.get('user_role', 'user')
            if user_role not in allowed_roles:
                flash(f"Access Denied: Your account role is '{user_role.upper()}', but this page requires '{'/'.join(allowed_roles).upper()}'.", "danger")
                return redirect(url_for('user_view'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@app.context_processor
def inject_user():
    """Injects current logged-in user information into all templates."""
    return {
        'current_user': {
            'is_authenticated': 'user_id' in session,
            'id': session.get('user_id'),
            'username': session.get('username'),
            'role': session.get('user_role', 'guest')
        }
    }


# ==============================================================================
# INDIVIDUAL LOGIN PORTALS (User Login, Content Writer Login, Admin Login)
# ==============================================================================
PORTAL_CONFIG = {
    'user': {
        'name': 'Citizen & User Portal',
        'title': 'User Login',
        'subtitle': 'Sign in to analyze articles & browse verified news feeds',
        'badge': 'USER ACCESS',
        'badge_class': 'success',
        'icon': 'fa-solid fa-user-check',
        'default_user': 'user',
        'default_pass': 'user123',
        'redirect': 'user_view'
    },
    'writer': {
        'name': 'Editorial & Content Writer Portal',
        'title': 'Content Writer Login',
        'subtitle': 'Sign in to moderate flagged claims and assign fact-check verdicts',
        'badge': 'EDITORIAL ACCESS',
        'badge_class': 'warning',
        'icon': 'fa-solid fa-user-pen',
        'default_user': 'writer',
        'default_pass': 'writer123',
        'redirect': 'writer_view'
    },
    'admin': {
        'name': 'System Administrator Portal',
        'title': 'Admin Login',
        'subtitle': 'Sign in to access analytics, ML model metrics & user accounts',
        'badge': 'ADMIN SECURE ACCESS',
        'badge_class': 'danger',
        'icon': 'fa-solid fa-shield-halved',
        'default_user': 'admin',
        'default_pass': 'admin123',
        'redirect': 'admin_view'
    }
}


@app.route("/login")
@app.route("/login", endpoint="login_view")
def login_default():
    """Redirects generic /login to the user portal."""
    target_role = request.args.get('role', 'user')
    if target_role not in PORTAL_CONFIG:
        target_role = 'user'
    return redirect(url_for('login_portal_view', role=target_role, next=request.args.get('next')))


@app.route("/login/<role>", methods=["GET", "POST"])
def login_portal_view(role):
    """
    Dedicated Individual Login Portals for:
    - /login/user   -> User Portal
    - /login/writer -> Editorial Writer Portal
    - /login/admin  -> System Administrator Portal
    """
    if role not in PORTAL_CONFIG:
        return redirect(url_for('login_portal_view', role='user'))

    config = PORTAL_CONFIG[role]
    next_page = request.args.get('next')

    # If already logged in, redirect to appropriate destination
    if 'user_id' in session:
        user_role = session.get('user_role', 'user')
        if role == 'admin' and user_role == 'admin':
            return redirect(url_for('admin_view'))
        elif role == 'writer' and user_role in ['writer', 'admin']:
            return redirect(url_for('writer_view'))
        else:
            return redirect(url_for('user_view'))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            user_role = user["role"]

            # Role validation for specific portal
            if role == "admin" and user_role != "admin":
                flash("Access Denied: This account does not have Administrator privileges. Please use the User or Writer Login portal.", "danger")
                return render_template("login.html", portal_role=role, portal=config, next=next_page)
            
            if role == "writer" and user_role not in ["writer", "admin"]:
                flash("Access Denied: This account does not have Content Writer privileges. Please use the User Login portal.", "danger")
                return render_template("login.html", portal_role=role, portal=config, next=next_page)

            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']
            flash(f"Welcome back, {user['username']}! Logged in to {config['name']}.", "success")

            # Redirect according to role & portal
            if next_page:
                return redirect(next_page)
            elif role == 'admin':
                return redirect(url_for('admin_view'))
            elif role == 'writer':
                return redirect(url_for('writer_view'))
            else:
                return redirect(url_for('user_view'))
        else:
            flash("Invalid username or password for this portal. Please try again.", "danger")

    return render_template("login.html", portal_role=role, portal=config, next=next_page)


@app.route("/api/demo-login/<role>", methods=["POST"])
def api_demo_login(role):
    """One-click demo login for academic evaluation (admin, writer, user)."""
    if role not in ["admin", "writer", "user"]:
        return jsonify({"success": False, "error": "Invalid role."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (role,))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['user_role'] = user['role']
        
        redirect_map = {
            'admin': url_for('admin_view'),
            'writer': url_for('writer_view'),
            'user': url_for('user_view')
        }
        return jsonify({
            "success": True, 
            "username": user['username'],
            "role": user['role'],
            "redirect_url": redirect_map.get(role, url_for('user_view'))
        })

    return jsonify({"success": False, "error": "Demo user not found."}), 404


@app.route("/register", methods=["GET", "POST"])
def register_view():
    """Allows new users to register."""
    if 'user_id' in session:
        return redirect(url_for('user_view'))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "user").strip()

        if role not in ["user", "writer", "admin"]:
            role = "user"

        if not username or not password:
            flash("Please enter both username and password.", "warning")
            return render_template("register.html")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            flash("Username is already taken. Please choose another.", "danger")
            return render_template("register.html")

        p_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, p_hash, role)
        )
        conn.commit()
        conn.close()

        flash(f"Account '{username}' created successfully as {role.upper()}! Please log in through the {role.upper()} portal.", "success")
        return redirect(url_for('login_portal_view', role=role))

    return render_template("register.html")


@app.route("/logout")
def logout_view():
    """Logs out the current user."""
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for('login_portal_view', role='user'))


@app.route("/report")
def report_view():
    """Renders the comprehensive Veritas Test Report, Credentials & Project Roadmap."""
    return render_template("test_report_and_roadmap.html")


# ==============================================================================
# ROUTE 1: USER VIEW (Submission Form + Live Predictions + Personal History & Resolutions + Live RSS)
# ==============================================================================
@app.route("/")
@app.route("/user")
def user_view():
    """
    Main User Interface:
    Allows user to analyze custom news articles, track their submitted claims and editorial resolutions,
    and browses live Malayala Manorama / Indian news feed.
    """
    live_news = get_live_news_feed(limit=6)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    user_id = session.get('user_id')
    username = session.get('username')
    
    if user_id:
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE user_id = ? OR username = ? 
            ORDER BY id DESC LIMIT 20
        """, (user_id, username))
    else:
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE user_id IS NULL OR username = 'Guest User'
            ORDER BY id DESC LIMIT 10
        """)
    user_history = cursor.fetchall()
    conn.close()

    return render_template(
        "user_view.html", 
        live_news=live_news, 
        user_history=user_history, 
        active_tab="user"
    )


@app.route("/api/user/history", methods=["GET"])
def api_user_history():
    """
    Returns the recent submission history and latest editorial resolution status
    for the current session/user.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    user_id = session.get('user_id')
    username = session.get('username')
    
    if user_id:
        cursor.execute("""
            SELECT id, title, text, prediction, raw_label, confidence, real_probability, fake_probability,
                   is_needs_review, writer_status, human_verdict, verdict_display, badge_class, explanation,
                   reviewed_by, reviewed_at, editorial_notes, created_at, username
            FROM predictions 
            WHERE user_id = ? OR username = ? 
            ORDER BY id DESC LIMIT 20
        """, (user_id, username))
    else:
        cursor.execute("""
            SELECT id, title, text, prediction, raw_label, confidence, real_probability, fake_probability,
                   is_needs_review, writer_status, human_verdict, verdict_display, badge_class, explanation,
                   reviewed_by, reviewed_at, editorial_notes, created_at, username
            FROM predictions 
            WHERE user_id IS NULL OR username = 'Guest User'
            ORDER BY id DESC LIMIT 10
        """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"success": True, "history": rows})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    AJAX Endpoint to analyze article text and record the prediction.
    Accessible to guests or logged-in users.
    If the news is predicted FAKE, it is queued for Content Writer review.
    """
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    title = data.get("title", "").strip()

    if not text:
        return jsonify({"success": False, "error": "Please enter article text or a headline."}), 400

    # Call ML Prediction engine
    result = predict_news(text=text, title=title)

    user_id = session.get('user_id')
    username = session.get('username', 'Guest User')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if this exact text/headline already has an authoritative editorial resolution
    cursor.execute("""
        SELECT human_verdict, verdict_display, explanation, reviewed_by, reviewed_at, editorial_notes
        FROM predictions 
        WHERE (text = ? OR (title = ? AND title != '')) AND writer_status = 'RESOLVED'
        ORDER BY id DESC LIMIT 1
    """, (text, title))
    prior_editorial = cursor.fetchone()

    if prior_editorial:
        human_v = prior_editorial["human_verdict"]
        rev_by = prior_editorial["reviewed_by"] or "Editorial Staff"
        notes = prior_editorial["editorial_notes"] or ""

        result["writer_status"] = "RESOLVED"
        result["human_verdict"] = human_v
        result["reviewed_by"] = rev_by
        result["reviewed_at"] = prior_editorial["reviewed_at"]
        result["editorial_notes"] = notes
        result["is_needs_review"] = False

        # Authoritative human verdict takes precedence in the final verdict
        result["prediction"] = human_v
        result["badge_class"] = "success" if human_v == "REAL" else "danger"
        result["verdict_display"] = f"Verified {human_v.capitalize()} (Editorial Review by {rev_by})"
        result["explanation"] = f"Authoritative human editorial verdict by {rev_by}: {human_v.capitalize()}." + (f" Editorial Note: {notes}" if notes else "")
        
        writer_status_val = "RESOLVED"
        human_verdict_val = human_v
        reviewed_by_val = rev_by
        editorial_notes_val = notes
        is_needs_rev_val = 0
    else:
        # ONLY news predicted as FAKE is flagged for Content Writer review
        is_fake = bool(result["prediction"] == "FAKE")
        
        if is_fake:
            is_needs_rev_val = 1
            writer_status_val = "PENDING"
            result["writer_status"] = "PENDING"
            result["is_needs_review"] = True
        else:
            is_needs_rev_val = 0
            writer_status_val = "NOT_REQUIRED"
            result["writer_status"] = "NOT_REQUIRED"
            result["is_needs_review"] = False

        human_verdict_val = None
        reviewed_by_val = None
        editorial_notes_val = None
        result["human_verdict"] = None
        result["reviewed_by"] = None
        result["editorial_notes"] = None

    # Persist to database
    cursor.execute("""
        INSERT INTO predictions 
        (user_id, username, title, text, prediction, raw_label, confidence, real_probability, fake_probability,
         is_needs_review, writer_status, human_verdict, reviewed_by, reviewed_at, editorial_notes,
         verdict_display, badge_class, explanation, source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
    """, (
        user_id,
        username,
        title or (text[:80] + "..."),
        text,
        result["prediction"],
        result["raw_label"],
        result["confidence"],
        result["real_probability"],
        result["fake_probability"],
        is_needs_rev_val,
        writer_status_val,
        human_verdict_val,
        reviewed_by_val,
        editorial_notes_val,
        result["verdict_display"],
        result["badge_class"],
        result["explanation"],
        f"User Submission ({username})"
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    result["id"] = record_id
    result["username"] = username
    return jsonify({"success": True, "data": result})


# ==============================================================================
# ROUTE 2: CONTENT WRITER VIEW (Editorial Moderation Queue for Fake News Only)
# ==============================================================================
@app.route("/writer")
@role_required(["writer", "admin"])
def writer_view():
    """
    Content Writer Moderation Interface:
    Restricted to 'writer' and 'admin' roles.
    Lists ONLY news articles classified as FAKE by the ML model for human fact-checking.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch only fake-classified claims (pending fake items first, followed by resolved items)
    cursor.execute("""
        SELECT * FROM predictions 
        WHERE (raw_label = 'FAKE' OR (writer_status = 'PENDING' AND is_needs_review = 1))
          AND writer_status != 'NOT_REQUIRED'
        ORDER BY CASE WHEN writer_status = 'PENDING' THEN 0 ELSE 1 END, id DESC
    """)
    flagged_items = cursor.fetchall()
    conn.close()

    return render_template(
        "writer_view.html", 
        flagged_items=flagged_items, 
        active_tab="writer"
    )


@app.route("/api/writer/review", methods=["POST"])
@role_required(["writer", "admin"])
def api_writer_review():
    """
    Endpoint for Content Writers / Admins to resolve a flagged claim.
    Saves reviewer identity, timestamp, and optional editorial reasoning notes.
    """
    data = request.get_json() or {}
    record_id = data.get("id")
    human_verdict = data.get("verdict")  # 'REAL' or 'FAKE'
    editorial_notes = data.get("notes", "").strip()

    if not record_id or human_verdict not in ["REAL", "FAKE"]:
        return jsonify({"success": False, "error": "Invalid review parameters."}), 400

    reviewer = session.get('username', 'Editorial Staff')
    badge_class = "success" if human_verdict == "REAL" else "danger"
    verdict_display = f"Verified {human_verdict.capitalize()} (Editorial Review by {reviewer})"
    
    explanation_parts = [f"Verified as {human_verdict.capitalize()} following editorial fact-check by {reviewer}."]
    if editorial_notes:
        explanation_parts.append(f"Editorial Note: {editorial_notes}")
    final_explanation = " ".join(explanation_parts)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch the target item text to update matching records
    cursor.execute("SELECT text FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    target_text = row["text"] if row else None

    cursor.execute("""
        UPDATE predictions 
        SET writer_status = 'RESOLVED',
            is_needs_review = 0,
            human_verdict = ?,
            prediction = ?,
            verdict_display = ?,
            badge_class = ?,
            explanation = ?,
            reviewed_by = ?,
            reviewed_at = CURRENT_TIMESTAMP,
            editorial_notes = ?
        WHERE id = ?
    """, (human_verdict, human_verdict, verdict_display, badge_class, final_explanation, reviewer, editorial_notes, record_id))

    # Also update any other identical claims in pending status
    if target_text:
        cursor.execute("""
            UPDATE predictions
            SET writer_status = 'RESOLVED',
                is_needs_review = 0,
                human_verdict = ?,
                prediction = ?,
                verdict_display = ?,
                badge_class = ?,
                explanation = ?,
                reviewed_by = ?,
                reviewed_at = CURRENT_TIMESTAMP,
                editorial_notes = ?
            WHERE text = ? AND writer_status = 'PENDING'
        """, (human_verdict, human_verdict, verdict_display, badge_class, final_explanation, reviewer, editorial_notes, target_text))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True, 
        "id": record_id, 
        "verdict": human_verdict, 
        "verdict_display": verdict_display, 
        "badge_class": badge_class, 
        "reviewed_by": reviewer, 
        "editorial_notes": editorial_notes
    })


# ==============================================================================
# ROUTE 3: ADMIN DASHBOARD (KPI Stat Cards + Model Metrics + User Accounts)
# ==============================================================================
@app.route("/admin")
@role_required(["admin"])
def admin_view():
    """
    Admin Dashboard Interface:
    Restricted to 'admin' role.
    - Real-time Stat Cards (Total Checked, Fake Detected, Real Detected, Pending Review)
    - Regularized Logistic Regression Model Metrics (Train/Test Accuracy, Regularization C)
    - Registered Users Table
    - SDG 16 Alignment Overview
    - Recent Prediction Activity Log with User Attribution
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Aggregate Statistics
    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total_checked = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as fakes FROM predictions WHERE prediction = 'FAKE' OR human_verdict = 'FAKE'")
    fake_detected = cursor.fetchone()['fakes']

    cursor.execute("SELECT COUNT(*) as reals FROM predictions WHERE prediction = 'REAL' OR human_verdict = 'REAL'")
    real_detected = cursor.fetchone()['reals']

    cursor.execute("SELECT COUNT(*) as pending FROM predictions WHERE is_needs_review = 1 AND writer_status = 'PENDING'")
    pending_review = cursor.fetchone()['pending']

    # Fetch all registered users
    cursor.execute("SELECT id, username, role, created_at FROM users ORDER BY id ASC")
    all_users = cursor.fetchall()

    # Fetch recent logs with user attribution & role
    cursor.execute("""
        SELECT p.*, COALESCE(u.role, CASE WHEN p.username = 'Guest User' THEN 'guest' ELSE 'user' END) as user_role 
        FROM predictions p 
        LEFT JOIN users u ON (p.user_id = u.id OR (p.user_id IS NULL AND p.username = u.username))
        ORDER BY p.id DESC LIMIT 30
    """)
    recent_logs = cursor.fetchall()
    conn.close()

    model_metadata = load_model_metrics()

    stats = {
        "total_checked": total_checked,
        "fake_detected": fake_detected,
        "real_detected": real_detected,
        "pending_review": pending_review
    }

    return render_template(
        "admin_view.html",
        stats=stats,
        model_meta=model_metadata,
        users_list=all_users,
        recent_logs=recent_logs,
        active_tab="admin"
    )


@app.route("/api/refresh-news", methods=["POST"])
def api_refresh_news():
    """Refreshes live news feed and runs automated predictions."""
    live_news = get_live_news_feed(limit=6)
    return jsonify({"success": True, "feed": live_news})


# ==============================================================================
# ROUTE 4: ADMIN DELETION & MANAGEMENT APIS
# ==============================================================================
@app.route("/api/admin/users/delete", methods=["POST"])
@role_required(["admin"])
def api_admin_delete_user():
    """
    Deletes a registered user account.
    Safeguard: Cannot delete the primary root admin account or currently active account.
    """
    data = request.get_json() or {}
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"success": False, "error": "User ID is required."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    if not target_user:
        conn.close()
        return jsonify({"success": False, "error": "Target user not found."}), 404

    target_uname = target_user["username"]

    # Safeguards: Prevent deleting 'admin' root account or currently logged-in account
    if target_uname == "admin":
        conn.close()
        return jsonify({"success": False, "error": "Cannot delete the primary root Administrator ('admin') account."}), 403

    if user_id == session.get("user_id"):
        conn.close()
        return jsonify({"success": False, "error": "Cannot delete your own currently active session account."}), 403

    # Delete user and disassociate/preserve prediction history
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    cursor.execute("UPDATE predictions SET user_id = NULL WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": f"User account '{target_uname}' was deleted successfully."})


@app.route("/api/admin/predictions/delete", methods=["POST"])
@role_required(["admin"])
def api_admin_delete_prediction():
    """
    Deletes an individual prediction/audit log entry from the database.
    """
    data = request.get_json() or {}
    pred_id = data.get("prediction_id")

    if not pred_id:
        return jsonify({"success": False, "error": "Prediction ID is required."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    if deleted == 0:
        return jsonify({"success": False, "error": "Prediction record not found."}), 404

    return jsonify({"success": True, "message": f"Prediction record #{pred_id} deleted successfully."})


@app.route("/api/admin/predictions/clear-all", methods=["POST"])
@role_required(["admin"])
def api_admin_clear_all_predictions():
    """
    Clears all prediction activity logs from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": f"Successfully cleared {deleted_count} prediction activity logs."})


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING FAKE NEWS DETECTION FLASK APP (SDG 16 ALIGNED)")
    print("=" * 70)
    print("User View:            http://127.0.0.1:5000/")
    print("Content Writer View:  http://127.0.0.1:5000/writer")
    print("Admin Dashboard:      http://127.0.0.1:5000/admin")
    print("User Login Portal:    http://127.0.0.1:5000/login/user")
    print("Writer Login Portal:  http://127.0.0.1:5000/login/writer")
    print("Admin Login Portal:   http://127.0.0.1:5000/login/admin")
    print("=" * 70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
