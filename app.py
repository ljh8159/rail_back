import os
import time
import sqlite3
import psycopg2
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from werkzeug.utils import secure_filename

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import hashlib
import secrets
from datetime import datetime, timedelta

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DATABASE = 'reports.db'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS 설정
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = 'mobilenetv2_stage_model.h5'
MODEL_IMG_SIZE = (224, 224)
model = load_model(MODEL_PATH)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = psycopg2.connect(os.environ.get("DATABASE_URL"))
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                type TEXT,
                photo_filename TEXT,
                location TEXT,
                lat REAL,
                lng REAL,
                timestamp TEXT,
                ai_stage INTEGER,
                extra TEXT,
                dispatch_user_id TEXT,
                for_userpage_type TEXT,
                for_userpage_stage INTEGER
            )
        ''')
        try:
            db.execute("ALTER TABLE reports ADD COLUMN dispatch_user_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE reports ADD COLUMN for_userpage_type TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE reports ADD COLUMN for_userpage_stage INTEGER")
        except sqlite3.OperationalError:
            pass
        db.commit()

def init_user_table():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                password TEXT
            )
        ''')
        db.commit()

init_db()
init_user_table()

@app.route('/')
def index():
    return "Flask 서버가 실행 중입니다."

@app.route('/reports')
def reports_page():
    return "신고/출동 API 서버입니다."

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload_photo', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # 원본 파일명 그대로 사용 (보안상 안전하게)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'filename': filename})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        img = image.load_img(file_path, target_size=MODEL_IMG_SIZE)
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        stage = int(np.argmax(preds)) + 1
        return jsonify({'stage': stage})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def save_report():
    data = request.get_json()
    user_id = data.get('user_id', 'guest')
    type_ = data.get('type', '')
    photo_filename = data.get('photo_filename')
    location = data.get('location')
    lat = data.get('lat')
    lng = data.get('lng')
    timestamp = data.get('timestamp')
    ai_stage = data.get('ai_stage')
    extra = data.get('extra', '')
    dispatch_user_id = data.get('dispatch_user_id', None)
    for_userpage_type = data.get('for_userpage_type', type_)
    for_userpage_stage = data.get('for_userpage_stage', ai_stage)

    # timestamp가 없거나 빈 값이면 한국시간(KST)으로 생성
    if not timestamp:
        kst = datetime.utcnow() + timedelta(hours=9)
        timestamp = kst.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+09:00'

    db = get_db()
    db.execute(
        '''INSERT INTO reports 
        (user_id, type, photo_filename, location, lat, lng, timestamp, ai_stage, extra, dispatch_user_id, for_userpage_type, for_userpage_stage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, type_, photo_filename, location, lat, lng, timestamp, ai_stage, extra, dispatch_user_id, for_userpage_type, for_userpage_stage)
    )
    db.commit()
    return jsonify({'result': 'success'})

@app.route('/api/report_update', methods=['POST'])
def update_report():
    data = request.get_json()
    location = data.get('location')
    dispatch_user_id = data.get('dispatch_user_id', None)
    db = get_db()
    db.execute(
        "UPDATE reports SET type='출동', ai_stage=1, dispatch_user_id=? WHERE location=? AND type='신고'",
        (dispatch_user_id, location)
    )
    db.commit()
    return jsonify({'result': 'updated'})

@app.route('/api/report_stats', methods=['GET'])
def report_stats():
    db = get_db()
    cur1 = db.execute("SELECT COUNT(*) FROM reports WHERE type='신고' AND ai_stage=3")
    blocked_count = cur1.fetchone()[0]
    cur2 = db.execute("SELECT COUNT(*) FROM reports WHERE type='출동' AND ai_stage=1")
    dispatched_count = cur2.fetchone()[0]
    return jsonify({
        "blocked_count": blocked_count,
        "dispatched_count": dispatched_count
    })

@app.route('/api/all_reports', methods=['GET'])
def all_reports():
    limit = request.args.get('limit', default=3, type=int)
    db = get_db()
    cur = db.execute(
        "SELECT type, location, timestamp FROM reports WHERE type='신고' AND ai_stage=3 "
        "UNION ALL "
        "SELECT type, location, timestamp FROM reports WHERE type='출동' AND ai_stage=1 "
        "ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    reports = []
    from datetime import datetime, timezone
    import math
    for row in cur.fetchall():
        try:
            t = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            diff = now - t
            seconds = diff.total_seconds()
            if seconds < 60:
                time_str = f"{int(seconds)}초 전"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                time_str = f"{minutes}분 전"
            elif seconds < 86400:
                hours = int(seconds // 3600)
                time_str = f"{hours}시간 전"
            elif diff.days < 30:
                time_str = f"{diff.days}일 전"
            else:
                months = math.floor(diff.days / 30)
                time_str = f"{months}달 전"
        except Exception:
            time_str = ""
        reports.append({
            "type": row["type"],
            "location": row["location"],
            "timestamp": row["timestamp"],
            "time": time_str
        })
    return jsonify(reports)

@app.route('/api/user_reports', methods=['GET'])
def user_reports():
    user_id = request.args.get('user_id', 'guest')
    limit = request.args.get('limit', default=3, type=int)
    db = get_db()
    cur = db.execute(
        "SELECT type, location, timestamp FROM reports WHERE user_id=? AND for_userpage_type='신고' AND for_userpage_stage=3 "
        "UNION ALL "
        "SELECT type, location, timestamp FROM reports WHERE dispatch_user_id=? AND type='출동' AND ai_stage=1 "
        "ORDER BY timestamp DESC LIMIT ?",
        (user_id, user_id, limit)
    )
    reports = []
    from datetime import datetime, timezone
    import math
    for row in cur.fetchall():
        try:
            t = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            diff = now - t
            seconds = diff.total_seconds()
            if seconds < 60:
                time_str = f"{int(seconds)}초 전"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                time_str = f"{minutes}분 전"
            elif seconds < 86400:
                hours = int(seconds // 3600)
                time_str = f"{hours}시간 전"
            elif diff.days < 30:
                time_str = f"{diff.days}일 전"
            else:
                months = math.floor(diff.days / 30)
                time_str = f"{months}달 전"
        except Exception:
            time_str = ""
        reports.append({
            "type": row["type"],
            "location": row["location"],
            "timestamp": row["timestamp"],
            "time": time_str
        })
    return jsonify(reports)

@app.route('/api/user_stats', methods=['GET'])
def user_stats():
    user_id = request.args.get('user_id', 'guest')
    db = get_db()
    # 신고건수
    cur1 = db.execute(
        "SELECT COUNT(*) FROM reports WHERE user_id=? AND for_userpage_type='신고' AND for_userpage_stage=3",
        (user_id,)
    )
    report_count = cur1.fetchone()[0]
    # 출동건수
    cur2 = db.execute(
        "SELECT COUNT(*) FROM reports WHERE dispatch_user_id=? AND type='출동' AND ai_stage=1",
        (user_id,)
    )
    dispatch_count = cur2.fetchone()[0]
    return jsonify({
        "report_count": report_count,
        "dispatch_count": dispatch_count
    })

@app.route('/api/user_point', methods=['GET'])
def user_point():
    user_id = request.args.get('user_id', 'guest')
    db = get_db()
    cur1 = db.execute("SELECT COUNT(*) FROM reports WHERE user_id=? AND for_userpage_type='신고' AND for_userpage_stage=3", (user_id,))
    report_count = cur1.fetchone()[0]
    cur2 = db.execute("SELECT COUNT(*) FROM reports WHERE dispatch_user_id=? AND type='출동' AND ai_stage=1", (user_id,))
    dispatch_count = cur2.fetchone()[0]
    point = report_count * 5000 + dispatch_count * 10000
    return jsonify({"point": point})

# --- 회원가입 API (테스트용) ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user_id = data.get('user_id')
    password = data.get('password')
    if not user_id or not password:
        return jsonify({'error': '아이디/비밀번호 필요'}), 400
    db = get_db()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        db.execute('INSERT INTO users (user_id, password) VALUES (?, ?)', (user_id, hashed_pw))
        db.commit()
        return jsonify({'result': 'success'})
    except sqlite3.IntegrityError:
        return jsonify({'error': '이미 존재하는 아이디'}), 400

# --- 로그인 API ---
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data.get('user_id')
    password = data.get('password')
    if not user_id or not password:
        return jsonify({'error': '아이디/비밀번호 필요'}), 400
    db = get_db()
    cur = db.execute('SELECT * FROM users WHERE user_id=?', (user_id,))
    user = cur.fetchone()
    if not user:
        return jsonify({'error': '존재하지 않는 아이디'}), 400
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    if user['password'] != hashed_pw:
        return jsonify({'error': '비밀번호 불일치'}), 400
    token = secrets.token_hex(16)
    return jsonify({'result': 'success', 'token': token})

@app.route('/api/reports', methods=['GET'])
def api_reports():
    db = get_db()
    cur = db.execute(
        "SELECT lat, lng, location, timestamp FROM reports WHERE type='신고' AND ai_stage=3"
    )
    reports = []
    for row in cur.fetchall():
        reports.append({
            "lat": row["lat"],
            "lng": row["lng"],
            "location": row["location"],
            "timestamp": row["timestamp"]
        })
    return jsonify(reports)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/admin_reports', methods=['GET'])
def admin_reports():
    db = get_db()
    cur = db.execute(
        "SELECT id, type, location, timestamp FROM reports WHERE type='신고' AND ai_stage=2 ORDER BY timestamp DESC"
    )
    reports = []
    from datetime import datetime, timezone
    import math
    for row in cur.fetchall():
        try:
            t = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            diff = now - t
            seconds = diff.total_seconds()
            if seconds < 60:
                time_str = f"{int(seconds)}초 전"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                time_str = f"{minutes}분 전"
            elif seconds < 86400:
                hours = int(seconds // 3600)
                time_str = f"{hours}시간 전"
            elif diff.days < 30:
                time_str = f"{diff.days}일 전"
            else:
                months = math.floor(diff.days / 30)
                time_str = f"{months}달 전"
        except Exception:
            time_str = ""
        reports.append({
            "id": row["id"],
            "type": row["type"],
            "location": row["location"],
            "timestamp": row["timestamp"],
            "time": time_str
        })
    return jsonify(reports)

@app.route('/api/admin_approve', methods=['POST'])
def admin_approve():
    data = request.get_json()
    report_id = data.get('id')
    stage = data.get('ai_stage')  # 3(승인), 5(취소)
    if not report_id or stage not in [3, 5]:
        return jsonify({'result': 'fail', 'error': 'invalid params'}), 400
    db = get_db()
    # ai_stage와 for_userpage_stage를 모두 업데이트
    db.execute(
        "UPDATE reports SET ai_stage=?, for_userpage_stage=? WHERE id=?",
        (stage, stage, report_id)
    )
    db.commit()
    return jsonify({'result': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
