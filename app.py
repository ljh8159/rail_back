import os
import time
import sqlite3
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from werkzeug.utils import secure_filename

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DATABASE = 'reports.db'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = 'mobilenetv2_stage_model.h5'
MODEL_IMG_SIZE = (224, 224)
model = load_model(MODEL_PATH)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
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
                extra TEXT
            )
        ''')
        db.commit()

init_db()

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
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"image_{int(time.time())}.{ext}"
        filename = secure_filename(filename)
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

    db = get_db()
    db.execute(
        'INSERT INTO reports (user_id, type, photo_filename, location, lat, lng, timestamp, ai_stage, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (user_id, type_, photo_filename, location, lat, lng, timestamp, ai_stage, extra)
    )
    db.commit()
    return jsonify({'result': 'success'})

@app.route('/api/report_update', methods=['POST'])
def update_report():
    data = request.get_json()
    location = data.get('location')
    # 출동 성공 시 기존 신고를 출동/1단계로 변경
    db = get_db()
    db.execute(
        "UPDATE reports SET type='출동', ai_stage=1 WHERE location=? AND type='신고'",
        (location,)
    )
    db.commit()
    return jsonify({'result': 'updated'})

@app.route('/api/reports', methods=['GET'])
def get_reports():
    # 3단계만 반환 (마커용)
    db = get_db()
    cur = db.execute('SELECT * FROM reports WHERE ai_stage=3 ORDER BY id DESC')
    reports = [dict(row) for row in cur.fetchall()]
    return jsonify(reports)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/report_stats', methods=['GET'])
def report_stats():
    db = get_db()
    # 3단계 신고 수
    cur1 = db.execute("SELECT COUNT(*) FROM reports WHERE ai_stage=3")
    blocked_count = cur1.fetchone()[0]
    # 출동+1단계 수
    cur2 = db.execute("SELECT COUNT(*) FROM reports WHERE type='출동' AND ai_stage=1")
    dispatched_count = cur2.fetchone()[0]
    return jsonify({
        "blocked_count": blocked_count,
        "dispatched_count": dispatched_count
    })

@app.route('/api/user_stats', methods=['GET'])
def user_stats():
    user_id = request.args.get('user_id', 'guest')
    db = get_db()
    # 사용자별 신고건수 (type='신고')
    cur1 = db.execute("SELECT COUNT(*) FROM reports WHERE user_id=? AND type='신고'", (user_id,))
    report_count = cur1.fetchone()[0]
    # 사용자별 출동건수 (type='출동')
    cur2 = db.execute("SELECT COUNT(*) FROM reports WHERE user_id=? AND type='출동'", (user_id,))
    dispatch_count = cur2.fetchone()[0]
    return jsonify({
        "report_count": report_count,
        "dispatch_count": dispatch_count
    })

@app.route('/api/user_reports', methods=['GET'])
def user_reports():
    user_id = request.args.get('user_id', 'guest')
    limit = request.args.get('limit', default=3, type=int)
    db = get_db()
    cur = db.execute(
        "SELECT type, location, timestamp FROM reports WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
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
            "time": time_str
        })
    return jsonify(reports)

@app.route('/api/user_point', methods=['GET'])
def user_point():
    user_id = request.args.get('user_id', 'guest')
    db = get_db()
    # 신고건수 (type='신고')
    cur1 = db.execute("SELECT COUNT(*) FROM reports WHERE user_id=? AND type='신고'", (user_id,))
    report_count = cur1.fetchone()[0]
    # 출동건수 (type='출동')
    cur2 = db.execute("SELECT COUNT(*) FROM reports WHERE user_id=? AND type='출동'", (user_id,))
    dispatch_count = cur2.fetchone()[0]
    # 포인트 계산: 신고 5000, 출동 10000
    point = report_count * 5000 + dispatch_count * 10000
    return jsonify({"point": point})

@app.route('/api/all_reports', methods=['GET'])
def all_reports():
    limit = request.args.get('limit', default=3, type=int)
    db = get_db()
    cur = db.execute(
        "SELECT type, location, timestamp FROM reports ORDER BY timestamp DESC LIMIT ?",
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
            "time": time_str
        })
    return jsonify(reports)

if __name__ == '__main__':
    app.run(debug=True)
