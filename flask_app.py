from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import os
import cv2
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Load YOLO Model --------------------
model = YOLO("best_augmented.pt")
FRAME_SKIP = 3
RESIZE_WIDTH = 640

# Global stats for frontend
stats = {
    "alert_msg":     "✅ Area Safe",
    "fps":           0.0,
    "detected":      0,       # count of animals (excluding humans)
    "animal_names":  [],      # e.g. ["Elephant", "Tiger"]
    "border_alerts": 0,
}

last_image_path = None

# -------------------- Database Setup --------------------
DB_PATH = "wildguard_alerts.db"

def init_db():
    """Initialize SQLite database and create alerts table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            state         TEXT    NOT NULL,
            alert_msg     TEXT    NOT NULL,
            detected      INTEGER NOT NULL,
            animal_names  TEXT    NOT NULL DEFAULT '',
            border_alerts INTEGER NOT NULL,
            fps           REAL    NOT NULL,
            source        TEXT    DEFAULT 'unknown'
        )
    ''')
    # Safely add animal_names column to existing DBs that don't have it yet
    try:
        c.execute("ALTER TABLE alerts ADD COLUMN animal_names TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # column already exists — that's fine
    conn.commit()
    conn.close()

init_db()


def log_alert(state, alert_msg, detected, animal_names, border_alerts, fps, source="unknown"):
    """Insert a new alert record into the database."""
    names_str = ", ".join(animal_names) if animal_names else ""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO alerts
            (timestamp, state, alert_msg, detected, animal_names, border_alerts, fps, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        state, alert_msg, detected, names_str, border_alerts, fps, source
    ))
    conn.commit()
    conn.close()


def get_logs(limit=100, state_filter=None):
    """Fetch recent alert logs from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if state_filter and state_filter != "ALL":
        c.execute(
            "SELECT * FROM alerts WHERE state = ? ORDER BY id DESC LIMIT ?",
            (state_filter, limit)
        )
    else:
        c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_summary():
    """Get aggregate statistics from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM alerts");           total   = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM alerts WHERE state='DANGER'");  danger  = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM alerts WHERE state='WARNING'"); warning = c.fetchone()[0]
    conn.close()
    return {"total": total, "danger": danger, "warning": warning}


# -------------------- Border Logic --------------------
def is_border_crossed(box, frame_height):
    _, y1, _, y2 = box
    center_y = (y1 + y2) / 2
    return center_y > frame_height * 0.7


# -------------------- Draw Results --------------------
def draw_results(frame, result):
    """
    Returns: (annotated_frame, detected_count, animal_names_list, border_alerts, state)
    animal_names_list — one entry per detected animal box, e.g. ["Elephant", "Tiger"]
    """
    h, w, _ = frame.shape
    cv2.line(frame, (0, int(h * 0.7)), (w, int(h * 0.7)), (0, 0, 255), 2)

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return frame, 0, [], 0, "SAFE"

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy()
    ids   = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [-1] * len(boxes)

    detected_animals = 0
    animal_names     = []
    border_alerts    = 0
    state            = "SAFE"

    for box, conf, cls, tid in zip(boxes, confs, clss, ids):
        label_name = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        # Human → draw but skip alert logic entirely
        # Covers common class names a custom model might use
        HUMAN_LABELS = {"person", "human", "people", "man", "woman", "pedestrian"}
        if label_name.lower() in HUMAN_LABELS:
            color = (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Human", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            continue

        # Animal
        detected_animals += 1
        animal_names.append(label_name)
        crossed = is_border_crossed(box, h)

        if crossed:
            border_alerts += 1
            state = "DANGER"
            color = (0, 0, 255)
        else:
            if state != "DANGER":
                state = "WARNING"
            color = (0, 255, 255)

        label = f"{label_name} | ID:{int(tid)} | {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # If every detection was a human, force state back to SAFE
    if detected_animals == 0:
        state = "SAFE"

    return frame, detected_animals, animal_names, border_alerts, state


# -------------------- Image Processing --------------------
def process_image(filepath):
    img    = cv2.imread(filepath)
    result = model.track(img, persist=True, tracker="botsort.yaml", conf=0.4)[0]
    output_img, detected, animal_names, border_alerts, state = draw_results(img.copy(), result)

    stats["detected"]      = int(detected)
    stats["animal_names"]  = animal_names
    stats["border_alerts"] = int(border_alerts)
    stats["fps"]           = 0.0

    if state == "DANGER":
        stats["alert_msg"] = "🚨 Animal Approaching Border"
    elif state == "WARNING":
        stats["alert_msg"] = "⚠️ Animal Detected Near Border"
    else:
        stats["alert_msg"] = "✅ Area Safe"

    log_alert(
        state=state,
        alert_msg=stats["alert_msg"],
        detected=detected,
        animal_names=animal_names,
        border_alerts=border_alerts,
        fps=0.0,
        source=os.path.basename(filepath)
    )

    output_path = os.path.join(UPLOAD_FOLDER, "processed_" + os.path.basename(filepath))
    cv2.imwrite(output_path, output_img)
    return "processed_" + os.path.basename(filepath)


# -------------------- Video / Live Generator --------------------
_last_log_time     = 0
_last_logged_state = None
_log_interval      = 5  # seconds between repeated SAFE log entries

def gen_frames(source):
    global stats, _last_logged_state, _last_log_time

    cap         = cv2.VideoCapture(source)
    frame_count = 0
    prev_time   = time.time()
    source_name = "live" if source == 0 else os.path.basename(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame = cv2.resize(
            frame,
            (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1]))
        )

        if frame_count % FRAME_SKIP == 0:
            result = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.4)[0]
            frame, detected, animal_names, border_alerts, state = draw_results(frame, result)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            if state == "DANGER":
                alert_msg = "🚨 Animal Approaching Border"
            elif state == "WARNING":
                alert_msg = "⚠️ Animal Detected Near Border"
            else:
                alert_msg = "✅ Area Safe"

            stats["fps"]           = round(fps, 1)
            stats["detected"]      = detected
            stats["animal_names"]  = animal_names
            stats["border_alerts"] = border_alerts
            stats["alert_msg"]     = alert_msg

            # Log DANGER/WARNING immediately; throttle SAFE logs
            should_log = state in ("DANGER", "WARNING") or \
                         (state == "SAFE" and (now - _last_log_time) >= _log_interval)

            if should_log:
                log_alert(
                    state=state, alert_msg=alert_msg,
                    detected=detected, animal_names=animal_names,
                    border_alerts=border_alerts, fps=round(fps, 1),
                    source=source_name
                )
                _last_log_time     = now
                _last_logged_state = state

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()


# -------------------- Routes --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    mode          = request.args.get('mode', 'photo')
    processed_img = None
    video_path    = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            if mode == 'photo':
                processed_img = process_image(filepath)
            elif mode == 'video':
                video_path = file.filename

    return render_template('index.html', mode=mode, processed_img=processed_img, video_path=video_path)


@app.route('/clear', methods=['POST'])
def clear():
    global stats, last_image_path
    stats = {
        "alert_msg":     "✅ Area Safe",
        "fps":           0.0,
        "detected":      0,
        "animal_names":  [],
        "border_alerts": 0,
    }
    last_image_path = None
    return '', 204


@app.route('/video_feed/<filename>')
def video_feed(filename):
    source = 0 if filename == "live" else os.path.join(UPLOAD_FOLDER, filename)
    return Response(gen_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/alert')
def alert():
    return jsonify(stats)


# -------------------- DB / Log Routes --------------------
@app.route('/logs')
def logs():
    limit        = int(request.args.get('limit', 100))
    state_filter = request.args.get('state', 'ALL').upper()
    return jsonify(get_logs(limit=limit, state_filter=state_filter))


@app.route('/logs/summary')
def logs_summary():
    return jsonify(get_summary())


@app.route('/logs/clear', methods=['POST'])
def clear_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})


# -------------------- Run --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)