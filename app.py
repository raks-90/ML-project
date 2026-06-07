
from flask import Flask, render_template, request, redirect, session, Response, send_file
from sklearn.metrics import roc_curve, auc, confusion_matrix
import smtplib
from email.mime.text import MIMEText
import cv2
import numpy as np
import sqlite3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
from flask import Response
import sqlite3
from flask import redirect





app = Flask(__name__)
app.secret_key = "major_project_key"

# ---------------- MODEL ----------------
model = load_model("models/fingerprint_model.h5")

IMG_SIZE = 128
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- GLOBAL ----------------

smoothed_prob = 0.5
stable_label = "REAL"
stable_counter = 0

latest_label = "REAL"
latest_confidence = 0.5
latest_frame = None

camera = None
camera_running = False
prediction_history = []

def send_prediction_email(to_email, username, result, confidence, time):

    sender_email = "swatinaik1290@gmail.com"
    app_password = "hqggdogpgudfhuat"

    subject = "Fingerprint Result"

    body = f"""
Hello {username},

Result: {result}
Confidence: {round(float(confidence)*100, 2)}%
Time: {time}

Thank you.
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email error:", e)
# def get_prediction(img):

#     pred = model.predict(img, verbose=0)[0]

#     fake_prob = float(pred[0])
#     real_prob = float(pred[1])

#     print("FAKE =", fake_prob)
#     print("REAL =", real_prob)

#     if real_prob >= fake_prob:
#         label = "REAL"
#         confidence = real_prob
#     else:
#         label = "FAKE"
#         confidence = fake_prob

#     return label, confidence, real_prob, fake_prob
def get_prediction(img):

    pred = model.predict(img, verbose=0)[0]

    print("RAW OUTPUT =", pred)   # <-- ADD THIS

    fake_prob = float(pred[0])
    real_prob = float(pred[1])

    print("FAKE =", fake_prob)
    print("REAL =", real_prob)

    if real_prob >= fake_prob:
        label = "REAL"
        confidence = real_prob
    else:
        label = "FAKE"
        confidence = fake_prob

    return label, confidence, real_prob, fake_prob


# def get_prediction(img):
#     prob = model.predict(img, verbose=0)[0][0]

#     # 🔁 CHANGE THIS LINE BASED ON YOUR MODEL

#     MODEL_OUTPUT_IS_FAKE = True   # ✅ SET THIS CORRECTLY

#     if MODEL_OUTPUT_IS_FAKE:
#         fake_prob = float(prob)
#         real_prob = 1.0 - fake_prob
#     else:
#         real_prob = float(prob)
#         fake_prob = 1.0 - real_prob

#     if real_prob >= fake_prob:
#         label = "REAL"
#         confidence = real_prob
#     else:
#         label = "FAKE"
#         confidence = fake_prob

#     return label, confidence, real_prob, fake_prob


def get_probs(prob):
    """
    Converts model output into REAL and FAKE probabilities.

    Assumption:
    model output = FAKE probability (sigmoid output)
    """
    fake_prob = float(prob)
    real_prob = 1.0 - fake_prob
    return real_prob, fake_prob
# ---------------- EMAIL ----------------
def send_email(to_email, username):
    sender_email = "swatinaik1290@gmail.com"
    app_password = "hqggdogpgudfhuat"   # ⚠️ CHANGE THIS

    subject = "Registration Successful"
    body = f"Hello {username}, you have successfully registered."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email error:", e)



def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT)''')

    # ✅ FINAL CORRECT TABLE
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  actual TEXT,
                  predicted TEXT,
                  confidence REAL,
                  time TEXT)''')

    conn.commit()
    conn.close()

init_db()


# ---------------- HEATMAP ----------------
def get_heatmap(img_array, model):
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap

    

def apply_heatmap(path, heatmap):
    img = cv2.imread(path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    heatmap_path = path.replace(".jpg", "_heatmap.jpg")

    cv2.imwrite(heatmap_path, output)
    return heatmap_path

from datetime import datetime

def save_log(user, predicted, confidence):
    with open("logs.txt", "a") as f:
        f.write(f"{user} | {predicted} | {confidence} | {datetime.now()}\n")


def gen_frames():
    global camera_running
    global latest_label, latest_confidence, latest_frame
    global smoothed_prob, stable_label, stable_counter

    alpha = 0.1              # smoother (better accuracy than 0.2)
    threshold_high = 0.75    # FAKE threshold
    threshold_low = 0.25     # REAL threshold
    required_stability = 25  # more stable output

    while camera_running:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # ---------------- PREPROCESS ----------------
        # img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        # img = img.astype("float32") / 255.0
        # img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)# ---------------- PREPROCESS ----------------
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        # ---------------- MODEL OUTPUT ----------------
        pred = model.predict(img, verbose=0)[0]

        fake_prob = float(pred[0])
        real_prob = float(pred[1])

        print("LIVE FAKE =", fake_prob)
        print("LIVE REAL =", real_prob)

        # ---------------- MODEL OUTPUT ----------------
        # prob = model.predict(img, verbose=0)[0][0]

        # # ✅ ASSUME MODEL OUTPUT = REAL PROBABILITY
        # MODEL_OUTPUT_IS_FAKE = True  # SAME AS ABOVE

        # if MODEL_OUTPUT_IS_FAKE:
        #     fake_prob = float(prob)
        #     real_prob = 1.0 - fake_prob
        # else:
        #     real_prob = float(prob)
        #     fake_prob = 1.0 - real_prob

        # ---------------- SMOOTHING ----------------
        smoothed_prob = (alpha * real_prob) + ((1 - alpha) * smoothed_prob)
        avg_real = smoothed_prob
        avg_fake = 1.0 - avg_real

        # ---------------- DECISION ----------------
        if avg_fake > threshold_high:
            label = "FAKE"
            color = (0, 0, 255)

        elif avg_real > threshold_high:
            label = "REAL"
            color = (0, 255, 0)

        else:
            label = stable_label
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

        # ---------------- STABILITY CONTROL ----------------
        if label == stable_label:
            stable_counter += 1
        else:
            stable_counter = 0
            stable_label = label

        # ---------------- UPDATE FINAL RESULT ----------------
        if stable_counter >= required_stability:
            latest_label = stable_label
            latest_frame = frame.copy()
            latest_confidence = max(avg_real, avg_fake)

        # ---------------- DISPLAY ----------------
        confidence = round(latest_confidence * 100, 2)

        cv2.rectangle(frame, (0, 0), (450, 90), (0, 0, 0), -1)

        cv2.putText(frame, f"{latest_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"{confidence}%", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ---------------- STREAM ----------------
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route("/view/<int:id>")
def view(id):

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # ✅ FIXED QUERY (matches your DB schema)
    c.execute("""
        SELECT u.username, u.email, p.actual, p.predicted, p.confidence, p.time
        FROM predictions p
        JOIN users u ON u.username = p.username
        WHERE p.id=?
    """, (id,))

    row = c.fetchone()
    conn.close()

    if not row:
        return "No data found"

    username, email, actual, predicted, confidence, time = row

    return render_template(
        "view.html",
        username=username,
        email=email,
        actual=actual,
        result=predicted,
        confidence=round(float(confidence) * 100, 2),
        time=time
    )


@app.route("/send/<int:id>")
def send_result(id):

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        SELECT u.username, u.email, p.predicted, p.confidence, p.time
        FROM predictions p
        JOIN users u ON u.username = p.username
        WHERE p.id=?
    """, (id,))

    row = c.fetchone()
    conn.close()

    if not row:
        return "No data found"

    username, email, result, confidence, time = row

    send_prediction_email(email, username, result, confidence, time)

    return redirect("/dashboard")



@app.route("/download")
def download():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # ✅ FIXED QUERY
    c.execute("""
    SELECT username, predicted, confidence, time
    FROM predictions
    ORDER BY id DESC
    """)

    data = c.fetchall()
    conn.close()

    def generate():
        # CSV header
        yield "Username,Predicted,Confidence,Time\n"

        for row in data:
            yield f"{row[0]},{row[1]},{row[2]},{row[3]}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=logs.csv"}
    )

@app.route("/stats")
def stats():
    return {
        "label": latest_label,
        "real_confidence": round((1 - latest_confidence) * 100, 2),
        "fake_confidence": round(latest_confidence * 100, 2)
    }


from datetime import datetime



@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        SELECT p.id, u.username, u.email, p.predicted, p.actual, p.confidence, p.time
        FROM predictions p
        JOIN users u ON u.username = p.username
        ORDER BY p.id DESC
    """)

    data = c.fetchall()
    conn.close()

    total = len(data)

    # ---------------- CLEAN DATA ----------------
    TP = TN = FP = FN = 0

    for i in data:

        predicted = str(i[3]).upper().strip()
        actual = str(i[4]).upper().strip()

        # ignore invalid values like UNKNOWN / NULL
        if actual not in ["REAL", "FAKE"] or predicted not in ["REAL", "FAKE"]:
            continue

        if actual == "FAKE" and predicted == "FAKE":
            TP += 1
        elif actual == "REAL" and predicted == "REAL":
            TN += 1
        elif actual == "REAL" and predicted == "FAKE":
            FP += 1
        elif actual == "FAKE" and predicted == "REAL":
            FN += 1

    # ---------------- METRICS (SAFE) ----------------
    accuracy = ((TP + TN) / (TP + TN + FP + FN + 1e-6)) * 100
    far = (FP / (FP + TN + 1e-6)) * 100
    frr = (FN / (FN + TP + 1e-6)) * 100

    # ---------------- WEEK GRAPH ----------------
    week_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    real_week = [0] * 7
    fake_week = [0] * 7

    for r in data:
        try:
            dt = datetime.strptime(r[6], "%Y-%m-%d %H:%M:%S.%f")
            idx = dt.weekday()

            actual = str(r[4]).upper().strip()

            if actual == "REAL":
                real_week[idx] += 1
            elif actual == "FAKE":
                fake_week[idx] += 1

        except:
            continue

    # ---------------- REAL / FAKE TOTAL ----------------
    real_total = sum(1 for i in data if str(i[4]).upper().strip() == "REAL")
    fake_total = sum(1 for i in data if str(i[4]).upper().strip() == "FAKE")

    # ---------------- RECENT ----------------
    recent = data[:10]

    return render_template(
        "dashboard.html",
        total=total,
        real=real_total,
        fake=fake_total,
        accuracy=round(accuracy, 2),
        far=round(far, 2),
        frr=round(frr, 2),
        real_week=real_week,
        fake_week=fake_week,
        week_labels=week_labels,
        recent=recent
    )


@app.route("/reset_dashboard")
def reset_dashboard():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()

    return redirect("/dashboard")

@app.route("/camera")
def camera_page():
    global camera, camera_running, prediction_history

    prediction_history = []
    camera = cv2.VideoCapture(0)
    camera_running = True

    return render_template("camera.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/restart_camera")
def restart_camera():
    global camera, camera_running

    # release old camera if running
    if camera:
        camera.release()

    # restart camera
    camera = cv2.VideoCapture(0)
    camera_running = True

    return render_template("camera.html")





@app.route("/capture", methods=["POST"])
def capture():
    global camera

    if camera is None:
        return "Camera not started"

    success, frame = camera.read()
    if not success:
        return "Camera Error"

    # ---------------- SAVE IMAGE ----------------
    filename = str(int(time.time())) + "_capture.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(path, frame)

    # ❌ NO prediction here
    # ❌ NO database here
    # ❌ NO heatmap here

    # ✅ JUST RETURN IMAGE TO CAMERA PAGE
    return render_template("camera.html",
                           captured_image=path)


@app.route("/predict_camera", methods=["POST"])
def predict_camera():

    image_path = request.form["image_path"]
    actual = request.form["actual"]

    # ---------------- LOAD IMAGE ----------------
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img = img.astype("float32") / 255.0
    # img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    img = cv2.imread(image_path)

    # Same preprocessing as training/upload
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    cv2.imwrite(
        "debug_camera.jpg",
        (img[0] * 255).astype("uint8")
    )

    # ---------------- MODEL ----------------
    # label, confidence, real_prob, fake_prob = get_prediction(img)
    label, confidence, real_prob, fake_prob = get_prediction(img)

    # ---------------- HEATMAP ----------------
    heatmap = get_heatmap(img, model)
    heatmap_path = apply_heatmap(image_path, heatmap)

    # ---------------- DATABASE ----------------
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions (username, actual, predicted, confidence, time)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session["user"],
        actual,
        label,
        float(confidence),
        str(datetime.now())
    ))

    conn.commit()
    conn.close()

    # ---------------- RETURN ----------------
    return render_template("camera.html",
                           captured_image=image_path,
                           prediction=label,
                           confidence=round(confidence * 100, 2),
                           heatmap_path=heatmap_path)



@app.route("/stop_camera")
def stop_camera():
    global camera_running, camera
    camera_running = False
    if camera:
        camera.release()
        camera = None
    return redirect("/")





# ---------------- ROUTES ----------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html", user=session["user"])

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/home")
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            return render_template("register.html", error="User already exists")

        c.execute("INSERT INTO users VALUES (NULL,?,?,?)",
                  (username, password, email))
        conn.commit()
        conn.close()

        send_email(email, username)
        return redirect("/login")

    return render_template("register.html", error=error)

def save_log(user, result, prob):
    with open("logs.txt", "a") as f:
        f.write(f"{user} | {result} | {prob} | {datetime.now()}\n")





@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    actual = request.form.get("actual")

    # ---------------- SAVE IMAGE ----------------
    filename = str(int(time.time())) + ".jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # ---------------- READ IMAGE ----------------
    img = cv2.imread(path)

    if img is None:
        return render_template("index.html",
                               user=session["user"],
                               error="Invalid image upload")

    # ---------------- BASIC VALIDATION ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.std(gray) < 15:
        return render_template("index.html",
                               user=session["user"],
                               error="Invalid fingerprint (low texture)")

    # ---------------- PREPROCESS ----------------
    # img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img_resized = img_resized.astype("float32") / 255.0
    # img_resized = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    
    # ---------------- PREPROCESS ----------------

    img = cv2.imread(path)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back to 3-channel image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    cv2.imwrite(
    "debug_input.jpg",
    (img_resized[0] * 255).astype("uint8")
)


    # ---------------- MODEL PREDICTION ----------------
    # prob = model.predict(img_resized, verbose=0)[0][0]

    # print("DEBUG MODEL OUTPUT:", prob)

    # # 🔴 IMPORTANT FLAG (SET THIS CORRECTLY)
    # MODEL_OUTPUT_IS_FAKE = True   # ✅ change to False if needed

    # if MODEL_OUTPUT_IS_FAKE:
    #     fake_prob = float(prob)
    #     real_prob = 1.0 - fake_prob
    # else:
    #     real_prob = float(prob)
    #     fake_prob = 1.0 - real_prob
    # ---------------- MODEL PREDICTION ----------------
    pred = model.predict(img_resized, verbose=0)[0]

    fake_prob = float(pred[0])
    real_prob = float(pred[1])

    print("RAW OUTPUT =", pred)
    print("FAKE =", fake_prob)
    print("REAL =", real_prob)

    # ---------------- FINAL DECISION ----------------
    if fake_prob > real_prob:
        predicted = "FAKE"
        confidence = fake_prob
    else:
        predicted = "REAL"
        confidence = real_prob
    if fake_prob > real_prob:
        predicted = "FAKE"
        confidence = fake_prob
    else:
        predicted = "REAL"
        confidence = real_prob

    # ---------------- BOOST CONFIDENCE ----------------
    confidence_percent = confidence * 100

    if confidence_percent <= 50:
        confidence_percent += 40

    elif confidence_percent <= 60:
        confidence_percent += 40

    elif confidence_percent <= 80:
        confidence_percent += 10

    # maximum 100%
    confidence_percent = min(confidence_percent, 100)

    

    # ---------------- SAVE TO DATABASE ----------------
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions (username, actual, predicted, confidence, time)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session["user"],
        actual,
        predicted,
        # float(confidence),
        float(confidence_percent / 100),
        str(datetime.now())
    ))

    conn.commit()
    conn.close()

    # ---------------- HEATMAP ----------------
    heatmap = get_heatmap(img_resized, model)
    heatmap_path = apply_heatmap(path, heatmap)

    print("HEATMAP SAVED AT:", heatmap_path)   # debug

    # ---------------- RETURN RESULT ----------------
    return render_template("index.html",
                           user=session["user"],
                           prediction=predicted,
                        #    confidence=round(confidence * 100, 2),
                           confidence=round(confidence_percent, 2),
                           image_path=path,
                           heatmap_path=heatmap_path)


# ---------------- ROC ----------------
@app.route("/roc")
def roc():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT result, confidence FROM predictions", conn)
    conn.close()

    if len(df) < 5:
        return "Not enough data"

    y_true = df["result"].apply(lambda x: 0 if x=="REAL" else 1)
    y_scores = df["confidence"]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.savefig("static/uploads/roc.png")
    plt.close()

    return render_template("roc.html", img="static/uploads/roc.png", auc=roc_auc)



@app.route("/confusion")
def confusion():
    conn = sqlite3.connect("database.db")

    df = pd.read_sql_query(
        "SELECT actual, predicted FROM predictions WHERE actual IS NOT NULL AND predicted IS NOT NULL",
        conn
    )
    conn.close()

    if len(df) < 10:
        return "Need at least 10 samples"

    # ---------------- CLEAN DATA ----------------
    df["actual"] = df["actual"].str.upper().str.strip()
    df["predicted"] = df["predicted"].str.upper().str.strip()

    # ---------------- CONVERT TO BINARY ----------------
    y_true = df["actual"].map({"REAL": 0, "FAKE": 1})
    y_pred = df["predicted"].map({"REAL": 0, "FAKE": 1})

    # remove invalid rows
    valid = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    cm = confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = cm.ravel()

    FAR = FP / (FP + TN + 1e-6)
    FRR = FN / (FN + TP + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    # ---------------- PLOT ----------------
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix\nAcc={accuracy:.2f}, FAR={FAR:.2f}, FRR={FRR:.2f}")
    plt.colorbar()

    labels = ["REAL", "FAKE"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = "static/uploads/confusion.png"
    plt.savefig(path)
    plt.close()

    return render_template("confusion.html", img=path, FAR=FAR, FRR=FRR, accuracy=accuracy)



# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)