from flask import Flask, request, jsonify, render_template, send_file
import cv2 as cv
import numpy as np
import os

import mysql.connector
import json
import textwrap
import urllib.request
from flask import Flask,render_template,request,render_template_string,url_for
from werkzeug.utils import secure_filename

from urllib.request import HTTPError
from pyzbar import pyzbar

import speech_recognition as sr
# from flask import redirect,url_for
import pyaudio
import time

app = Flask(__name__,template_folder='template')

# Define Upload Folder and Allowed Extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the Upload Directory Exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Check if File is Allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Dummy Pose Estimation Function (Replace with your model)
def perform_pose_estimation(image_path):
    # Load the image
    image = cv.imread(image_path)

    # Placeholder Pose Estimation Logic
    # For example, using MediaPipe or OpenPose here
    height, width, _ = image.shape
    dummy_keypoints = {
        "nose": (int(width * 0.5), int(height * 0.2)),
        "left_shoulder": (int(width * 0.4), int(height * 0.4)),
        "right_shoulder": (int(width * 0.6), int(height * 0.4)),
    }

    return dummy_keypoints

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# TensorFlow model for pose estimation
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Parameters
inWidth = 368
inHeight = 368
thr = 0.2

# Body parts and pairs
# Body parts and pairs for drawing
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Function to calculate angle between three points using cosine similarity
def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ab = np.array([a[0] - b[0], a[1] - b[1]])
    cb = np.array([c[0] - b[0], c[1] - b[1]])
    dot_product = np.dot(ab, cb)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_cb = np.linalg.norm(cb)
    if magnitude_ab * magnitude_cb == 0:
        return None
    cos_angle = dot_product / (magnitude_ab * magnitude_cb)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


# Define helper functions (calculate_angle, pose_estimation, calculate_squat_score, etc.)
def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Extract keypoints

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair         #string
        idFrom = BODY_PARTS[partFrom]   #int
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    return frame, points
# Copy the helper functions here

# Function to calculate squat score based on knee, hip, and ankle angles
def calculate_squat_score(points):
    # right_knee_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]],
    #                                    points[BODY_PARTS["RAnkle"]])
    # left_knee_angle = calculate_angle(points[BODY_PARTS["LHip"]], points[BODY_PARTS["LKnee"]],
    #                                   points[BODY_PARTS["LAnkle"]])
    #
    # # Assuming an ideal squat angle is around 90 degrees at the knees
    # ideal_squat_angle = 90
    # if right_knee_angle and left_knee_angle:
    #     score = (abs(ideal_squat_angle - right_knee_angle) + abs(ideal_squat_angle - left_knee_angle)) / 2
    #     return 100 - score  # The closer to 90 degrees, the better the score
    # return 0

#   SQUAT FORMULA RP
    right_knee_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]],
                                       points[BODY_PARTS["RAnkle"]])
    left_knee_angle = calculate_angle(points[BODY_PARTS["LHip"]], points[BODY_PARTS["LKnee"]],
                                      points[BODY_PARTS["LAnkle"]])

    # Assuming corrected angles for hip and knee joints
    ideal_hip_angle = 90  # Example value, adjust as needed
    ideal_knee_angle = 90  # Example value, adjust as needed

    if right_knee_angle and left_knee_angle:
        right_hip_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]],
                                          points[BODY_PARTS["RKnee"]])
        left_hip_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["LHip"]],
                                         points[BODY_PARTS["LKnee"]])

        if right_hip_angle and left_hip_angle:
            right_squat_score = (ideal_hip_angle - abs(ideal_hip_angle - right_hip_angle)) / ideal_hip_angle + \
                                (ideal_knee_angle - abs(ideal_knee_angle - right_knee_angle)) / ideal_knee_angle
            left_squat_score = (ideal_hip_angle - abs(ideal_hip_angle - left_hip_angle)) / ideal_hip_angle + \
                               (ideal_knee_angle - abs(ideal_knee_angle - left_knee_angle)) / ideal_knee_angle

            squat_score = (right_squat_score + left_squat_score) / 2 * 100  # Normalize to a score out of 100
            return squat_score
        # print("Working Squats Score")
    return 0

# Function to calculate plank score based on the back and hip angles
def calculate_plank_score(points):
    # back_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]], points[BODY_PARTS["RAnkle"]])
    #
    # # Assuming an ideal plank angle is close to 180 degrees (straight line)
    # ideal_plank_angle = 180
    # if back_angle:
    #     score = abs(ideal_plank_angle - back_angle)
    #     return 100 - score  # The closer to 180 degrees, the better the score
    # return 0

#   PLANK FORMULA RP
    # Calculate angles for upper and lower limbs
    upper_limb_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]],
                                       points[BODY_PARTS["RKnee"]])
    lower_limb_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]],
                                       points[BODY_PARTS["RAnkle"]])
    elbow_angle = calculate_angle(points[BODY_PARTS["RShoulder"]], points[BODY_PARTS["RElbow"]],
                                  points[BODY_PARTS["RWrist"]])

    if upper_limb_angle and lower_limb_angle and elbow_angle:
        # Calculate the plank score using the provided formula
        upper_limb_score = 90 - abs(upper_limb_angle - lower_limb_angle) / 90
        elbow_score = 90 - abs(90 - elbow_angle) / 90
        plank_score = (upper_limb_score + elbow_score) / 2 * 100  # Normalize to a score out of 100
        return plank_score
    # print("Working Plank Score")
    return 0

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        output_path = process_video(video_path)
        return send_file(output_path, as_attachment=True)

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], "output.mp4")
    # output_path = os.path.join(app.config['OUTPUT_FOLDER'], "output.mp4")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        estimated_frame, points = pose_estimation(frame)
        squat_score = calculate_squat_score(points)
        plank_score = calculate_plank_score(points)
        cv.putText(estimated_frame, f"Squat Score: {squat_score:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(estimated_frame, f"Plank Score: {plank_score:.2f}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        out.write(estimated_frame)

    cap.release()
    out.release()
    return output_path

# LOGIN BUTTON
@app.route('/login_button',methods=['POST'])
def login_button():
    global email,password
    global role
    global u,p
    u=""
    p=""
    # localhost_addr="https://localhost:5000"

    db=mysql.connector.connect(
        host="localhost",
        user='root',
        password='lsk12312',
        database='Fitness'
    )
    mycursor=db.cursor()

    email=request.form['email']
    password=request.form['password']

    print(email)
    print(password)

    if request.form['click']=='btn_click':

        time.sleep(3)
        # email asd
        # passw asdasd

        select_query=f"SELECT * FROM USERS WHERE EMAIL='{email}'"
        mycursor.execute(select_query)
        existing_record=mycursor.fetchone()

        if(existing_record):
            try:
                username_query = f"SELECT USERNAME FROM USERS WHERE EMAIL='{email}'" #parameter Passing
                mycursor.execute(username_query)  # type NONE
                username_record = mycursor.fetchone()
                # print(username_record)                   # username_record ---> tuple

                # If password is wrong Type Error Occurs

                u = "".join(username_record)  # tuple to str conversion
                # print(u)

                password_query=f"SELECT PASSWORD FROM USERS WHERE EMAIL='{email}' AND PASSWORD= '{password}' "
                # password_query = f"SELECT PASSWORD FROM USERS WHERE PASSWORD='{password}'"
                mycursor.execute(password_query)
                password_record = mycursor.fetchone()
                # print(password_record)              # password_record ---> tuple
                p = "".join(password_record)  # tuple to str conversion
                # print(p)

                if (u == email and password == p):
                    print("Correct email and pass")

                    #localhost_address=localhost_addr
                    return render_template('home.html')    #Localhost address of Home page

            # 200 404 error
            # asd xzczxczc
            except TypeError as e:
                incorrect_pass="""
                <html>
                <head>
                <script>
                alert("Incorrect Password.... ")
                </script>
                </head>
                <body>
                </body>
                </html>
                """
                print("tada")
                return render_template_string(incorrect_pass)


        else:
            print("Please Register...")
            register_alert = """
                                <html>
                                <head>
                                <script>
                                alert("PLEASE REGISTER TO ACCESS...")
                                </script>
                                </head>
                                <body>
                                </body>
                                </html>
                                """

            return render_template_string(register_alert)


        # query="INSERT INTO USER(username,password,role) VALUES('"+username+"','"+password+"','"+role+"')"
    return render_template('index.html')

# SIGN UP LINK
@app.route('/signup')
def signup_link():
    print("Sign up link")
    return render_template('signup.html')

# Login Page⬆️
# Sign Up Page ⬇️

# SIGNUP BUTTON
@app.route('/signup_button', methods=['POST'])
def signup():
    global fullname,username,email
    global username,password

    # print("Check point 2")

    db=mysql.connector.connect(
        host='localhost',
        user='root',
        password='lsk12312',
        database='Fitness'
    )
    mycursor=db.cursor()

    fullname = request.form['fullname']
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    print(username)
    print(password)
    print(email)

    if request.form['click']=='btn_click':
        print("asdasd")
        query=("INSERT INTO USERS(name,username,email,password) VALUES('"+fullname+"','"+username+"','"+email+"','"+password+"')")
        mycursor.execute(query)
        db.commit()
    return render_template('signup.html') # new page Submit successfully

# LOGIN LINK
@app.route('/login')
def login_link():
    print("Login Back")
    return render_template("login.html")

# LANDING LINK
@app.route('/explore')
def land_link():
    print("Landing page")
    return render_template("land.html")

# LANDING PAGE🔥
@app.route('/explore_button',methods=['POST'])
def explore_link():
    if request.form['click'] == 'btn_click':
        return render_template("login.html")


# Route to  Image Upload
@app.route('/upload_img', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform Pose Estimation
        keypoints = perform_pose_estimation(filepath)

        return jsonify({
            "message": "Pose estimation completed successfully.",
            "keypoints": keypoints
        })
    else:
        # return jsonify({"error": "File type not allowed. Allowed types: png, jpg, jpeg"}), 400.
        return keypoints


@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('land.html')


if __name__ == '__main__':
    app.run(debug=True,port=4000)

# TABLE CREATION
# CREATE TABLE USERS(ID INT PRIMARY KEY,NAME VARCHAR(50) NOT NULL,USERNAME VARCHAR(50) UNIQUE,PASSWORD VARCHAR(50),
# EMAIL VARCHAR(100) NOT NULL);