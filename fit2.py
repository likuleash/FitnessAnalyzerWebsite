# TRY 2
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt



# Load the TensorFlow model for pose estimation
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Image size for the model
inWidth = 368
inHeight = 368
thr = 0.2

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


# Pose estimation function
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
        partFrom, partTo = pair
        idFrom = BODY_PARTS[partFrom]
        print(type(idFrom))
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    return frame, points


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


# Function to calculate crunches score based on torso angle
def calculate_crunches_score(points):
    # neck_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]])
    #
    # # Assuming an ideal crunch angle is smaller (closer to 0 degrees) indicating torso flexion
    # ideal_crunch_angle = 45  # Flexed torso
    # if neck_angle:
    #     score = abs(neck_angle - ideal_crunch_angle)
    #     return max(0, 100 - score)  # Lower score indicates better crunch
    # return 0

# CRUNCHES FORMULA RP
    neck_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]])
    hip_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]], points[BODY_PARTS["RAnkle"]])

    # Assuming an ideal crunch angle is smaller (closer to 0 degrees) indicating torso flexion
    ideal_crunch_angle = 45  # Flexed torso
    if neck_angle and hip_angle:
        corrected_angle = ideal_crunch_angle
        score = (corrected_angle - abs(corrected_angle - hip_angle)) / corrected_angle * 100
        return max(0, score)  # Lower score indicates better crunch
    return 0

# Function to calculate bent-over rows score based on back angle
def calculate_bent_over_rows_score(points):
    # hip_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]],
    #                             points[BODY_PARTS["RShoulder"]])
    #
    # # Ideal bent-over rows angle for back should be around 45 degrees relative to the legs
    # ideal_row_angle = 45
    # if hip_angle:
    #     score = abs(hip_angle - ideal_row_angle)
    #     return max(0, 100 - score)
    # return 0

#   BENT OVER ROWS FORMULA RP
    upper_extremity_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RShoulder"]],
                                            points[BODY_PARTS["RHip"]])
    upper_arm_angle = calculate_angle(points[BODY_PARTS["RShoulder"]], points[BODY_PARTS["RElbow"]],
                                      points[BODY_PARTS["RWrist"]])
    lower_arm_angle = calculate_angle(points[BODY_PARTS["RElbow"]], points[BODY_PARTS["RWrist"]],
                                      points[BODY_PARTS["RAnkle"]])

    if upper_extremity_angle and upper_arm_angle and lower_arm_angle:
        score = (abs(90 - upper_extremity_angle) / 90 * 0.50) + (abs(90 - upper_arm_angle) / 90 * 0.25) + (
                    abs(90 - lower_arm_angle) / 90 * 0.25)
        return max(0, score * 100)  # Normalize to a percentage
    return 0

# Processing a video
def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        estimated_frame, points = pose_estimation(frame)

        squat_score = calculate_squat_score(points)
        plank_score = calculate_plank_score(points)

        crunches_score = calculate_crunches_score(points)
        rows_score = calculate_bent_over_rows_score(points)

        cv.putText(estimated_frame, f"Squat Score: {squat_score:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 0, 0), 2)
        if(squat_score >= 100):
            print("CORRECT SQUAT POSITION (>=100)")
        cv.putText(estimated_frame, f"Plank Score: {plank_score:.2f}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 0, 0), 2)
        if (plank_score >= 100):
            print("CORRECT PLANK POSITION (>=100)")
        # cv.putText(estimated_frame, f"Crunches Score: {crunches_score:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
        #            (255, 255, 255), 2)
        # cv.putText(estimated_frame, f"Rows Score: {rows_score:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7,
        #            (255, 255, 255), 2)

        # Adjust the window size here
        cv.namedWindow('Pose Estimation', cv.WINDOW_NORMAL)
        cv.resizeWindow('Pose Estimation', 800, 600)  # Set the desired window size

        cv.imshow('Pose Estimation', estimated_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Example usage
process_video('pl1.mp4')

# sq1.mp4
# pl1.mp4

# LUNGE MOVEMENT FORMULA
# def calculate_lunge_score(points):
#     # Calculate the angle between the hip, knee, and ankle for both legs
#     right_leg_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]], points[BODY_PARTS["RAnkle"]])
#     left_leg_angle = calculate_angle(points[BODY_PARTS["LHip"]], points[BODY_PARTS["LKnee"]], points[BODY_PARTS["LAnkle"]])
#
#     # Assuming an ideal lunge angle is around 90 degrees for both legs
#     ideal_lunge_angle = 90
#
#     # Calculate the score based on the deviation from the ideal angle
#     if right_leg_angle and left_leg_angle:
#         right_leg_score = max(0, 100 - abs(right_leg_angle - ideal_lunge_angle))
#         left_leg_score = max(0, 100 - abs(left_leg_angle - ideal_lunge_angle))
#         return (right_leg_score + left_leg_score) / 2  # Average score for both legs
#     return 0

# lunge_score = calculate_lunge_score(points)
# cv.putText(estimated_frame, f"Lunge Score: {lunge_score:.2f}", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
#                    (255, 255, 255), 2)


# PUSH UP MOVEMENT
# def calculate_pushup_score(points):
#     # Calculate the angle between the shoulder, elbow, and wrist for both arms
#     right_arm_angle = calculate_angle(points[BODY_PARTS["RShoulder"]], points[BODY_PARTS["RElbow"]], points[BODY_PARTS["RWrist"]])
#     left_arm_angle = calculate_angle(points[BODY_PARTS["LShoulder"]], points[BODY_PARTS["LElbow"]], points[BODY_PARTS["LWrist"]])
#
#     # Assuming an ideal push-up angle is around 90 degrees for both arms
#     ideal_pushup_angle = 90
#
#     # Calculate the score based on the deviation from the ideal angle
#     if right_arm_angle and left_arm_angle:
#         right_arm_score = max(0, 100 - abs(right_arm_angle - ideal_pushup_angle))
#         left_arm_score = max(0, 100 - abs(left_arm_angle - ideal_pushup_angle))
#         return (right_arm_score + left_arm_score) / 2  # Average score for both arms
#     return 0
#   pushup_score = calculate_pushup_score(points)
#   cv.putText(estimated_frame, f"Push-Up Score: {pushup_score:.2f}", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
#                    (255, 255, 255), 2)