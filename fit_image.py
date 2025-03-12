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
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    return frame, points

# SQUAT FUNCTION
# Function to calculate squat score based on knee, hip, and ankle angles
def calculate_squat_score(points):
    right_knee_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]],
                                       points[BODY_PARTS["RAnkle"]])
    left_knee_angle = calculate_angle(points[BODY_PARTS["LHip"]], points[BODY_PARTS["LKnee"]],
                                      points[BODY_PARTS["LAnkle"]])

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
    return 0
# def calculate_crunches_score(points):
#     # neck_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]])
#     #
#     # # Assuming an ideal crunch angle is smaller (closer to 0 degrees) indicating torso flexion
#     # ideal_crunch_angle = 45  # Flexed torso
#     # if neck_angle:
#     #     score = abs(neck_angle - ideal_crunch_angle)
#     #     return max(0, 100 - score)  # Lower score indicates better crunch
#     # return 0



# Function to process an image
def process_image(image_path):
    frame = cv.imread(image_path)
    if frame is None:
        raise IOError("Cannot open image")

    estimated_frame, points = pose_estimation(frame)
    # SQUAT SCORE
    squat_score = calculate_squat_score(points)

    cv.putText(estimated_frame, f"Squat Score: {squat_score:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (255, 255, 255), 2)
    print(f"Squat Score: {squat_score:.2f}")

    # Adjust the window size here
    cv.namedWindow('Pose Estimation', cv.WINDOW_NORMAL)
    cv.resizeWindow('Pose Estimation', 800, 600)  # Set the desired window size

    cv.imshow('Pose Estimation', estimated_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage
process_image('sq2.jpeg')

# Sample Image for Points
# sq1.jpeg 129
# sq2.jpeg 111
# sq3.jpg 131
#




# <future>

# # PLANK FUNCTION
# # Function to calculate plank score based on the back and hip angles
# def calculate_plank_score(points):
#     # back_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]], points[BODY_PARTS["RAnkle"]])
#     #
#     # # Assuming an ideal plank angle is close to 180 degrees (straight line)
#     # ideal_plank_angle = 180
#     # if back_angle:
#     #     score = abs(ideal_plank_angle - back_angle)
#     #     return 100 - score  # The closer to 180 degrees, the better the score
#     # return 0
#
# #   PLANK FORMULA RP
#     # Calculate angles for upper and lower limbs
#     upper_limb_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]],
#                                        points[BODY_PARTS["RKnee"]])
#     lower_limb_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]],
#                                        points[BODY_PARTS["RAnkle"]])
#     elbow_angle = calculate_angle(points[BODY_PARTS["RShoulder"]], points[BODY_PARTS["RElbow"]],
#                                   points[BODY_PARTS["RWrist"]])
#
#     if upper_limb_angle and lower_limb_angle and elbow_angle:
#         # Calculate the plank score using the provided formula
#         upper_limb_score = 90 - abs(upper_limb_angle - lower_limb_angle) / 90
#         elbow_score = 90 - abs(90 - elbow_angle) / 90
#         plank_score = (upper_limb_score + elbow_score) / 2 * 100  # Normalize to a score out of 100
#         return plank_score
#     # print("Working Plank Score")
#     return 0

# PLANK SCORE
    # plank_score = calculate_squat_score(points)
    #
    # cv.putText(estimated_frame, f"Plank Score: {plank_score:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7,
    #             (255, 255, 255), 2)
    # print(f"Plank Score: {plank_score:.2f}")


# # CRUNCHES SCORE
# crunches_score = calculate_crunches_score(points)
#
# cv.putText(estimated_frame, f"Crunches Score: {crunches_score:.2f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7,
#            (255, 255, 255), 2)
# print(f"Crunches Score: {crunches_score:.2f}")
#


# # CRUNCHES FORMULA RP
#     neck_angle = calculate_angle(points[BODY_PARTS["Neck"]], points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]])
#     hip_angle = calculate_angle(points[BODY_PARTS["RHip"]], points[BODY_PARTS["RKnee"]], points[BODY_PARTS["RAnkle"]])
#
#     # Assuming an ideal crunch angle is smaller (closer to 0 degrees) indicating torso flexion
#     ideal_crunch_angle = 45  # Flexed torso
#     if neck_angle and hip_angle:
#         corrected_angle = ideal_crunch_angle
#         score = (corrected_angle - abs(corrected_angle - hip_angle)) / corrected_angle * 100
#         return max(0, score)  # Lower score indicates better crunch
#     return 0
#