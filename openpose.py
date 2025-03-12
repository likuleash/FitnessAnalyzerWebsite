# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('OpenPose using OpenCV', frame)

    # try 1 plex
    # import tensorflow as tf
    # from tensorflow.keras import layers, models
    # import cv2
    # import numpy as np
    #
    #
    # # Define the U-Net model for skeleton extraction
    # def unet_model(input_size=(256, 256, 1)):
    #     inputs = layers.Input(input_size)
    #
    #     # Encoder
    #     c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    #     c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    #     p1 = layers.MaxPooling2D((2, 2))(c1)
    #
    #     c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    #     c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    #     p2 = layers.MaxPooling2D((2, 2))(c2)
    #
    #     c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    #     c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    #     p3 = layers.MaxPooling2D((2, 2))(c3)
    #
    #     # Bottleneck
    #     c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    #     c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    #
    #     # Decoder
    #     u5 = layers.UpSampling2D((2, 2))(c4)
    #     u5 = layers.concatenate([u5, c3])
    #     c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    #     c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    #
    #     u6 = layers.UpSampling2D((2, 2))(c5)
    #     u6 = layers.concatenate([u6, c2])
    #     c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    #     c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    #
    #     u7 = layers.UpSampling2D((2, 2))(c6)
    #     u7 = layers.concatenate([u7, c1])
    #     c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    #     c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    #
    #     outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    #
    #     model = models.Model(inputs=[inputs], outputs=[outputs])
    #     return model
    #
    #
    # # Load and preprocess the image
    # def preprocess_image(image_path):
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image, (256, 256))
    #     image = image / 255.0  # Normalize to [0, 1]
    #     return np.expand_dims(image, axis=-1)
    #
    #
    # # Predict skeleton from an image
    # def predict_skeleton(model, image_path):
    #     image = preprocess_image(image_path)
    #     prediction = model.predict(np.array([image]))
    #     skeleton = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)  # Binarize the output
    #     return skeleton
    #
    #
    # # Example usage
    # if __name__ == "__main__":
    #     model = unet_model()
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    #     # Load pre-trained weights if available
    #     # model.load_weights('path_to_weights.h5')
    #
    #     skeleton_image = predict_skeleton(model, 'C:\Users\likul\Desktop\ISBN\img1.jpeg')
    #
    #     # Display the skeleton image
    #     cv2.imshow('Skeleton', skeleton_image * 255)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # import cv2
    # import numpy as np
    #
    # # Paths to the pre-trained OpenPose models
    # protoFile = "pose_deploy_linevec.prototxt"
    # weightsFile = "pose_iter_440000.caffemodel"
    #
    # # Read the network into OpenCV
    # net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    #
    #
    # # Function to process an image or frame for skeleton detection
    # def extract_skeleton(image):
    #     # Set input parameters for the model
    #     inputWidth = 368
    #     inputHeight = 368
    #     inputScale = 1.0 / 255
    #     inBlob = cv2.dnn.blobFromImage(image, inputScale, (inputWidth, inputHeight),
    #                                    (0, 0, 0), swapRB=False, crop=False)
    #
    #     net.setInput(inBlob)
    #     output = net.forward()
    #
    #     H, W = image.shape[:2]
    #
    #     # Define the points based on OpenPose keypoints
    #     points = []
    #     nPoints = 15  # Adjust this if using a different model
    #     threshold = 0.1
    #
    #     for i in range(nPoints):
    #         # Confidence map of corresponding body part.
    #         probMap = output[0, i, :, :]
    #
    #         # Find global maxima of the probMap
    #         minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    #
    #         # Scale the point to match the original image size
    #         x = (W * point[0]) / output.shape[3]
    #         y = (H * point[1]) / output.shape[2]
    #
    #         if prob > threshold:
    #             points.append((int(x), int(y)))
    #         else:
    #             points.append(None)
    #
    #     # Define the pairs of connected keypoints in OpenPose skeleton structure
    #     POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    #                   [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    #                   [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
    #
    #     # Draw lines between the detected keypoints
    #     for pair in POSE_PAIRS:
    #         partA = pair[0]
    #         partB = pair[1]
    #
    #         if points[partA] and points[partB]:
    #             cv2.line(image, points[partA], points[partB], (0, 255, 255), 2)
    #             cv2.circle(image, points[partA], 5, (0, 0, 255), -1)
    #             cv2.circle(image, points[partB], 5, (0, 0, 255), -1)
    #
    #     return image
    #
    #
    # # Function to process video
    # def process_video(video_path):
    #     cap = cv2.VideoCapture(video_path)
    #
    #     if not cap.isOpened():
    #         print("Error: Could not open video.")
    #         return
    #
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #
    #         # Extract skeleton from each frame
    #         frame_with_skeleton = extract_skeleton(frame)
    #
    #         # Show the frame
    #         cv2.imshow('Skeleton', frame_with_skeleton)
    #
    #         # Press 'q' to exit
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     cap.release()
    #     cv2.destroyAllWindows()
    #
    #
    # # Function to process a single image
    # def process_image(image_path):
    #     image = cv2.imread('img1.jpeg')
    #     if image is None:
    #         print("Error: Could not read image.")
    #         return
    #
    #     skeleton_image = extract_skeleton(image)
    #     cv2.imshow('Skeleton Image', skeleton_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #
    # # Example usage:
    # # For image processing
    # process_image("img1.jpeg")
    #
    # # For video processing
    # # process_video("path_to_video.mp4")

    # import cv2
    # import mediapipe as mp
    # import numpy as np
    # import os
    #
    #
    # class SkeletonExtractor:
    #     def __init__(self):
    #         self.mp_drawing = mp.solutions.drawing_utils
    #         self.mp_pose = mp.solutions.pose
    #         self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    #
    #     def extract_skeleton(self, image):
    #         # Convert the BGR image to RGB
    #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #         # Process the image and get the pose landmarks
    #         results = self.pose.process(image_rgb)
    #
    #         # Create a black image with the same dimensions as the input image
    #         skeleton_image = np.zeros(image.shape, dtype=np.uint8)
    #
    #         if results.pose_landmarks:
    #             # Draw the pose landmarks on the black image
    #             self.mp_drawing.draw_landmarks(
    #                 skeleton_image,
    #                 results.pose_landmarks,
    #                 self.mp_pose.POSE_CONNECTIONS,
    #                 landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
    #                 connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    #             )
    #
    #         return skeleton_image
    #
    #     def process_image(self, image_path, output_path):
    #         # Read the image
    #         image = cv2.imread(image_path)
    #
    #         # Extract skeleton
    #         skeleton = self.extract_skeleton(image)
    #
    #         # Save the skeleton image
    #         cv2.imwrite(output_path, skeleton)
    #         print(f"Skeleton image saved to {output_path}")
    #
    #     def process_video(self, video_path, output_path):
    #         # Open the video file
    #         cap = cv2.VideoCapture(video_path)
    #
    #         # Get video properties
    #         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         fps = int(cap.get(cv2.CAP_PROP_FPS))
    #
    #         # Create VideoWriter object
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    #
    #         while cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #
    #             # Extract skeleton from the frame
    #             skeleton = self.extract_skeleton(frame)
    #
    #             # Write the skeleton frame
    #             out.write(skeleton)
    #
    #         # Release everything
    #         cap.release()
    #         out.release()
    #         print(f"Skeleton video saved to {output_path}")
    #
    #
    # def main():
    #     extractor = SkeletonExtractor()
    #
    #     # Process an image
    #     image_path = "img1.jpeg"
    #     image_output_path = "C:\Users\likul\Desktop\ISBN\   "
    #     extractor.process_image(image_path, image_output_path)
    #
    #     # # Process a video
    #     # video_path = "path/to/your/video.mp4"
    #     # video_output_path = "path/to/your/output_skeleton_video.mp4"
    #     # extractor.process_video(video_path, video_output_path)
    #
    #
    # if __name__ == "__main__":
    #     main()

    # TRY 3
    # import cv2
    # import numpy as np
    # import math
    # from openpose import pyopenpose as op
    #
    # def get_skeleton(image_path):
    #     # Initialize OpenPose
    #     params = {
    #         "model_folder": "path/to/openpose/models/",
    #         "net_resolution": "256x256"
    #     }
    #     opWrapper = op.WrapperPython()
    #     opWrapper.configure(params)
    #     opWrapper.start()
    #
    #     # Read image
    #     image = cv2.imread(image_path)
    #
    #     # Process image
    #     datum = op.Datum()
    #     datum.cvInputData = image
    #     opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    #
    #     # Get keypoints
    #     keypoints = datum.poseKeypoints[0]
    #
    #     # Draw skeleton
    #     skeleton_image = np.zeros(image.shape, dtype=np.uint8)
    #     for i in range(25):
    #         if keypoints[i][2] > 0:
    #             cv2.circle(skeleton_image, (int(keypoints[i][0]), int(keypoints[i][1])), 3, (255, 255, 255), -1)
    #
    #     # Draw connections
    #     pairs = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    #     for pair in pairs:
    #         if keypoints[pair[0]][2] > 0 and keypoints[pair[1]][2] > 0:
    #             cv2.line(skeleton_image,
    #                      (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1])),
    #                      (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1])),
    #                      (255, 255, 255), 2)
    #
    #     return skeleton_image, keypoints
    #
    # def calculate_angle(a, b, c):
    #     ba = a - b
    #     bc = c - b
    #     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    #     angle = np.arccos(cosine_angle)
    #     return np.degrees(angle)
    #
    # def evaluate_squat(keypoints):
    #     # Calculate hip angle
    #     hip_angle = calculate_angle(keypoints[1], keypoints[8], keypoints[9])
    #
    #     # Calculate knee angle
    #     knee_angle = calculate_angle(keypoints[8], keypoints[9], keypoints[10])
    #
    #     # Define correct angles (these should be adjusted based on expert input)
    #     correct_hip_angle = 90
    #     correct_knee_angle = 90
    #
    #     # Calculate scores
    #     hip_score = max(0, 1 - abs(hip_angle - correct_hip_angle) / correct_hip_angle)
    #     knee_score = max(0, 1 - abs(knee_angle - correct_knee_angle) / correct_knee_angle)
    #
    #     # Overall score
    #     overall_score = (hip_score + knee_score) / 2 * 100
    #
    #     return overall_score, hip_angle, knee_angle
    #
    # def main():
    #     image_path = "Squats.mp4"
    #     skeleton_image, keypoints = get_skeleton(image_path)
    #
    #     # Save skeleton image
    #     cv2.imwrite("skeleton.jpg", skeleton_image)
    #
    #     # Evaluate squat (you can add more exercises)
    #     score, hip_angle, knee_angle = evaluate_squat(keypoints)
    #
    #     print(f"Squat Score: {score:.2f}%")
    #     print(f"Hip Angle: {hip_angle:.2f} degrees")
    #     print(f"Knee Angle: {knee_angle:.2f} degrees")
    #
    # if __name__ == "__main__":
    #     main()
