import cv2
import numpy as np
import os
import json

# Paths to the model files
proto_file = "pose_deploy_linevec.prototxt"
weights_file = "pose_iter_440000.caffemodel"
n_points = 18  # Updated number of points
smoothing_window = 5  # Number of frames to average for smoothing

# Define the pairs of points for drawing the skeleton
pose_pairs = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]
]

# Load the network
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
# Set preferable backend and target to CUDA (GPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

def smooth_points(points_buffer):
    # Compute the average of points in the buffer
    averaged_points = []
    for i in range(len(points_buffer[0])):
        x_sum, y_sum, count = 0, 0, 0
        for points in points_buffer:
            if points[i] is not None:
                x_sum += points[i][0]
                y_sum += points[i][1]
                count += 1
        if count > 0:
            averaged_points.append((x_sum // count, y_sum // count))
        else:
            averaged_points.append(None)
    return averaged_points

def process_video(input_video_path, output_data_path, skeleton_video_path, overlay_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_skeleton = None
    out_overlay = None

    pose_data = []
    frame_count = 0
    points_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # Skip 4 frames at a time
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = []
        for i in range(n_points):
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            x = (frame.shape[1] * point[0]) / W
            y = (frame.shape[0] * point[1]) / H

            if prob > 0.1:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        points_buffer.append(points)
        if len(points_buffer) > smoothing_window:
            points_buffer.pop(0)

        smoothed_points = smooth_points(points_buffer)
        pose_data.append(smoothed_points)

        # Create a white background for skeleton only output
        skeleton_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        overlay_frame = frame.copy()

        # Draw skeleton
        for pair in pose_pairs:
            part_a = pair[0]
            part_b = pair[1]

            if smoothed_points[part_a] and smoothed_points[part_b]:
                cv2.line(skeleton_frame, smoothed_points[part_a], smoothed_points[part_b], (0, 0, 0), 2)
                cv2.circle(skeleton_frame, smoothed_points[part_a], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(skeleton_frame, smoothed_points[part_b], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)

                cv2.line(overlay_frame, smoothed_points[part_a], smoothed_points[part_b], (0, 255, 0), 2)
                cv2.circle(overlay_frame, smoothed_points[part_a], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(overlay_frame, smoothed_points[part_b], 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

        if out_skeleton is None:
            out_skeleton = cv2.VideoWriter(skeleton_video_path, fourcc, 30.0, (frame_width, frame_height))

        if out_overlay is None:
            out_overlay = cv2.VideoWriter(overlay_video_path, fourcc, 30.0, (frame_width, frame_height))

        out_skeleton.write(skeleton_frame)
        out_overlay.write(overlay_frame)

    with open(output_data_path, 'w') as f:
        json.dump(pose_data, f)

    cap.release()
    if out_skeleton is not None:
        out_skeleton.release()
    if out_overlay is not None:
        out_overlay.release()

def process_folder(input_folder, output_folder, skeleton_video_folder, overlay_video_folder):
    videos = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.mp4', '.mov'))]
    for idx, video in enumerate(videos):
        input_video_path = os.path.join(input_folder, video)
        output_data_path = os.path.join(output_folder, f"{os.path.splitext(video)[0]}.json")
        skeleton_video_path = os.path.join(skeleton_video_folder, f"{os.path.splitext(video)[0]}_skeleton.mp4")
        overlay_video_path = os.path.join(overlay_video_folder, f"{os.path.splitext(video)[0]}_overlay.mp4")
        process_video(input_video_path, output_data_path, skeleton_video_path, overlay_video_path)
        print(f"Processed {idx + 1}/{len(videos)}: {video}")

if __name__ == "__main__":
    input_folder = "input_videos"
    output_folder = "pose_data"
    skeleton_video_folder = "skeleton_videos"
    overlay_video_folder = "overlay_videos"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(skeleton_video_folder, exist_ok=True)
    os.makedirs(overlay_video_folder, exist_ok=True)
    process_folder(input_folder, output_folder, skeleton_video_folder, overlay_video_folder)
