import cv2
import numpy as np

# Paths to the model files
proto_file = "pose_deploy_linevec.prototxt"
weights_file = "pose_iter_440000.caffemodel"

n_points = 15
pose_pairs = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [1, 5], [5, 6], [6, 7], [1, 14],
    [14, 8], [8, 9], [9, 10], [14, 11],
    [11, 12], [12, 13]
]

# Load the network
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

def process_video(input_video_path, output_video_path1, output_video_path2):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) // 2

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(output_video_path1, fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_video_path2, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every other frame to reduce frame rate by half
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # Prepare the frame to feed to the network
        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inp)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []
        for i in range(n_points):
            # Confidence map of corresponding body's part
            prob_map = output[0, i, :, :]

            # Find global maxima of the prob_map
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            # Scale the point to fit on the original image
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H

            if prob > 0.1:  # If the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Create a white background for skeleton only output
        skeleton_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Draw skeleton
        for pair in pose_pairs:
            part_a = pair[0]
            part_b = pair[1]

            if points[part_a] and points[part_b]:
                cv2.line(frame, points[part_a], points[part_b], (0, 255, 255), 2)
                cv2.line(skeleton_frame, points[part_a], points[part_b], (0, 0, 0), 2)
                cv2.circle(frame, points[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[part_b], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(skeleton_frame, points[part_a], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(skeleton_frame, points[part_b], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)

        # Write the frames to the output videos
        out1.write(frame)
        out2.write(skeleton_frame)

    # Release everything if job is finished
    cap.release()
    out1.release()
    out2.release()

if __name__ == "__main__":
    input_video_path = "input.mp4"  # Update with your input video path
    output_video_path1 = "output_with_pose.avi"  # Update with your desired output video path
    output_video_path2 = "output_skeleton.avi"  # Update with your desired output video path
    process_video(input_video_path, output_video_path1, output_video_path2)
