import os
import json
import numpy as np
from classifiernetwork import normalize_data, n_points
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from deviationmodel import custom_loss

# Path to the folder containing test JSON files
test_json_folder = 'model/pose_data/test'
def custom_loss(y_true, y_pred):
    mask = K.cast(y_true != -1, dtype=K.floatx())
    euclidean_loss = K.sum(mask[:, :, 0] * mask[:, :, 1] * K.sqrt(K.square(y_true[:, :, 0] - y_pred[:, :, 0]) + K.square(y_true[:, :, 1] - y_pred[:, :, 1]))) / K.sum(mask[:, :, 0] * mask[:, :, 1])
    return euclidean_loss

# Load the models
classification_model = load_model('pose_classification_model.h5')
deviation_model = load_model('pose_deviation_model.h5', custom_objects={'custom_loss': custom_loss})
def mask_invalid_keypoints(data):
    mask = (data != -1).astype(np.float32)
    return mask

def provide_feedback(classification_model, deviation_model, poses):
    normalized_poses = normalize_data(np.array(poses))
    mask = mask_invalid_keypoints(normalized_poses).reshape(-1, n_points, 2, 1)
    normalized_poses_flat = normalized_poses.reshape(-1, n_points * 2)  # For the classification model
    normalized_poses_reshaped = normalized_poses.reshape(-1, n_points, 2, 1)  # For the deviation model

    is_correct = classification_model.predict(normalized_poses_flat)
    deviations = deviation_model.predict([normalized_poses_reshaped, mask])

    feedbacks = []
    for idx, pose in enumerate(poses):
        if is_correct[idx] >= 0.6:
            feedback = [f"Frame {idx}: Good Form ({round(is_correct[idx][0] * 100, 4)}%), Suggestions:"]
            corrected_pose = deviations[idx].reshape(-1)
            feedback.extend(
                [f"Keypoint {i}: distance deviation = {np.linalg.norm(corrected_pose[2 * i:2 * i + 2]):.2f}" for i in
                 range(n_points)])
            feedbacks.append("\t".join(feedback))
        else:
            feedback = [f"Frame {idx}: Bad Form ({round(is_correct[idx][0] * 100, 4)}%), Suggestions:"]
            corrected_pose = deviations[idx].reshape(-1)
            feedback.extend([f"Keypoint {i}: distance deviation = {np.linalg.norm(corrected_pose[2 * i:2 * i + 2]):.2f}" for i in range(n_points)])
            feedbacks.append("\t".join(feedback))
    return feedbacks

# Function to evaluate poses from JSON files in a folder
def evaluate_pose(test_json_folder, classification_model, deviation_model):
    for filename in os.listdir(test_json_folder):
        if filename.endswith(".json"):
            test_json_path = os.path.join(test_json_folder, filename)
            with open(test_json_path, 'r') as f:
                poses = json.load(f)
                all_poses = []
                frame_indices = []
                for frame_index, person in enumerate(poses):
                    if isinstance(person, list) and len(person) > 0:
                        flat_pose = [coord for keypoint in person for coord in (keypoint if keypoint is not None else [-1, -1])]
                        if len(flat_pose) == n_points * 2:
                            all_poses.append(flat_pose)
                            frame_indices.append(frame_index)
                        else:
                            print(f"Incomplete data in frame {frame_index} of file {filename}")
                    else:
                        print(f"Incomplete data in frame {frame_index} of file {filename}")

                # Generate feedback for poses in the current file
                feedbacks = provide_feedback(classification_model, deviation_model, all_poses)
                for frame_index, feedback in zip(frame_indices, feedbacks):
                    print(f"Feedback for frame {frame_index} in file {filename}:\t{feedback}")

# Evaluate poses
evaluate_pose(test_json_folder, classification_model, deviation_model)