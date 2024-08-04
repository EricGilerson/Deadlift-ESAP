import os
import json

import numpy as np

from classifiernetwork import normalize_data, n_points
from tensorflow.keras.models import load_model

# Path to the folder containing test JSON files
test_json_folder = 'model/pose_data/test'

# Load the models
classification_model = load_model('pose_classification_model.h5')
deviation_model = load_model('pose_deviation_model.h5')


def evaluate_pose(test_json_folder, classification_model, deviation_model):
    all_poses = []
    file_frame_indices = []

    for filename in os.listdir(test_json_folder):
        if filename.endswith(".json"):
            test_json_path = os.path.join(test_json_folder, filename)
            with open(test_json_path, 'r') as f:
                poses = json.load(f)
                for frame_index, person in enumerate(poses):
                    if isinstance(person, list) and len(person) > 0:
                        flat_pose = [coord for keypoint in person for coord in (keypoint if keypoint is not None else [-1, -1])]
                        if len(flat_pose) == n_points * 2:
                            all_poses.append(flat_pose)
                            file_frame_indices.append((filename, frame_index))
                        else:
                            print(f"Incomplete data in frame {frame_index} of file {filename}")
                    else:
                        print(f"Incomplete data in frame {frame_index} of file {filename}")

    # Generate feedback for all poses at once
    feedbacks = provide_feedback(classification_model, deviation_model, all_poses)
    for (filename, frame_index), feedback in zip(file_frame_indices, feedbacks):
        print(f"Feedback for frame {frame_index} in file {filename}:\t{feedback}")
def provide_feedback(classification_model, deviation_model, poses):
    normalized_poses = normalize_data(np.array(poses))
    is_correct = classification_model.predict(normalized_poses)
    deviations = deviation_model.predict(normalized_poses)

    feedbacks = []
    for idx, pose in enumerate(poses):
        if is_correct[idx] >= 0.6:
            #feedbacks.append(f"Frame {idx}: Good Form ({round(is_correct[idx][0]*100,4)}%)")

            feedback = [(f"Frame {idx}: Good Form ({round(is_correct[idx][0]*100,4)}%)")]
            feedback.extend([f"Keypoint {i}: x deviation = {deviations[idx][2 * i]:.2f}, y deviation = {deviations[idx][2 * i + 1]:.2f}" for i in range(n_points)])
            feedbacks.append("\t".join(feedback))
        else:
            feedback = [f"Frame {idx}: Bad Form ({round(is_correct[idx][0]*100,4)}%), Suggestions:"]
            feedback.extend([f"Keypoint {i}: x deviation = {deviations[idx][2 * i]:.2f}, y deviation = {deviations[idx][2 * i + 1]:.2f}" for i in range(n_points)])
            feedbacks.append("\t".join(feedback))
    return feedbacks
# Evaluate poses
evaluate_pose(test_json_folder, classification_model, deviation_model)
