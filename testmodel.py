import os
from network import evaluate_pose, correct_pose
from tensorflow.keras.models import load_model

# Path to the folder containing test JSON files
test_json_folder = 'model/pose_data/test'

# Load the model
model = load_model('pose_model.h5')

# Iterate through all JSON files in the folder and evaluate each pose
for filename in os.listdir(test_json_folder):
    if filename.endswith(".json"):
        test_json_path = os.path.join(test_json_folder, filename)
        print(f"Evaluating pose for file: {test_json_path}")
        evaluate_pose(test_json_path, model, correct_pose)
