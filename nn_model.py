import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

n_points = 18

# Load pose data from JSON files
def load_pose_data(data_folder, form_label, angle_label):
    data = []
    labels = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r') as f:
                poses = json.load(f)

            for frame_index, person in enumerate(poses):
                if isinstance(person, list) and len(person) > 0:
                    flat_pose = []
                    for keypoint in person:
                        if keypoint is not None:
                            flat_pose.extend(keypoint)
                        else:
                            flat_pose.extend([0, 0])  # Use a placeholder for missing keypoints
                    if len(flat_pose) == n_points * 2:  # Ensure there are 18 keypoints with 2 coordinates each
                        data.append(flat_pose)
                        labels.append([form_label, angle_label])
                    else:
                        print(f"Incomplete data in frame {frame_index} of file {filename}")
                else:
                    print(f"Incomplete data in frame {frame_index} of file {filename}")
    return np.array(data), np.array(labels)

# Prepare the dataset
good_form_folder_front = "/content/drive/MyDrive/pose_data/good_front"
bad_form_folder_front = "/content/drive/MyDrive/pose_data/bad_front"
good_form_folder_side = "/content/drive/MyDrive/pose_data/good_side"
bad_form_folder_side = "/content/drive/MyDrive/pose_data/bad_side"
good_form_folder_middle = "/content/drive/MyDrive/pose_data/good_middle"
bad_form_folder_middle = "/content/drive/MyDrive/pose_data/bad_middle"

good_data_front, good_labels_front = load_pose_data(good_form_folder_front, 1, 0)
bad_data_front, bad_labels_front = load_pose_data(bad_form_folder_front, 0, 0)
good_data_side, good_labels_side = load_pose_data(good_form_folder_side, 1, 1)
bad_data_side, bad_labels_side = load_pose_data(bad_form_folder_side, 0, 1)
good_data_middle, good_labels_middle = load_pose_data(good_form_folder_middle, 1, 2)
bad_data_middle, bad_labels_middle = load_pose_data(bad_form_folder_middle, 0, 2)

# Combine and split the dataset
data = np.concatenate((good_data_front, bad_data_front, good_data_side, bad_data_side, good_data_middle, bad_data_middle), axis=0)
labels = np.concatenate((good_labels_front, bad_labels_front, good_labels_side, bad_labels_side, good_labels_middle, bad_labels_middle), axis=0)
X_train, X_test, y_train, y_test = train_test_split(data, labels[:, 0], test_size=0.2, random_state=42)  # Only use form label for training

# Define the neural network
model = Sequential([
    Flatten(input_shape=(n_points * 2,)),  # 2 coordinates (x, y) per keypoint
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: good or bad form
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('pose_model.h5')

# Function to calculate deviations and provide feedback
def provide_feedback(test_pose, correct_pose, threshold=10):
    deviations = np.linalg.norm(test_pose.reshape(n_points, 2) - correct_pose.reshape(n_points, 2), axis=1)
    feedback = []
    for i, deviation in enumerate(deviations):
        if deviation > threshold:
            feedback.append(f"Keypoint {i} is off by {deviation:.2f} units.")
    return feedback

# Function to predict and provide suggestions
def evaluate_pose(pose_json, model, correct_pose):
    pose_data = []
    with open(pose_json) as f:
        poses = json.load(f)
        if isinstance(poses, list):
            for person in poses:
                if isinstance(person, list):
                    flat_pose = []
                    for keypoint in person:
                        if keypoint is not None:
                            flat_pose.extend(keypoint)
                        else:
                            flat_pose.extend([0, 0])  # Use a placeholder for missing keypoints
                    if len(flat_pose) == n_points * 2:
                        pose_data.append(flat_pose)
                    else:
                        print(f"Skipping incomplete data in file {pose_json}")
                else:
                    print(f"Skipping unexpected structure in file {pose_json}: person is not a list")
        else:
            print(f"Skipping file {pose_json}: poses is not a list")

    if len(pose_data) == 0:
        print("No valid pose data found.")
        return

    pose_data = np.array(pose_data, dtype=np.float32)
    predictions = model.predict(pose_data)
    for i, pred in enumerate(predictions):
        form_pred = pred[0]
        percentage = form_pred * 100
        if form_pred < 0.6:
            feedback = provide_feedback(pose_data[i], correct_pose)
            print(f"Pose {i}: Bad form ({percentage:.2f}% confidence). Suggestions: {feedback}")
        else:
            print(f"Pose {i}: Good form ({percentage:.2f}% confidence).")

# Load the correct form keypoints for feedback (example format)
correct_pose = np.array([
    # Add the correct form keypoints here
    [x, y] for x, y in [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
        (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
        (16, 16), (17, 17)
    ]
]).flatten()




# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
