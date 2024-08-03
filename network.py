import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization, LeakyReLU
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow_estimator.python.estimator import early_stopping

n_points = 18

def normalize_data(data):
    # Normalize the keypoint data, assuming keypoints are within a certain range (e.g., 0 to 1)
    max_value = 1.0
    normalized_data = np.where(data != -1, data / max_value, -1)
    return normalized_data

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
                            flat_pose.extend([-1, -1])  # Use a special placeholder for missing keypoints
                    if len(flat_pose) == n_points * 2:  # Ensure there are 18 keypoints with 2 coordinates each
                        data.append(flat_pose)
                        labels.append([form_label, angle_label])
                    else:
                        print(f"Incomplete data in frame {frame_index} of file {filename}")
                else:
                    print(f"Incomplete data in frame {frame_index} of file {filename}")
    return np.array(data), np.array(labels)



def create_model():
    model = Sequential([
        Flatten(input_shape=(n_points * 2,)),  # 2 coordinates (x, y) per keypoint
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(128, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: good or bad form
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Function to calculate deviations and provide feedback
def provide_feedback(test_pose, correct_pose, threshold=10):
    deviations = np.linalg.norm(test_pose.reshape(n_points, 2) - correct_pose.reshape(n_points, 2), axis=1)
    feedback = []
    for i, deviation in enumerate(deviations):
        if deviation > threshold:
            feedback.append(f"Keypoint {i} is off by {deviation:.2f} units.")
    return feedback

# Function to predict and provide suggestions
def evaluate_pose(pose_json, model, correct_pose, threshold=10):
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
                            flat_pose.extend([-1, -1])  # Use a special placeholder for missing keypoints
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
    pose_data = normalize_data(pose_data)

    predictions = model.predict(pose_data)
    for i, pred in enumerate(predictions):
        form_pred = pred[0]
        percentage = form_pred * 100
        if form_pred < 0.6:
            feedback = provide_feedback(pose_data[i], correct_pose, threshold)
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



if __name__ == "__main__":
    # Example code to load data, train, and save the model
    good_form_folder_front = "model/pose_data/good_front"
    bad_form_folder_front = "model/pose_data/bad_front"
    good_form_folder_side = "model/pose_data/good_side"
    bad_form_folder_side = "model/pose_data/bad_side"
    good_form_folder_middle = "model/pose_data/good_middle"
    bad_form_folder_middle = "model/pose_data/bad_middle"

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

    # Normalize training and testing data
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    model = create_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Train the neural network
    history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    model.save('pose_model.h5')

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('model_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('model_loss.png')
    plt.show()