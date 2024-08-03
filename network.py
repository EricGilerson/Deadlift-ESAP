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

def create_classification_model():
    model = Sequential()
    model.add(Dense(256, input_dim=n_points*2))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification for correct/incorrect form
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_deviation_model():
    model = Sequential()
    model.add(Dense(256, input_dim=n_points*2))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(n_points * 2, activation='linear'))  # Output deviations for each keypoint
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def provide_feedback(classification_model, deviation_model, poses):
    normalized_poses = normalize_data(np.array(poses))
    is_correct = classification_model.predict(normalized_poses)
    deviations = deviation_model.predict(normalized_poses)

    feedbacks = []
    for idx, pose in enumerate(poses):
        if is_correct[idx] >= 0.6:
            feedbacks.append(f"Frame {idx}: Good Form ({round(is_correct[idx][0]*100,4)}%)")
        else:
            feedback = [f"Frame {idx}: Bad Form ({round(is_correct[idx][0]*100,4)}%), Suggestions:"]
            feedback.extend([f"Keypoint {i}: x deviation = {deviations[idx][2 * i]:.2f}, y deviation = {deviations[idx][2 * i + 1]:.2f}" for i in range(n_points)])
            feedbacks.append("\t".join(feedback))
    return feedbacks
def main_training_cycle():
    good_form_folder_front = "model/pose_data/good_front"
    bad_form_folder_front = "model/pose_data/bad_front"
    good_form_folder_side = "model/pose_data/good_side"
    bad_form_folder_side = "model/pose_data/bad_side"
    good_form_folder_middle = "model/pose_data/good_middle"
    bad_form_folder_middle = "model/pose_data/bad_middle"

    # Load data for classification model
    good_data_front, good_labels_front = load_pose_data(good_form_folder_front, 1, 0)
    bad_data_front, bad_labels_front = load_pose_data(bad_form_folder_front, 0, 0)
    good_data_side, good_labels_side = load_pose_data(good_form_folder_side, 1, 1)
    bad_data_side, bad_labels_side = load_pose_data(bad_form_folder_side, 0, 1)
    good_data_middle, good_labels_middle = load_pose_data(good_form_folder_middle, 1, 2)
    bad_data_middle, bad_labels_middle = load_pose_data(bad_form_folder_middle, 0, 2)

    # Combine and split the dataset for classification model
    data_classification = np.concatenate((good_data_front, bad_data_front, good_data_side, bad_data_side, good_data_middle, bad_data_middle), axis=0)
    labels_classification = np.concatenate((good_labels_front[:, 0], bad_labels_front[:, 0], good_labels_side[:, 0], bad_labels_side[:, 0], good_labels_middle[:, 0], bad_labels_middle[:, 0]), axis=0)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(data_classification, labels_classification, test_size=0.2, random_state=42)

    # Normalize training and testing data for classification model
    X_train_cls = normalize_data(X_train_cls)
    X_test_cls = normalize_data(X_test_cls)

    # Load data for deviation model
    data_deviation = data_classification
    labels_deviation = data_classification
    X_train_dev, X_test_dev, y_train_dev, y_test_dev = train_test_split(data_deviation, labels_deviation, test_size=0.2, random_state=42)

    # Normalize training and testing data for deviation model
    X_train_dev = normalize_data(X_train_dev)
    X_test_dev = normalize_data(X_test_dev)
    y_train_dev = normalize_data(y_train_dev)
    y_test_dev = normalize_data(y_test_dev)

    classification_model = create_classification_model()
    deviation_model = create_deviation_model()

    reduce_lr_cls = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    reduce_lr_dev = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Train the classification model
    history_cls = classification_model.fit(X_train_cls, y_train_cls, epochs=2000, validation_data=(X_test_cls, y_test_cls), callbacks=[reduce_lr_cls])

    # Train the deviation model
    history_dev = deviation_model.fit(X_train_dev, y_train_dev, epochs=20000, validation_data=(X_test_dev, y_test_dev), callbacks=[reduce_lr_dev])

    # Evaluate the classification model
    loss_cls, accuracy_cls = classification_model.evaluate(X_test_cls, y_test_cls)
    print(f"Classification Model - Test Accuracy: {accuracy_cls * 100:.2f}%")

    # Evaluate the deviation model
    loss_dev, mae_dev = deviation_model.evaluate(X_test_dev, y_test_dev)
    print(f"Deviation Model - Test MAE: {mae_dev:.2f}")

    # Save the trained models
    classification_model.save('pose_classification_model.h5')
    deviation_model.save('pose_deviation_model.h5')

    # Plot training & validation accuracy values for classification model
    plt.figure()
    plt.plot(history_cls.history['accuracy'])
    plt.plot(history_cls.history['val_accuracy'])
    plt.title('Classification Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('classification_model_accuracy.png')
    plt.show()

    # Plot training & validation loss values for classification model
    plt.figure()
    plt.plot(history_cls.history['loss'])
    plt.plot(history_cls.history['val_loss'])
    plt.title('Classification Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('classification_model_loss.png')
    plt.show()

    # Plot training & validation MAE values for deviation model
    plt.figure()
    plt.plot(history_dev.history['mae'])
    plt.plot(history_dev.history['val_mae'])
    plt.title('Deviation Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('deviation_model_mae.png')
    plt.show()

    # Plot training & validation loss values for deviation model
    plt.figure()
    plt.plot(history_dev.history['loss'])
    plt.plot(history_dev.history['val_loss'])
    plt.title('Deviation Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('deviation_model_loss.png')
    plt.show()

if __name__ == '__main__':
    main_training_cycle()
