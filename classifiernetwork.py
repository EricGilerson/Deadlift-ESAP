import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, LeakyReLU, Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential

n_points = 18

def normalize_data(data):
    max_value = 1.0
    normalized_data = np.where(data != -1, data / max_value, -1)
    return normalized_data

def load_pose_data(data_folder, form_label):
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
                            flat_pose.extend([-1, -1])
                    if len(flat_pose) == n_points * 2:
                        data.append(flat_pose)
                        labels.append(form_label)
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
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main_training_cycle_classifier():
    good_form_folder_front = "model/pose_data/good_front"
    bad_form_folder_front = "model/pose_data/bad_front"
    good_form_folder_side = "model/pose_data/good_side"
    bad_form_folder_side = "model/pose_data/bad_side"
    good_form_folder_middle = "model/pose_data/good_middle"
    bad_form_folder_middle = "model/pose_data/bad_middle"

    good_data_front, good_labels_front = load_pose_data(good_form_folder_front, 1)
    bad_data_front, bad_labels_front = load_pose_data(bad_form_folder_front, 0)
    good_data_side, good_labels_side = load_pose_data(good_form_folder_side, 1)
    bad_data_side, bad_labels_side = load_pose_data(bad_form_folder_side, 0)
    good_data_middle, good_labels_middle = load_pose_data(good_form_folder_middle, 1)
    bad_data_middle, bad_labels_middle = load_pose_data(bad_form_folder_middle, 0)

    data_classification = np.concatenate((good_data_front, bad_data_front, good_data_side, bad_data_side, good_data_middle, bad_data_middle), axis=0)
    labels_classification = np.concatenate((good_labels_front, bad_labels_front, good_labels_side, bad_labels_side, good_labels_middle, bad_labels_middle), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data_classification, labels_classification, test_size=0.2, random_state=42)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    model = create_classification_model()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Classification Model - Test Accuracy: {accuracy * 100:.2f}%")

    model.save('pose_classification_model.h5')

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Classification Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('classification_model_accuracy.png')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Classification Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('classification_model_loss.png')
    plt.show()

if __name__ == '__main__':
    main_training_cycle_classifier()
