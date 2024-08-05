import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.layers import BatchNormalization, LeakyReLU, Dropout, Dense, Input, Conv2D, GlobalAveragePooling2D, Flatten, Reshape
from keras.models import Model
from keras.src.layers import Multiply, Lambda
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
import altModels
from altModels import *

n_points = 18
temporal_dim = 10
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def normalize_data(data):
    max_value = np.max(data[data != -1])
    normalized_data = np.where(data != -1, data / max_value, -1)
    return normalized_data

def load_pose_data(data_folder):
    data = []
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
                    else:
                        print(f"Incomplete data in frame {frame_index} of file {filename}: expected {n_points * 2} elements, got {len(flat_pose)}")
                else:
                    print(f"Incomplete data in frame {frame_index} of file {filename}")
    data = np.array(data)
    if data.size % (n_points * 2) != 0:
        raise ValueError(f"Data size {data.size} is not a multiple of {n_points * 2}")
    return data

def mask_invalid_keypoints(data):
    mask = (data != -1).astype(np.float32)
    return mask

def custom_loss(y_true, y_pred):
    mask = K.cast(y_true != -1, dtype=K.floatx())
    euclidean_loss = K.sum(mask[:, :, 0] * mask[:, :, 1] * K.sqrt(K.square(y_true[:, :, 0] - y_pred[:, :, 0]) + K.square(y_true[:, :, 1] - y_pred[:, :, 1]))) / K.sum(mask[:, :, 0] * mask[:, :, 1])
    return euclidean_loss

def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.8
    epochs_drop = 25.0
    lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lr

def main_training_cycle_deviation():
    good_form_folder_front = "model/pose_data/good_front"
    good_form_folder_side = "model/pose_data/good_side"
    good_form_folder_middle = "model/pose_data/good_middle"

    good_data_front = load_pose_data(good_form_folder_front)
    good_data_side = load_pose_data(good_form_folder_side)
    good_data_middle = load_pose_data(good_form_folder_middle)

    good_data_combined = np.concatenate((good_data_front, good_data_side, good_data_middle), axis=0)

    data_combined = good_data_combined
    labels_combined = good_data_combined

    X_train, X_test, y_train, y_test = train_test_split(data_combined, labels_combined, test_size=0.2, random_state=42)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    y_train = normalize_data(y_train)
    y_test = normalize_data(y_test)

    mask_train = mask_invalid_keypoints(X_train)
    mask_test = mask_invalid_keypoints(X_test)

    # Calculate the number of samples
    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]

    input_shape = (n_points, 2, 1)
    X_train = X_train.reshape(num_samples_train, n_points, 2, 1)
    X_test = X_test.reshape(num_samples_test, n_points, 2, 1)
    y_train = y_train.reshape(num_samples_train, n_points, 2, 1)
    y_test = y_test.reshape(num_samples_test, n_points, 2, 1)
    mask_train = mask_train.reshape(num_samples_train, n_points, 2, 1)
    mask_test = mask_test.reshape(num_samples_test, n_points, 2, 1)

    model = altModels.create_2Dconv_model(input_shape)

    reduce_lr = LearningRateScheduler(step_decay)

    history = model.fit([X_train, mask_train], y_train, epochs=750, batch_size=32, validation_data=([X_test, mask_test], y_test),
                        callbacks=[reduce_lr])
    loss, mae = model.evaluate([X_test, mask_test], y_test)
    print(f"Deviation Model - Test MAE: {mae:.2f}")

    model.save('pose_deviation_model.h5')

    plt.figure()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Deviation Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('deviation_model_mae.png')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Deviation Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('deviation_model_loss.png')
    plt.show()

    return model, history, X_test, y_test

def analyze_results(model, X_test, y_test):
    mask_test = mask_invalid_keypoints(X_test).reshape(-1, n_points, 2, 1)
    y_pred = model.predict([X_test, mask_test])

    for i in range(5):  # Display 5 example predictions
        print(f"Example {i + 1}:")
        print("Predicted:", y_pred[i].reshape(-1))
        print("Actual:", y_test[i].reshape(-1))

    plt.figure()
    plt.scatter(y_test.flatten(), y_pred.flatten())
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Deviations')
    plt.show()

if __name__ == '__main__':
    model, history, X_test, y_test = main_training_cycle_deviation()
    analyze_results(model, X_test, y_test)
