import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, LeakyReLU, Dropout, Dense, Input, Conv1D, GlobalAveragePooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

n_points = 18

def normalize_data(data):
    max_value = 1.0
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
                        print(f"Incomplete data in frame {frame_index} of file {filename}")
                else:
                    print(f"Incomplete data in frame {frame_index} of file {filename}")
    return np.array(data)

def create_deviation_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_points * 2, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def main_training_cycle_deviation():
    good_form_folder_front = "model/pose_data/good_front"
    good_form_folder_side = "model/pose_data/good_side"
    good_form_folder_middle = "model/pose_data/good_middle"

    good_data_front = load_pose_data(good_form_folder_front)
    good_data_side = load_pose_data(good_form_folder_side)
    good_data_middle = load_pose_data(good_form_folder_middle)

    # Combine good data for training the deviation model
    data_combined = np.concatenate((good_data_front, good_data_side, good_data_middle), axis=0)
    labels_combined = data_combined  # Use good form data as labels

    X_train, X_test, y_train, y_test = train_test_split(data_combined, labels_combined, test_size=0.2, random_state=42)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    y_train = normalize_data(y_train)
    y_test = normalize_data(y_test)

    input_shape = (n_points * 2, 1)
    X_train = X_train.reshape(-1, n_points * 2, 1)
    X_test = X_test.reshape(-1, n_points * 2, 1)
    y_train = y_train.reshape(-1, n_points * 2, 1)
    y_test = y_test.reshape(-1, n_points * 2, 1)

    model = create_deviation_model(input_shape)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=5000, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    loss, mae = model.evaluate(X_test, y_test)
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

def provide_feedback(classification_model_path, deviation_model_path, poses):
    from tensorflow.keras.models import load_model

    classification_model = load_model(classification_model_path)
    deviation_model = load_model(deviation_model_path)

    normalized_poses = normalize_data(np.array(poses))
    normalized_poses = normalized_poses.reshape(-1, n_points * 2, 1)
    is_correct = classification_model.predict(normalized_poses)
    deviations = deviation_model.predict(normalized_poses)

    feedbacks = []
    for idx, pose in enumerate(poses):
        if is_correct[idx] >= 0.6:
            feedbacks.append(f"Frame {idx}: Good Form ({round(is_correct[idx][0]*100,4)}%)")
        else:
            feedback = [f"Frame {idx}: Bad Form ({round(is_correct[idx][0]*100,4)}%), Suggestions:"]
            corrected_pose = deviations[idx].reshape(-1) * 1.0  # Denormalize if necessary
            feedback.extend([f"Keypoint {i}: x correction = {corrected_pose[2 * i]:.2f}, y correction = {corrected_pose[2 * i + 1]:.2f}" for i in range(n_points)])
            feedbacks.append("\t".join(feedback))
    return feedbacks

if __name__ == '__main__':
    main_training_cycle_deviation()
