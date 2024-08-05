from keras import Model, Input
from keras.src.layers import Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dense, Dropout, Reshape, \
    Lambda, Multiply, GlobalAveragePooling1D, Conv1D, LSTM, Conv3D, GlobalAveragePooling3D, Flatten
from keras.src.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

n_points = 18

def custom_loss(y_true, y_pred):
    mask = K.cast(y_true != -1, dtype=K.floatx())
    euclidean_loss = K.sum(mask[:, :, 0] * mask[:, :, 1] * K.sqrt(K.square(y_true[:, :, 0] - y_pred[:, :, 0]) + K.square(y_true[:, :, 1] - y_pred[:, :, 1]))) / K.sum(mask[:, :, 0] * mask[:, :, 1])
    return euclidean_loss

def create_dense_model(input_shape):
    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)

    x = Flatten()(inputs)
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    mask_reshaped = Reshape((n_points, 2))(mask)
    masked_outputs = Multiply()([outputs, mask_reshaped])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model
#converged after about 500 epochs
#best model with 32 batch size
def create_2Dconv_model(input_shape):
    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    mask_reshaped = Lambda(lambda m: m[:, :, :, 0])(mask)

    masked_outputs = Multiply()([outputs, mask_reshaped])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model

#Use 2D convolutions to process the spatial relationships.
#Follow up with 1D convolutions to capture sequential dependencies.
#converged around 500
def create_conv2d_conv1d_model(input_shape):
    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)

    # 2D Convolutions
    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    # Convert 2D to 1D
    x = Reshape((input_shape[0], -1))(x)

    # 1D Convolutions
    x = Conv1D(256, kernel_size=3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv1D(512, kernel_size=3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    mask_reshaped = Lambda(lambda m: m[:, :, :, 0])(mask)
    masked_outputs = Multiply()([outputs, mask_reshaped])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model

#Use LSTMs to capture temporal dependencies if the data is sequential (e.g., video frames).
#doesn't really converge but 500 epochs is enough
def create_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Reshape((input_shape[0], -1))(x)

    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(512, return_sequences=False)(x)

    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    mask_reshaped = Lambda(lambda m: m[:, :, :, 0])(mask)
    masked_outputs = Multiply()([outputs, mask_reshaped])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model

#Integrate attention layers to focus on important features and keypoints.
#converged after about 400-500 epochs
def create_attention_model(input_shape):
    from keras.layers import Attention

    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Reshape((input_shape[0], -1))(x)

    x = LSTM(256, return_sequences=True)(x)
    x = Attention()([x, x])
    x = LSTM(512, return_sequences=False)(x)

    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    mask_reshaped = Lambda(lambda m: m[:, :, :, 0])(mask)
    masked_outputs = Multiply()([outputs, mask_reshaped])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model

#Use transformer models to capture relationships across the entire sequence of keypoints.
#converged after about 500 epochs
def create_transformer_model(input_shape):
    from keras.layers import MultiHeadAttention, LayerNormalization, Add

    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Reshape((input_shape[0], -1))(x)

    # Add Transformer layers
    attention = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    attention = Add()([x, attention])
    attention = LayerNormalization()(attention)

    x = GlobalAveragePooling1D()(attention)
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    mask_reshaped = Lambda(lambda m: m[:, :, :, 0])(mask)
    masked_outputs = Multiply()([outputs, mask_reshaped])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model

#Use 3D convolutions if you have temporal data (e.g., sequences of frames) to capture spatiotemporal features.
#converged around 350-400 epochs
def create_3d_conv_model(input_shape):
    inputs = Input(shape=input_shape)
    mask = Input(shape=input_shape)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv3D(256, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(n_points * 2, activation='linear')(x)
    outputs = Reshape((n_points, 2))(x)

    masked_outputs = Multiply()([outputs, mask])

    model = Model(inputs=[inputs, mask], outputs=masked_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])
    return model
