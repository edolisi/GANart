import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense
from keras.layers import UpSampling2D, BatchNormalization, Reshape
from keras import backend as K
# set keras/tensorflow configuration as channel-first
K.set_image_dim_ordering('th')

# set parameters and load the data
batch_size = 32
epochs = 200
X_all = np.load('X_13styles.npy')
X_all = X_all.transpose((0,3,1,2))
X_all = X_all.astype('float32')
# shrink data between 0 and 1 and extract training set and test set
X_all /= 255
X_train = X_all[:8000]
X_test = X_all[8000:]

# encoder
input_img = Input(shape=(3, 64, 64))
x = Conv2D(128, (3,3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Reshape((16*4*4,))(x)
encoded = Dense(100, activation='relu')(x)

# decoder
x = Dense(16*4*4, activation='relu')(encoded)
x = Reshape((16,4,4))(x)
x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(128, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(3, (3,3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

# autoencoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train the autoencoder
history_AE = autoencoder.fit(X_train, X_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, X_test),
                    shuffle=True)

# define encoder and decoder as separate models
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(100,))
deco = autoencoder.layers[-21](encoded_input)
deco = autoencoder.layers[-20](deco)
deco = autoencoder.layers[-19](deco)
deco = autoencoder.layers[-18](deco)
deco = autoencoder.layers[-17](deco)
deco = autoencoder.layers[-16](deco)
deco = autoencoder.layers[-15](deco)
deco = autoencoder.layers[-14](deco)
deco = autoencoder.layers[-13](deco)
deco = autoencoder.layers[-12](deco)
deco = autoencoder.layers[-11](deco)
deco = autoencoder.layers[-10](deco)
deco = autoencoder.layers[-9](deco)
deco = autoencoder.layers[-8](deco)
deco = autoencoder.layers[-7](deco)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)

# encode the test set
z = encoder.predict(X_test)

# save encoder weights and decoder weights
encoder.save('encoder.h5')
decoder.save('decoder.h5')
# save training loss
loss_AE = history_AE.history["loss"]
loss_AE = np.array(loss_AE)
np.savetxt("loss_history.txt", loss_AE, fmt='%f')