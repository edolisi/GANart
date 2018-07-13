import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam
from keras import backend
from keras import initializers
# set keras/tensorflow configuration as channel-first
backend.set_image_dim_ordering('th')

# set parameters and load dataset
randomDim = 100
dLosses = []
gLosses = []
adam = Adam(lr=0.0002, beta_1=0.5)
X_train = np.load('X_13styles.npy')
# set the dataset as channel-first and reshape it between -1 and 1
X_train = X_train.transpose((0,3,1,2))
X_train = X_train.astype(np.float32)/127.5 - 1.0

# generator
g_input = Input(shape=(randomDim,))
g = Dense(256*8*8,
          kernel_initializer=initializers.RandomNormal(stddev=0.02))(g_input)
g = LeakyReLU(0.2)(g)
g = Reshape((256, 8, 8))(g)
g = UpSampling2D(size=(2, 2))(g)
g = Conv2D(128, kernel_size=(5, 5), padding='same')(g)
g = LeakyReLU(0.2)(g)
g = UpSampling2D(size=(2, 2))(g)
g = Conv2D(64, kernel_size=(5, 5), padding='same')(g)
g = LeakyReLU(0.2)(g)
g = UpSampling2D(size=(2, 2))(g)
g_output = Conv2D(3, kernel_size=(5, 5), padding='same', activation='tanh')(g)
generator = Model(g_input, g_output)
generator.compile(loss='binary_crossentropy', optimizer=adam)

# discriminator
d_input = Input(shape=(3,64,64))
d = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
           kernel_initializer=initializers.RandomNormal(stddev=0.02))(d_input)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Flatten()(d)
d_output = Dense(1, activation='sigmoid')(d)
discriminator = Model(d_input, d_output)
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# GAN
discriminator.trainable = False
gan_input = Input(shape=(randomDim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=adam)

# fuction that saves the weights of generator and discriminator
def saveModels(epoch):
    generator.save('generator_epoch_%d.h5' % epoch)
    discriminator.save('discriminator_epoch_%d.h5' % epoch)

# training function
def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0],
                                                   size=batchSize)]

            # generate fake images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # one-sided label smoothing
            yDis[:batchSize] = 0.9

            # train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # train generator (using combined GAN model)
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        
        # save model every 20 epochs
        if e == 1 or e % 20 == 0:
            saveModels(e)

    # save losses from every epoch
    np.savetxt('gLoss.txt', np.array(gLosses), fmt='%f')
    np.savetxt('dLoss.txt', np.array(dLosses), fmt='%f')

if __name__ == '__main__':
    train(200, 128)