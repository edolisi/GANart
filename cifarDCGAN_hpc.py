import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras import backend
from keras import initializers

backend.set_image_dim_ordering('th')
np.random.seed(1000)
randomDim = 100
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#idx_tr = np.random.choice(50000, 10000, replace=False)
#X_train = X_train[idx_tr,:,:]
#y_train = y_train[idx_tr]
X_train = (X_train.astype(np.float32) - 127.5)/127.5
adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(128*8*8, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 8, 8)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(3, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(3, 32, 32), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)
dLosses = []
gLosses = []

def saveModels(epoch):
    generator.save('dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('dcgan_discriminator_epoch_%d.h5' % epoch)

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
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # one-sided label smoothing
            yDis[:batchSize] = 0.9

            # train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # train generator (using combined G-D, 'gan')
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        
        # save model and generated images
        # change '2' to higher values to save less often
        if e == 1 or e % 20 == 0:
            saveModels(e)

if __name__ == '__main__':
    train(200, 128)