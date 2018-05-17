# DCGAN applied to the MNIST dataset (python 3.6)

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend
from keras import initializers

# set keras input format as "theano"-style
backend.set_image_dim_ordering('th')
# set seed
np.random.seed(1000)
# dimension of random input of generator
randomDim = 10

# load MNIST dataset and extract 1/4 of it for faster training
(X_train, y_train), (X_test, y_test) = mnist.load_data()
idx_tr = np.random.choice(60000, 15000, replace=False)
idx_te = np.random.choice(10000, 2500, replace=False)
X_train = X_train[idx_tr,:,:]
y_train = y_train[idx_tr]
X_test = X_test[idx_te,:,:]
y_test = y_test[idx_te]
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]

# set hyperparameters of adam optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# define conolutional generator
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

# define conolutional discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# define combined generator-discriminator for training of generator
# discriminator weights are frozen during generator training
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# function that produces plot of losses of G and D
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # change directory to desired path
    plt.savefig('C:/Users/Edoardo/Desktop/Project/mnistGANs/images/dcgan_loss_epoch_%d.png' % epoch)

# function that produces plot of 100 generated fake MNIST samples
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    # change directory to desired path
    plt.savefig('C:/Users/Edoardo/Desktop/Project/mnistGANs/images/dcgan_generated_image_epoch_%d.png' % epoch)

# function that saves the model parameters
def saveModels(epoch):
    generator.save('C:/Users/Edoardo/Desktop/Project/mnistGANs/models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('C:/Users/Edoardo/Desktop/Project/mnistGANs/models/dcgan_discriminator_epoch_%d.h5' % epoch)

# training function for both G and D
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
        if e == 1 or e % 2 == 0:
            plotGeneratedImages(e)
            saveModels(e)

    # plot losses from every epoch
    plotLoss(e)

if __name__ == '__main__':
    train(30, 128)