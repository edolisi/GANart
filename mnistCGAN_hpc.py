import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Dropout, Flatten, Input, Conv2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras import backend

backend.set_image_dim_ordering('th')
np.random.seed(1000)
randomDim = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]
adam = Adam(lr=0.0002, beta_1=0.5)

noise = Input(shape=(randomDim,))
img_1 = Input(shape=(1,28,28))
n1 = Dense(64*7*7, kernel_initializer=initializers.RandomNormal(stddev=0.02))(noise)
n1 = LeakyReLU(alpha=0.2)(n1)
n1 = Reshape((64, 7, 7))(n1)
n1 = UpSampling2D(size=(2,2))(n1)
d1 = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(img_1)
d1 = LeakyReLU(alpha=0.2)(d1)
dn1 = Concatenate(axis=1)([d1, n1])
dn2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same')(dn1)
dn2 = LeakyReLU(alpha=0.2)(dn2)
dn2 = UpSampling2D(size=(2,2))(dn2)
output_img = Conv2D(1, kernel_size=(5,5), padding='same', activation='tanh')(dn2)
generator = Model([noise, img_1], output_img)
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same', input_shape=(2, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.trainable = False
ganNoise = Input(shape=(randomDim,))
ganImage = Input(shape=(1,28,28))
x = generator([ganNoise, ganImage])
disInput = Concatenate(axis=1)([ganImage, x])
ganOutput = discriminator(disInput)
gan = Model(inputs=[ganNoise, ganImage], outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

def saveModels(epoch):
    generator.save('dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('dcgan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = 16263 // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        
        X_train_0 = X_train[y_train==0,:,:,:]
        X_train_0 = X_train_0[np.random.choice(5923,5421,replace=False),:,:,:]
        X_train_1 = X_train[y_train==1,:,:,:]
        X_train_1 = X_train_1[np.random.choice(6742,5421,replace=False),:,:,:]
        X_train_2 = X_train[y_train==2,:,:,:]
        X_train_2 = X_train_2[np.random.choice(5958,5421,replace=False),:,:,:]
        X_train_3 = X_train[y_train==3,:,:,:]
        X_train_3 = X_train_3[np.random.choice(6131,5421,replace=False),:,:,:]
        #X_train_4 = X_train[y_train==4,:,:,:]
        #X_train_4 = X_train_4[np.random.choice(5842,5421,replace=False),:,:,:]
        #X_train_5 = X_train[y_train==5,:,:,:]
        #X_train_5 = X_train_5[np.random.choice(5421,5421,replace=False),:,:,:]
        #X_train_6 = X_train[y_train==6,:,:,:]
        #X_train_6 = X_train_6[np.random.choice(5918,5421,replace=False),:,:,:]
        #X_train_7 = X_train[y_train==7,:,:,:]
        #X_train_7 = X_train_7[np.random.choice(6265,5421,replace=False),:,:,:]
        #X_train_8 = X_train[y_train==8,:,:,:]
        #X_train_8 = X_train_8[np.random.choice(5851,5421,replace=False),:,:,:]
        #X_train_9 = X_train[y_train==9,:,:,:]
        #X_train_9 = X_train_9[np.random.choice(5949,5421,replace=False),:,:,:]
        #X_minus = np.concatenate([X_train_0,X_train_1,X_train_2,X_train_3,X_train_4,X_train_4,X_train_6,X_train_7,X_train_8])
        #X_plus = np.concatenate([X_train_1,X_train_2,X_train_3,X_train_4,X_train_4,X_train_6,X_train_7,X_train_8,X_train_9])
        X_minus = np.concatenate([X_train_0,X_train_1,X_train_2])
        X_plus = np.concatenate([X_train_1,X_train_2,X_train_3])
        
        for _ in tqdm(range(batchCount)):
            # get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch_1_G = X_minus[np.random.randint(0, X_minus.shape[0], size=batchSize)]
            
            idx_Batch_D = np.random.randint(0, X_minus.shape[0], size=batchSize)
            imageBatch_1_D = X_minus[idx_Batch_D]
            imageBatch_2_D = X_plus[idx_Batch_D]
            
            # generate fake MNIST images
            generatedImages = generator.predict([noise, imageBatch_1_G])
            X_1 = np.concatenate([imageBatch_1_D, imageBatch_1_G])
            X_2 = np.concatenate([imageBatch_2_D, generatedImages])

            # labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # one-sided label smoothing
            yDis[:batchSize] = 0.9

            # train discriminator
            discriminator.trainable = True
            X = np.concatenate([X_1,X_2], axis=1)
            dloss = discriminator.train_on_batch(X, yDis)

            # train generator (using combined G-D, 'gan')
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch_1_G = X_minus[np.random.randint(0, X_minus.shape[0], size=batchSize)]
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch([noise, imageBatch_1_G], yGen)

        # store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        
        # save model and generated images
        # change '2' to higher values to save less often
        if e == 1 or e % 20 == 0:
            saveModels(e)

    # plot losses from every epoch
    np.savetxt('gLoss.txt', np.array(gLosses), fmt='%f')
    np.savetxt('dLoss.txt', np.array(dLosses), fmt='%f')

if __name__ == '__main__':
    train(200, 128)