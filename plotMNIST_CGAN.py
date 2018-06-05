import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend
from keras.datasets import mnist

backend.set_image_dim_ordering('th')
randomDim = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]
X_train_0 = X_train[y_train==0,:,:,:]
X_train_1 = X_train[y_train==1,:,:,:]
X_train_2 = X_train[y_train==2,:,:,:]
X_train_3 = X_train[y_train==3,:,:,:]

examples = 49
imageBatch = X_train_2[np.random.randint(0, X_train_2.shape[0], size=examples)]

def plotGeneratedImages(epoch, examples=49, dim=(7,7), figsize=(7,7)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    #imageBatch_1_G = X_in[np.random.randint(0, X_in.shape[0], size=examples)]
    generatedImages = generator.predict([noise, imageBatch])
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i,0,:,:])
        plt.axis('off')
    plt.tight_layout()
    # change directory to desired path
    plt.savefig('C:/Users/Edoardo/Desktop/MNICGAN_epoch_%d.png' % epoch)

np.random.seed(1000)
generator = load_model('dcgan_generator_epoch_200.h5')
plotGeneratedImages(200)