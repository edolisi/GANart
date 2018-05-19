import numpy as np
from matplotlib import pyplot as plt

from keras.models import load_model
from keras import backend

backend.set_image_dim_ordering('th')
randomDim = 10

def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('dcgan_epoch_%d.png' % epoch)
    
np.random.seed(1000)
generator = load_model('dcgan_generator_epoch_200.h5')
plotGeneratedImages(99)