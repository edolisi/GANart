import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend

backend.set_image_dim_ordering('th')
randomDim = 100

def plotGeneratedImages(epoch, examples=49, dim=(7, 7), figsize=(7, 7)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = (generator.predict(noise)*127.5 + 127.5).astype(np.uint8)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i,:,:,:].transpose((1,2,0)))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('dcgan_epoch_%d.png' % epoch)
    
np.random.seed(1000)
generator = load_model('dcgan_generator_epoch_100.h5')
plotGeneratedImages(100)