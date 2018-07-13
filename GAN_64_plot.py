import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend
# set keras/tensorflow configuration as channel-first
backend.set_image_dim_ordering('th')

# set dimension of random noise input to generator
randomDim = 100
# load generator model
generator = load_model('generator_epoch_200.h5')
# produce 49 instances of random noise input
noise = np.random.normal(0, 1, size=[49, randomDim])
# generate images
generatedImages = (generator.predict(noise)*127.5 + 127.5).astype(np.uint8)
# plot the 49 images in a 7-by-7 square
plt.figure(figsize=(7,7))
for i in range(generatedImages.shape[0]):
    plt.subplot(7, 7, i+1)
    plt.imshow(generatedImages[i,:,:,:].transpose((1,2,0)))
    plt.axis('off')
plt.show()