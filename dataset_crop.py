import numpy as np
from PIL import Image
import glob

# initialize image array, names list and iteration number
X_train = np.empty([0,64,64,3], dtype=np.uint8)
namelist = []
i = 0
# iterate through all jpgs in the current folder
for filename in glob.glob('*.jpg'):
    try:
        # open the image and extract dimensions
        im = Image.open(filename)
        # extract the largest possible central square
        dims = im.size
        start_1 = (dims[0] - min(dims)) // 2
        start_2 = (dims[1] - min(dims)) // 2
        im = im.crop((start_1, start_2, start_1+min(dims), start_2+min(dims)))
        # reshape the central square as a 64-by-64 square
        im.thumbnail((64,64), Image.ANTIALIAS)
        # turn the image into a numpy array and concatenate it to X_train
        image_array = np.array(im)
        image_array = image_array[np.newaxis,:,:,:]
        X_train = np.concatenate([X_train, image_array])
        namelist.append(filename)
    except:
        pass
    i = i+1
    if i % 100 == 0:
        print('Iteration:', i)

# save the image array and names array
namelist = np.array(namelist)
np.save('X_train.npy', X_train)
np.save('namelist.npy', namelist)