import csv
import numpy as np
from matplotlib import pyplot as plt

# load the full image array and names array
X = np.load('X_train.npy')
namelist = np.load('namelist.npy')
# load the information dataset and extract information of each image
# note that the information dataset contains more rows than X
with open('all_data_info.csv', newline='', encoding="utf8") as csvfile:
    art_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    art_info = list(art_reader)
info_type = []
info_name = []
info_style = []
info_artist = []
for i in range(1,103251):
    info_type.append(art_info[i][2])
    info_name.append(art_info[i][11])
    info_style.append(art_info[i][7])
    info_artist.append(art_info[i][0])
info_type = np.array(info_type)
info_name = np.array(info_name)
info_style = np.array(info_style)
info_artist = np.array(info_artist)

# logical array that exclude weird types of paintings from the dataset
a1 = np.array(info_type!='sketch and study')
a2 = np.array(info_type!='illustration')
a3 = np.array(info_type!='design')
a4 = np.array(info_type!='interior')

# extract non-weird images matching 13 select styles (in chronological order)
b = np.array(info_style=='High Renaissance')
my_type = info_name[b & a1 & a2 & a3 & a4]
X1 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Mannerism (Late Renaissance)')
my_type = info_name[b & a1 & a2 & a3 & a4]
X2 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Baroque')
my_type = info_name[b & a1 & a2 & a3 & a4]
X3 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Rococo')
my_type = info_name[b & a1 & a2 & a3 & a4]
X4 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Neoclassicism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X5 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Romanticism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X6 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Realism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X7 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Impressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X8 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Post-Impressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X9 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Expressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X10 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Surrealism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X11 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Abstract Expressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X12 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Pop Art')
my_type = info_name[b & a1 & a2 & a3 & a4]
X13 = X[np.in1d(namelist, my_type)]

# extract the same number of images from each style subset
X1 = X1[np.random.choice(X1.shape[0], 703, replace=False)]
X2 = X2[np.random.choice(X2.shape[0], 703, replace=False)]
X3 = X3[np.random.choice(X3.shape[0], 703, replace=False)]
X4 = X4[np.random.choice(X4.shape[0], 703, replace=False)]
X5 = X5[np.random.choice(X5.shape[0], 703, replace=False)]
X6 = X6[np.random.choice(X6.shape[0], 703, replace=False)]
X7 = X7[np.random.choice(X7.shape[0], 703, replace=False)]
X8 = X8[np.random.choice(X8.shape[0], 703, replace=False)]
X9 = X9[np.random.choice(X9.shape[0], 703, replace=False)]
X10 = X10[np.random.choice(X10.shape[0], 703, replace=False)]
X11 = X11[np.random.choice(X11.shape[0], 703, replace=False)]
X12 = X12[np.random.choice(X12.shape[0], 703, replace=False)]
X13 = X13[np.random.choice(X13.shape[0], 703, replace=False)]

# plot a selection of 49 images from a particular style
figsize = (7,7)
dim = (7,7)
examples = 49
X_sample = X1[np.random.choice(X1.shape[0], examples, replace=False)]
plt.figure(figsize=figsize)
for i in range(X_sample.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(X_sample[i,:,:,:])
    plt.axis('off')
plt.show()

# concatenate the 13 style subsets and create a label array
X_all = np.concatenate([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13])
y = [0,1,2,3,4,5,6,7,8,9,10,11,12]
y_all = np.repeat(y,703)
# shuffle images and labels and save them as numpy files
idx_shuffle = np.random.choice(9139, 9139, replace=False)
X_all_shuff = X_all[idx_shuffle]
y_all_shuff = y_all[idx_shuffle]
np.save('X_13styles.npy', X_all_shuff)
np.save('y_13styles.npy', y_all_shuff)