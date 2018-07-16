import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Reshape
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam
from keras import backend, initializers
# set keras/tensorflow configuration as channel-first
backend.set_image_dim_ordering('th')

# set parameters and load data
BATCH_SIZE = 64
N_EPOCHS = 300
RANDOM_DIM = 100
adam = Adam(lr=0.0002, beta_1=0.5)
# load and reshape images
X_train = np.load('X_13styles.npy')
X_train = X_train.transpose((0,3,1,2))
X_train = X_train.astype(np.float32)/127.5 - 1.0
# load latent encoder outputs and delete useless columns (constants)
z = np.load('z_13styles.npy')
z_train = (z[:,0] - z[:,0].mean()) / np.sqrt(z[:,0].var())
z_train = z_train.reshape((9139,1))
for i in range(1,100):
    if z[:,i].var() > 0.001:
        z_i = (z[:,i] - z[:,i].mean()) / np.sqrt(z[:,i].var())
        z_i = z_i.reshape((9139,1))
        z_train = np.concatenate([z_train, z_i], axis=1)
num_batches = int(X_train.shape[0]/BATCH_SIZE)
LATENT_DIM = z_train.shape[1]
# compute mean and covariance of all latent vectors
mu = np.zeros((LATENT_DIM,))
Sigma = np.cov(z_train.transpose())
exp_replay = []
dLosses = []
gLosses = []

# conditional generator
g_in_ran = Input(shape=(RANDOM_DIM,))
g_in_lat = Input(shape=(LATENT_DIM,))
g_in = Concatenate()([g_in_ran, g_in_lat])
g = Dense(256*8*8,
          kernel_initializer=initializers.RandomNormal(stddev=0.02))(g_in)
g = LeakyReLU(0.2)(g)
g = Reshape((256, 8, 8))(g)
g = UpSampling2D(size=(2, 2))(g)
g = Conv2D(128, kernel_size=(5, 5), padding='same')(g)
g = LeakyReLU(0.2)(g)
g = UpSampling2D(size=(2, 2))(g)
g = Conv2D(64, kernel_size=(5, 5), padding='same')(g)
g = LeakyReLU(0.2)(g)
g = UpSampling2D(size=(2, 2))(g)
g_out = Conv2D(3, kernel_size=(5, 5), padding='same', activation='tanh')(g)
generator = Model([g_in_ran, g_in_lat], g_out)
generator.compile(loss='binary_crossentropy', optimizer=adam)

# conditional discriminator
d_in_img = Input(shape=(3,64,64))
d_in_lat = Input(shape=(LATENT_DIM,))
d = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
           kernel_initializer=initializers.RandomNormal(stddev=0.02))(d_in_img)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Flatten()(d)
d = Dense(256, activation='relu')(d)
d = Concatenate()([d, d_in_lat])
d = Dense(256, activation='relu')(d)
d_out = Dense(1, activation='sigmoid')(d)
discriminator = Model([d_in_img, d_in_lat], d_out)
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# conditional GAN
discriminator.trainable = False
disc_condition_input = Input(shape=(LATENT_DIM,))
gen_condition_input = Input(shape=(LATENT_DIM,))
gan_input = Input(shape=(RANDOM_DIM,))
x = generator([gan_input, gen_condition_input])
gan_out = discriminator([x, disc_condition_input])
gan = Model([gan_input, gen_condition_input, disc_condition_input], gan_out)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# functions for sampling random noise and random latent vectors
def generate_noise(n_samples, noise_dim):
    X = np.random.normal(0, 1, size=(n_samples, noise_dim))
    return X

def generate_random_latent(n_samples):
    random_latent = np.random.multivariate_normal(mu, Sigma, n_samples)
    random_latent[:,40] = -0.01046103
    return random_latent

# training
for epoch in range(N_EPOCHS):

    cum_d_loss = 0.
    cum_g_loss = 0.
  
    for batch_idx in range(num_batches):
        # get the next set of real images to be used in this iteration
        images = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        latent = z_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        
        # generate noise, random latent vectors and then images
        noise_data = generate_noise(BATCH_SIZE, RANDOM_DIM)
        rand_latent = generate_random_latent(BATCH_SIZE)
        generated_images = generator.predict([noise_data, rand_latent])
        
        # train on soft targets (add noise) and randomly flip 5% of targets
        noise_prop = 0.05
        
        # prepare labels for real data
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0,
                              high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)),
                                       size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        
        # train discriminator on real data
        discriminator.trainable = True
        d_loss_true = discriminator.train_on_batch([images, latent],
                                                   true_labels)
        
        # prepare labels for generated data
        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0,
                             high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)),
                                       size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
        
        # train discriminator on generated data
        d_loss_gene = discriminator.train_on_batch([generated_images,
                                                    rand_latent],
                                                   gene_labels)
        
        # store a random point for experience replay
        r_idx = np.random.randint(BATCH_SIZE)
        exp_replay.append([generated_images[r_idx], rand_latent[r_idx],
                           gene_labels[r_idx]])
        
        # if we have enough points, do experience replay
        if len(exp_replay) == BATCH_SIZE:
            generated_images = np.array([p[0] for p in exp_replay])
            latents = np.array([p[1] for p in exp_replay])
            gene_labels = np.array([p[2] for p in exp_replay])
            exp_loss = discriminator.train_on_batch([generated_images,latents],
                                                    gene_labels)
            exp_replay = []
            break
        
        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
        cum_d_loss += d_loss
        
        # train generator
        noise_data = generate_noise(BATCH_SIZE, RANDOM_DIM)
        random_latent = generate_random_latent(BATCH_SIZE)
        discriminator.trainable = False
        g_loss = gan.train_on_batch([noise_data, random_latent, random_latent],
                                    np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss
    
    # store losses and periodically save models
    dLosses.append(cum_d_loss / num_batches)
    gLosses.append(cum_g_loss / num_batches)
    epoch_plus = epoch+1
    if epoch_plus % 40 == 0:
        generator.save('G_%d.h5' % epoch_plus)
    
# save losses
np.savetxt('gLoss.txt', np.array(gLosses), fmt='%f')
np.savetxt('dLoss.txt', np.array(dLosses), fmt='%f')