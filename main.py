import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import pickle
from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
import os
import cv2


DATA_PATH = "DataSet/emojisJPG"

def load_ds(path):
	img = []
	for filename in os.listdir(path):
		image = cv2.imread(os.path.join(path,filename))
		if image is not None:
			img.append(image)
	return img

def plot(img,n):
	for i in range(n*n):
		plt.subplot(n,n,1+i)
		plt.axis('off')
		plt.imshow(img[i])
	plt.show()

# define the standalone discriminator model
# def discriminator(in_shape=(72,72,3)):
# 	model = keras.Sequential()
# 	# normal
# 	model.add(layers.Conv2D(128, (5,5), padding='same', input_shape=in_shape))
# 	model.add(layers.LeakyReLU(alpha=0.2))
# 	# downsample to 36*36
# 	model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
# 	model.add(layers.LeakyReLU(alpha=0.2))
# 	# downsample to 18*18
# 	model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
# 	model.add(layers.LeakyReLU(alpha=0.2))
# 	# downsample to 9*9
# 	model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
# 	model.add(layers.LeakyReLU(alpha=0.2))
# 	# # downsample to 5x5
# 	# model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
# 	# model.add(layers.LeakyReLU(alpha=0.2))
# 	# classifier
# 	model.add(layers.Flatten())
# 	model.add(layers.Dropout(0.5))
# 	model.add(layers.Dense(1, activation='sigmoid'))
# 	# compile model
# 	opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
# 	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# 	return model

def discriminator():
	model = keras.Sequential()
	model.add(layers.Input((72,72,3)))
	model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding = 'same'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding = 'same'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation="sigmoid"))
	model.summary()
	opt = keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(loss='binary_crossentropy',optimizer = opt, metrics =['accuracy'])

	return model


def generator():
	model = keras.Sequential()
	model.add(layers.Dense(256*9*9, input_dim = 100))
	model.add(layers.LeakyReLU(alpha = 0.2))
	model.add(layers.Reshape((9,9,256)))
	# upsample to 18*18
	model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	# upsample to 36*36
	model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	# upsample to 72*72
	model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	# # upsample to 80x80
	# model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	# model.add(layers.LeakyReLU(alpha=0.2))
	# output layer 80x80x3
	model.add(layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
	model.summary()
	return model

def gan(g_model, d_model):
	d_model.trainalbe = False
	model = keras.Sequential()
	model.add(g_model)
	model.add(d_model)

	opt = keras.optimizers.Adam(lr=0.0002, beta_1 = 0.5)
	model.compile(loss = 'binary_crossentropy', optimizer = opt)
	return model

def convert(data):
	x_train, x_test = train_test_split(data, test_size=0.2)
	x_train = (x_train - 127.5)/127.5

	return x_train

def real_emo(data,n_samples):
	idx = np.random.randint(0,data.shape[0], n_samples)
	emo = data[idx]
	label = np.ones((n_samples, 1))
	return emo, label


def latent_point(latent_dim, n_samples):
	x_input = np.random.randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)

	return x_input

def fake_emo(g_model, latent_dim, n_samples):
	x_input = latent_point(latent_dim, n_samples)

	X = g_model.predict(x_input)

	y = np.zeros((n_samples,1))
	return X,y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=64):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = real_emo(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = fake_emo(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = latent_point(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = np.ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

# create and save a plot of generated images
def save_plot(examples, epoch, n=10):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = real_emo(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = fake_emo(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)

ls = load_ds(DATA_PATH)
dataset = np.array(ls)
dataset = dataset

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = discriminator()
# create the generator
g_model = generator()
# create the gan
gan_model = gan(g_model, d_model)
# load image data
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)



