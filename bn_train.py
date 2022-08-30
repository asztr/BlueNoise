#!/usr/bin/env python3
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import sys
from fft import *

step = 0
SIZE = 128
batch_size = 64
lr = 0.0005
samples = 100*64
epochs = 12

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
	tf.config.experimental.set_memory_growth(device, True)

# Gaussian Mask (alternative to circular mask for blue spectrum loss)
def gaussianMask(size, fwhm = 3, center=None):
	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]
	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]
	return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

# Uniform Spectrum Loss: square of 2d laplacian and directional derivatives
def uniform_spectrum_loss(m):
	s = tf_absfft(m)
	laplacian = (-4.0*s + tf.roll(s, 1, axis=2) + tf.roll(s, -1, axis=2) +
				 tf.roll(s, 1, axis=1) + tf.roll(s, -1, axis=1))/5.0
	g_x = (s - tf.roll(s, 1, axis=1)) / 2.0
	g_y = (s - tf.roll(s, 1, axis=2)) / 2.0
	return  tf.math.square(laplacian) + tf.math.square(g_x) + tf.math.square(g_y)

# Histogram Loss: squared difference of sorted values as compared to the reference distribution
def histogram_loss(m, SIZE, batch_size):
	ref_range0 = tf.cast(tf.linspace(-1, 1, SIZE*SIZE), tf.float32)
	ref_range = tf.tile([ref_range0], [batch_size, 1])
	return tf.math.square(tf.sort(tf.reshape(m, [batch_size, -1]), axis=-1) - ref_range)

# Blue Spectrum Loss: squared deviation from low frequency in the center
def blue_spectrum_loss(m, SIZE):
	s = tf_absfft(m)
	CUTOFF_FREQ = 0.7
	aux1 = tf.tensordot(tf.math.square(tf.cast(tf.linspace(1, -1, SIZE), tf.float32)), tf.ones(SIZE), axes=0)
	aux2 = tf.tensordot(tf.ones(SIZE), tf.math.square(tf.cast(tf.linspace(1,-1,SIZE), tf.float32)), axes=0)
	low_freq_weights = tf.maximum(CUTOFF_FREQ - aux1 - aux2, 0.0) / CUTOFF_FREQ
	return (low_freq_weights**2)*tf.cast(tf.math.square(s), tf.float32)

def clip(x):
	return tf.clip_by_value(x, 0.0, 1.0)

def loss(y_true, y_pred):
	#global step,SIZE, batch_size

	us_loss = uniform_spectrum_loss(y_pred)
	h_loss = histogram_loss(y_pred, SIZE, batch_size)
	bs_loss = blue_spectrum_loss(y_pred, SIZE)
	gmask = gaussianMask(SIZE, int(0.3*SIZE/2.0), (SIZE//2, SIZE//2))[np.newaxis, :, :]
	#tf.math.square(tf_absfft(y_pred)*gmask)

	us_loss_s = tf.math.reduce_mean(us_loss)
	h_loss_s = tf.math.reduce_mean(h_loss)
	bs_loss_s = tf.math.reduce_mean(bs_loss)
	final_loss = 0.01*us_loss_s + 2.2*h_loss_s + 0.1*bs_loss_s

	#step += 1
	#if step % 100 == 0:
		#print("\n")
		#print('us:', us_loss_s.numpy(), "h:", h_loss_s.numpy(), "bs:", bs_loss_s.numpy())

	return final_loss

def model_fc_realio(SIZE):
	inp = tf.keras.layers.Input(shape=(SIZE, SIZE))
	x = tf.keras.layers.Flatten()(inp)
	x = tf.keras.layers.Dense(100, activation='tanh')(x)
	x = tf.keras.layers.Dense(100, activation='tanh')(x)
	x = tf.keras.layers.Dense(SIZE*SIZE, activation='tanh')(x)
	x = tf.keras.layers.Reshape((SIZE, SIZE))(x)
	return tf.keras.Model(inputs=inp, outputs=x)

def main():
	global SIZE, batch_size
	train_images = np.random.uniform(-1.0, 1.0, size=(samples, SIZE, SIZE))

	# Training
	model = model_fc_realio(SIZE)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, run_eagerly=False)
	model.fit(train_images, train_images, epochs=epochs, use_multiprocessing=False, batch_size=batch_size, shuffle=True)

	model_json = "model" + str(SIZE) + ".json"
	model_h5 = "model" + str(SIZE) + ".h5"
	with open(model_json, "w") as json_file:
		json_file.write(model.to_json())
	model.save_weights(model_h5)

if __name__ == "__main__":
	main()
