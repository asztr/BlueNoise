import numpy as np
import tensorflow as tf

def tf_fft(x):
	return tf.signal.fftshift(tf.signal.fft2d(tf.cast(x, tf.complex64)))

def tf_absfft(x):
	return tf.math.abs(tf_fft(x))

def tf_ifft(x):
	return tf.signal.ifft2d(tf.signal.ifftshift(tf.cast(x, tf.complex64)))

def tf_absifft(x):
	return tf.math.abs(tf_ifft(x))

def np_absfft(img_f):
	return np.abs(np.fft.fftshift(np.fft.fft2(img_f, axes=[0,1])))

def np_absifft(img_f):
	return np.abs(np.fft.ifft2(np.fft.ifftshift(img_f), axes=[0,1]))

def vis(x, norm=True):
	_x = x.copy()
	n = _x.shape[0]
	_x[n//2][n//2] = 0.0
	_x = np.abs(_x)
	if norm is True:
		_x = _x/_x.max()
	return _x

# def imshow(x):
# 	plt.imshow(vis(x), cmap="Greys_r")
# 	plt.colorbar()
# 	plt.show()
#
# def tfimshow(x):
# 	plot(x.numpy())
#
# def plot(x):
# 	n = x.shape[0]
# 	plt.plot(range(n), vis(x)[n//2, :])
#
# def tfplot(x):
# 	plot(x.numpy())
