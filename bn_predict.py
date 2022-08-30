#!/usr/bin/env python3
from tensorflow.keras.models import load_model, model_from_json
from PIL import Image
import argparse
import os
import os.path as op
import glob
from fft import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('h5s', nargs='+', help='Input models (h5)')
parser.add_argument('--fft', action='store_true', default=False, help=' ')
parser.add_argument('--bins', action='store_true', default=False, help=' ')
parser.add_argument('--bin_mids', type=float, nargs='+', default=[0.0, 0.025, 0.05, 0.075, 0.1], help='Thresholds for binarisation')
parser.add_argument('--pred_prefix', default='pred', help='filename prefix for predicted noise')
parser.add_argument('--predfft_prefix', default='predfft', help='filename prefix for fft of predicted noise')
parser.add_argument('--input_res', type=int, default=128, help='size/resolution of input noise image')
parser.add_argument('--cpu', action='store_true', default=False, help='use CPU instead of GPU')
args = parser.parse_args()

if args.cpu is True:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sample = np.random.uniform(-1.0, 1.0, size = (1, args.input_res, args.input_res))

def gls_to_fnames(gls):
    eu = op.expanduser
    fnames = sum([glob.glob(eu(gl)) for gl in gls if gl[0] != "!"], [])
    fnames_exclude = sum([glob.glob(eu(gl[1:])) for gl in gls if gl[0] == "!"], [])
    return [fname for fname in fnames if fname not in fnames_exclude]

def binarise(x, mid):
	return np.where(x<=mid, 0.0, 1.0)

def save_image(fname, img_f, verbose=True):
	Image.fromarray(img_f*256).convert('L').save(fname)
	if verbose is True:
		print("\tWrote", fname)

def predict_h5(model_h5):
	model_json = model_h5.replace(".h5", ".json")
	with open(model_json, 'r') as f:
		json_str = f.read()
	loaded_model = model_from_json(json_str)
	loaded_model.load_weights(model_h5)
	print("Loaded", model_h5)

	loaded_model.summary()

	img = loaded_model.predict(sample)[0]
	img = (img + 1.0) / 2.0
	img_png = model_h5.replace("model", args.pred_prefix).replace(".h5", ".png")
	save_image(img_png, img)

	if args.fft is True:
		imgfft = np_absfft(img)
		n = imgfft.shape[0]
		imgfft[n//2][n//2] = 0.0
		imgfft = imgfft/imgfft.max()
		imgfft_png = model_h5.replace("model", args.predfft_prefix).replace(".h5", ".png")
		save_image(imgfft_png, imgfft)

	if args.bins is True:
		for bin_mid in args.bin_mids:
			imgf = binarise(img, bin_mid)
			img_png = model_h5.replace("model", args.pred_prefix).replace(".h5", "_bin"+str(bin_mid)+".png")
			save_image(img_png, imgf)

			if args.fft is True:
				imgfft = np_absfft(imgf)
				imgfft_png = model_h5.replace("model", args.predfft_prefix).replace(".h5", "_bin"+str(bin_mid)+".png")
				save_image(imgfft_png, imgfft)

if __name__ == "__main__":
	h5s = gls_to_fnames(args.h5s)
	print("h5s:", h5s)
	print("#h5s:", len(h5s))
	print("input shape:", sample.shape)

	for h5 in h5s:
		predict_h5(h5)
