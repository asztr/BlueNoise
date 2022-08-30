#!/usr/bin/env python
import numpy as np
import argparse
import os.path as op
import image_io
import glob
from tqdm import tqdm

def dither(img, noise, bits=1, margin=0): #function expects noise in [0,1] but transforms to [-1,1]
	noise_mat = (np.abs(noise) - 0.5) * 2.0
	if margin > 0:
		noise_mat = noise_mat[margin:-margin, margin:-margin]
	noise_wrp = np.tile(noise_mat, reps=[10, 10])[0:img.shape[0], 0:img.shape[1]]
	noise_wrp_3d = np.stack((noise_wrp, noise_wrp, noise_wrp), axis=-1)
	dth_scale = float(bits)
	img_dth = 1.0 / dth_scale * np.round(dth_scale * img + 0.5 * noise_wrp_3d)
	return img_dth

def gls_to_fnames(gls):
    eu = op.expanduser
    fnames = sum([glob.glob(eu(gl)) for gl in gls if gl[0] != "!"], [])
    fnames_exclude = sum([glob.glob(eu(gl[1:])) for gl in gls if gl[0] == "!"], [])
    return [fname for fname in fnames if fname not in fnames_exclude]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--images', nargs='+', default=['~/bluenoise/dgiunchi/data/photo1.png'], help='images to dither')
	parser.add_argument('--noises', nargs='+', required=True, help='noise images (grayscale dither masks)')
	parser.add_argument('--bits', nargs='+', type=int, default=[1], help='number of bits for dithering')
	parser.add_argument('--margin', type=int, default=0, help='remove n pixels from noise borders')
	parser.add_argument('--suffix', default='dither_', help=' ')
	parser.add_argument('--destdir', type=str, default='./', help='destination dir for output files')
	parser.add_argument('--verbose', choices=['none', 'fnames', 'pbar', 'all'], default='pbar', help=' ')
	args = parser.parse_args()

	img_fnames = gls_to_fnames(args.images)
	noise_fnames = gls_to_fnames(args.noises)

	for img_fname in tqdm(img_fnames, disable=(args.verbose in ['none', 'fnames'])):
		img = image_io.read_image(img_fname)

		if len(img.shape) == 3:
			img = img[:, :, :3]

		for noise_fname in noise_fnames:
			noise = image_io.read_image(noise_fname)

			if len(noise.shape) == 3:
				noise = noise[:, :, 0]

			for bits in args.bits:
				img_dth = dither(img, noise, bits=bits, margin=args.margin)

				noise_str = op.splitext(op.basename(noise_fname))[0].replace("pred_", "")
				img_str = op.splitext(op.basename(img_fname))[0]
				fname_out = op.join(args.destdir, args.suffix + img_str + '_' + noise_str + '_bits' + str(bits) + ".png")
				image_io.write_image(fname_out, img_dth)

				if args.verbose in ['fnames', 'all']:
					print(img_fname, '=>', fname_out)
