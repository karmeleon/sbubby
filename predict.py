import argparse
import os
import os.path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = 256

def main():
	parser = argparse.ArgumentParser(description='Pass in an image, get a sub.')
	parser.add_argument('image_path', help='A path to an image to test')

	args = parser.parse_args()

	im = Image.open(args.image_path)
	# TODO: dedupe this
	im = im.convert('RGB')
	# Resize the image down, then paste it on a square canvas to pad it to 1:1
	im.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
	canvas = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
	upper_left = (
		int((IMAGE_SIZE - im.size[0]) / 2),
		int((IMAGE_SIZE - im.size[1]) / 2),
	)
	canvas.paste(im, upper_left)

	im_array = np.expand_dims(np.asarray(canvas), axis=0)

	model = keras.models.load_model('sbubby.h5')

	tmp = model.outputs
	model.outputs = [model.layers[-1].output]

	print(model.predict(im_array))

if __name__ == '__main__':
	main()