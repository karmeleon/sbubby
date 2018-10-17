import argparse
import json
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

	metadata_file = os.path.abspath('sbubby_metadata.json')

	# TODO: dedupe this
	if not os.path.isfile(metadata_file):
		print("Couldn't find the metadata JSON -- did you run preprocess yet?")
		exit(1)
	
	with open(metadata_file, 'r') as metadata_file:
		metadata = json.load(metadata_file)

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
	# convert it to a float array
	im_array = im_array.astype(np.float16)
	# then divide by 255 to have pixel values between 0 and 1
	im_array /= 255

	model = keras.models.load_model('sbubby.h5')

	output_vector = model.predict(im_array)

	for sub, weight in zip(metadata['mapping'], output_vector[0]):
		print(f'{sub.ljust(15)}: {weight:.3f}')

if __name__ == '__main__':
	main()