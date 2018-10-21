import argparse
import json
import os
import os.path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

import common

def main():
	parser = argparse.ArgumentParser(description='Pass in an image, get a sub.')
	parser.add_argument('image_path', help='A path to an image to test')

	args = parser.parse_args()

	# the GPU is overkill
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	im = Image.open(args.image_path)

	predictor = Predictor()
	predictor.print_result_tensor(predictor.predict(im))

class Predictor(object):
	def __init__(self):
		metadata_file = os.path.abspath('sbubby_metadata.json')

		# TODO: dedupe this
		if not os.path.isfile(metadata_file):
			print("Couldn't find the metadata JSON -- did you run preprocess yet?")
			exit(1)
		
		with open(metadata_file, 'r') as metadata_file:
			self.metadata = json.load(metadata_file)
		
		self.model = keras.models.load_model('sbubby.h5')

		print('Loaded Keras predictor.')
	
	def predict(self, image):
		"""
		:param image: a PIL image. This method will transform it into a model-friendly format.
		:returns: a zip of (sub_name, probability)
		"""
		im = common.process_image(image)

		im_array = np.expand_dims(np.asarray(im), axis=0)
		# convert it to a float array
		im_array = im_array.astype(np.float16)
		# then divide by 255 to have pixel values between 0 and 1
		im_array /= 255
		
		# actually predict
		output_tensor = self.model.predict(im_array)

		zipped = zip(self.metadata['mapping'], output_tensor[0])
		return sorted(zipped, key=lambda entry: entry[1], reverse=True)
	
	def print_result_tensor(self, labeled_tensor):
		for sub, weight in labeled_tensor:
			print('{}: {:.3f}'.format(sub.ljust(15), weight))

if __name__ == '__main__':
	main()
