import argparse
import json
import os
import os.path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from common import process_image

IMAGE_SIZE = 256

def main():
	parser = argparse.ArgumentParser(description='Pass in an image, get a sub.')
	parser.add_argument('image_path', help='A path to an image to test')
	parser.add_argument('-l', action='store_true', help='Use TensorFlow Lite to predict, convering the h5 as necessary')

	args = parser.parse_args()

	# the GPU is overkill
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	metadata_file = os.path.abspath('sbubby_metadata.json')

	# TODO: dedupe this
	if not os.path.isfile(metadata_file):
		print("Couldn't find the metadata JSON -- did you run preprocess yet?")
		exit(1)
	
	with open(metadata_file, 'r') as metadata_file:
		metadata = json.load(metadata_file)

	im = Image.open(args.image_path)
	im = process_image(im, IMAGE_SIZE)

	im_array = np.expand_dims(np.asarray(im), axis=0)
	# convert it to a float array
	im_array = im_array.astype(np.float16)
	# then divide by 255 to have pixel values between 0 and 1
	im_array /= 255

	if args.l:
		predict_with_tflite(im_array, metadata)
	else:
		predict_with_keras(im_array, metadata)

def predict_with_tflite(im_array, metadata):
	if not os.path.isfile('sbubby.tflite'):
		convert_model_to_tflite()
	
	print('Predicting with tflite')
	
	interpreter = tf.contrib.lite.Interpreter(model_path='sbubby.tflite')
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	interpreter.set_tensor(input_details[0]['index'], im_array)
	interpreter.invoke()

	output_tensor = interpreter.get_tensor(output_details[0]['index'])
	print_output_tensor(output_tensor, metadata)

def predict_with_keras(im_array, metadata):
	print('Predicting with keras')
	model = keras.models.load_model('sbubby.h5')

	output_tensor = model.predict(im_array)
	print_output_tensor(output_tensor, metadata)

def print_output_tensor(output_tensor, metadata):
	for sub, weight in zip(metadata['mapping'], output_tensor[0]):
		print('{}: {:.3f}'.format(sub.ljust(15), weight))
	
def convert_model_to_tflite():
	print('Converting h5 model to tflite...')
	converter = tf.contrib.lite.TocoConverter.from_keras_model_file('sbubby.h5')
	tflite_model = converter.convert()
	with open('sbubby.tflite', 'wb') as f:
		f.write(tflite_model)

if __name__ == '__main__':
	main()
