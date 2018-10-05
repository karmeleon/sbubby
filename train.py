import argparse
import json
from multiprocessing import Pool
import os
import os.path

import tensorflow as tf

import common
from cnn import cnn_model_fn

SPLIT_PERCENTAGE = 0.8

def main():
	output_dir = os.path.abspath('output')
	records_dir = os.path.abspath('records')

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if not os.path.isdir(records_dir):
		print("Couldn't find the records directory -- did you run preprocess yet?")
		exit(1)

	"""
	estimator = tf.estimator.DNNRegressor(
		feature_columns=[
			tf.feature_column.numeric_column('image', shape=[384, 384, 3], dtype=tf.float16),
		],
		hidden_units=[1024, 512, 256],
		model_dir=output_dir,
	)
	"""

	estimator = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir=output_dir,
	)

	"""
	tensors_to_log = [
		'mean_absolute_error',
		#'global_step',
		'rmse',
		#'loss',
	]
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log,
		every_n_secs=20,
	)
	"""

	print('Training')
	estimator.train(
		input_fn=lambda: input_fn(False, records_dir),
		#hooks=[logging_hook],
	)

	print('Evaluating')
	results = estimator.evaluate(input_fn=lambda: input_fn(True, records_dir))
	for key in sorted(results):
		print(f'{key}: {results[key]}')

def parse_example(example_proto):
	features = {
		'image': tf.FixedLenFeature((), tf.string),
	}

	labels = {
		'score': tf.FixedLenFeature((), tf.int64, default_value=0),
	}

	image = tf.parse_single_example(example_proto, features)['image']
	labels = tf.parse_single_example(example_proto, labels)

	image = tf.decode_raw(image, tf.uint8)
	image = tf.image.convert_image_dtype(image, tf.float16)

	return {'image': image}, labels['score']

def input_fn(is_training, records_dir):
	# load the data from the .tfrecords files
	files = list(map(lambda s: os.path.join(records_dir, s), tf.gfile.ListDirectory(records_dir)))
	dataset = tf.data.TFRecordDataset(files)
	# split it into training and test sections
	if is_training:
		dataset = dataset.take(int(SPLIT_PERCENTAGE * len(files)))
	else:
		dataset = dataset.skip(int(SPLIT_PERCENTAGE * len(files)))
		dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, 150))

	# process it into tensors
	dataset = dataset.map(parse_example, num_parallel_calls=os.cpu_count() * 4)
	# eat them up in batches
	dataset = dataset.batch(32)
	dataset = dataset.prefetch(1)
	# get an iterator
	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()

	return features, labels

if __name__ == '__main__':
	main()
