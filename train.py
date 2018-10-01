import argparse
import json
from multiprocessing import Pool
import os
import os.path

import tensorflow as tf

import common

SPLIT_PERCENTAGE = 0.8

def main():
	output_dir = os.path.abspath('output')
	records_dir = os.path.abspath('records')

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if not os.path.isdir(records_dir):
		print("Couldn't find the records directory -- did you run preprocess yet?")
		exit(1)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	estimator = tf.estimator.DNNRegressor(
		feature_columns=[
			tf.feature_column.numeric_column('image', shape=[384, 384, 3], dtype=tf.float16),
		],
		hidden_units=[1024, 512, 256],
		model_dir=output_dir,
		config=tf.estimator.RunConfig(session_config=config),
	)

	print('Training')
	estimator.train(input_fn=lambda: input_fn(False, records_dir))

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
		# TODO: increase shuffle buffer
		dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, 10))

	# process it into tensors
	dataset = dataset.map(parse_example, num_parallel_calls=os.cpu_count() * 4)
	#dataset = dataset.cache()
	# eat them up in batches
	dataset = dataset.batch(16)
	dataset = dataset.prefetch(1)
	# get an iterator
	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()

	return features, labels

def load_image(id_file, score):
	_, filename = id_file
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)
	image_converted = tf.image.convert_image_dtype(image_decoded, tf.float16)
	image_resized = tf.image.resize_image_with_crop_or_pad(image_converted, 384, 384)

	tf.debugging.check_numerics(image_resized, "shit's broke yo")

	return {'image': image_resized}, score

def verify_files(args):
	if not os.path.isfile(args.file):
		print("Couldn't find JSON file. Exiting.")
		exit(1)

	if not os.path.isdir('images'):
		print("Couldn't find images/ directory. Exiting.")
		exit(1)

def extract_data_from_json_record(post):
	return (
		post['id'],
		os.path.abspath(os.path.join('images', f'{post["id"]}.jpg')),
		post['score'],
	)

if __name__ == '__main__':
	main()
