import argparse
import json
import os
import os.path

import tensorflow as tf
from tensorflow import keras

import common

SPLIT_PERCENTAGE = 0.8
IMAGE_SIZE = 256
BATCH_SIZE = 96
NUM_EPOCHS = 10

def main():
	output_dir = os.path.abspath('output')
	records_dir = os.path.abspath('records')
	metadata_file = os.path.abspath('sbubby_metadata.json')

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if not os.path.isdir(records_dir):
		print("Couldn't find the records directory -- did you run preprocess yet?")
		exit(1)
	
	if not os.path.isfile(metadata_file):
		print("Couldn't find the metadata JSON -- did you run preprocess yet?")
		exit(1)
	
	with open(metadata_file, 'r') as metadata_file:
		metadata = json.load(metadata_file)
	
	total_samples = metadata['totalSamples']
	sub_count = len(metadata['mapping'])

	training_dataset, training_size = get_dataset(True, records_dir, total_samples, sub_count)

	model = keras.Sequential()

	# First conv layer
	model.add(keras.layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		padding='same',
		activation='relu',
		data_format='channels_last',
		use_bias=True,
		input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
	))

	# First pool layer
	model.add(keras.layers.MaxPooling2D(
		pool_size=(2, 2),
		strides=2,
		padding='same',
		data_format='channels_last',
	))

	# Second conv layer
	model.add(keras.layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		padding='same',
		activation='relu',
		data_format='channels_last',
		use_bias=True,
	))

	# Second pool layer
	model.add(keras.layers.MaxPooling2D(
		pool_size=(2, 2),
		strides=2,
		padding='same',
		data_format='channels_last',
	))

	# flat as a washboard
	model.add(keras.layers.Flatten())

	# fkin d e n s e
	# we're looking at a 64 * 64 vector, do that many layers
	model.add(keras.layers.Dense(1024, 'relu'))
	# d-d-d-dropout
	model.add(keras.layers.Dropout(0.5))
	# then cut it down
	model.add(keras.layers.Dense(256, 'relu'))
	# overfitting is bad okay
	model.add(keras.layers.Dropout(0.5))
	# output layer
	model.add(keras.layers.Dense(sub_count, 'softmax'))

	# compile it
	model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'],
	)

	model.summary()

	print('Training')
	model.fit(
		training_dataset,
		epochs=NUM_EPOCHS,
		steps_per_epoch=int(training_size / BATCH_SIZE),
		callbacks=[
			keras.callbacks.TensorBoard(log_dir='./output', write_images=True),
			keras.callbacks.ModelCheckpoint(filepath='./checkpoint'),
		],
	)

	print('Saving model')
	model.save('sbubby.h5')

	print('Evaluating')
	eval_dataset, eval_size = get_dataset(False, records_dir, total_samples, sub_count)
	print(model.evaluate(eval_dataset, steps=int(eval_size / BATCH_SIZE)))

def parse_example(example_proto, sub_count):
	features = {
		'image': tf.FixedLenFeature((), tf.string),
	}

	labels = {
		'subreddit': tf.FixedLenFeature((), tf.int64),
	}

	image = tf.parse_single_example(example_proto, features)['image']

	image = tf.decode_raw(image, tf.uint8)
	image = tf.reshape(image, (256, 256, 3))
	image = tf.image.convert_image_dtype(image, tf.float16)

	label = tf.parse_single_example(example_proto, labels)
	label = tf.one_hot(label['subreddit'], sub_count, dtype=tf.float16)

	return image, label

def get_dataset(is_training, records_dir, example_count, sub_count):
	# load the data from the .tfrecords files
	files = list(map(lambda s: os.path.join(records_dir, s), tf.gfile.ListDirectory(records_dir)))
	dataset = tf.data.TFRecordDataset(files)
	# split it into training and test sections
	if is_training:
		dataset = dataset.take(int(SPLIT_PERCENTAGE * example_count))
		dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, NUM_EPOCHS))
	else:
		dataset = dataset.skip(int(SPLIT_PERCENTAGE * example_count))

	# process it into tensors
	dataset = dataset.map(lambda e: parse_example(e, sub_count), num_parallel_calls=os.cpu_count() * 4)
	# eat them up in batches
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(1)

	return dataset, example_count * SPLIT_PERCENTAGE if is_training else example_count * (1.0 - SPLIT_PERCENTAGE)

if __name__ == '__main__':
	main()
