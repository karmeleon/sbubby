import argparse
import functools
import json
from multiprocessing import Pool
import os.path
import random

from PIL import Image
from tqdm import tqdm
import tensorflow as tf

import common

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def main():
	parser = argparse.ArgumentParser(description='Convert JSON and .jpeg data into a .tfrecords file for use in train.py.')
	parser.add_argument('file', help='.json file to read from, in https://github.com/karmeleon/reddit-scraper format.')

	args = parser.parse_args()

	verify_files(args)

	output_dir = os.path.abspath('records')

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	# load the raw json
	print('Loading JSON')
	with open(args.file, 'r', encoding='utf-8') as f:
		json_data = json.load(f)
	
	# mix it up to improve training data
	random.shuffle(json_data)

	# see which images we should be skipping
	print('Discovering image files')
	post_ids_with_images = common.get_filenames_in_directory_without_extension('images')

	with Pool(processes=os.cpu_count()) as pool:
		print('Enumerating subreddits')
		subs = list(set(pool.imap_unordered(extract_sub, json_data, 300)))

		sub_to_number = {sub: idx for (idx, sub) in enumerate(subs)}

		print('Processing data')
		map_partial = functools.partial(process_post, post_ids_with_images, sub_to_number, output_dir)
		samples = pool.imap_unordered(map_partial, json_data, 30)
		
		shard_num = 0
		samples_per_shard = 500
		sample_count = 0
		total_samples = 0

		writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, '{}.tfrecords'.format(shard_num)))

		for sample in tqdm(samples, total=len(json_data)):
			if sample is None:
				continue
			writer.write(sample)
			sample_count += 1
			total_samples += 1
			if sample_count == samples_per_shard:
				writer.close()
				sample_count = 0
				shard_num += 1
				writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, '{}.tfrecords'.format(shard_num)))
		
		writer.close()

	metadata = {
		'mapping': subs,
		'totalSamples': total_samples,
	}

	# dump some metadata for the trainer and predictor
	with open('sbubby_metadata.json', 'w') as outfile:
		json.dump(metadata, outfile)

	
	print('Output to {}'.format(output_dir))

def extract_sub(post):
	return post['subreddit']

def process_post(post_ids_with_images, sub_to_number, output_dir, post):
	if post['id'] not in post_ids_with_images:
		return
	try:
		im = Image.open(os.path.abspath(os.path.join('images', '{}.jpg').format(post["id"])))
	except OSError:
		print('Failed to process {}, dropping'.format(post["id"]))
		return None

	feature = {
		'subreddit': _int64_feature(sub_to_number[post['subreddit']]),
		'image': _bytes_feature(im.tobytes()),
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))

	im.close()

	return example.SerializeToString()


def verify_files(args):
	if not os.path.isfile(args.file):
		print("Couldn't find JSON file. Exiting.")
		exit(1)

	if not os.path.isdir('images'):
		print("Couldn't find images/ directory. Exiting.")
		exit(1)

if __name__ == '__main__':
	main()
