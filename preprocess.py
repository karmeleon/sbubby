import argparse
import functools
import json
from multiprocessing import Pool
import os.path

from PIL import Image, ImageOps
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
	
	# see which images we should be skipping
	print('Discovering image files')
	post_ids_with_images = common.get_filenames_in_directory_without_extension('images')

	print('Processing data')
	with Pool(processes=os.cpu_count()) as pool:
		map_partial = functools.partial(process_post, post_ids_with_images, output_dir)
		samples = pool.imap_unordered(map_partial, json_data, 30)
		
		shard_num = 0
		samples_per_shard = 500
		sample_count = 0

		writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, f'{shard_num}.tfrecords'))

		for sample in tqdm(samples, total=len(json_data)):
			if sample is None:
				continue
			writer.write(sample)
			sample_count += 1
			if sample_count == samples_per_shard:
				writer.close()
				sample_count = 0
				shard_num += 1
				writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, f'{shard_num}.tfrecords'))
		
		writer.close()

	
	print(f'Output to {output_dir}')
	
def process_post(post_ids_with_images, output_dir, post):
	if post['id'] not in post_ids_with_images:
		return
	try:
		im = Image.open(os.path.abspath(os.path.join('images', f'{post["id"]}.jpg')))
		im = im.convert('RGB')
		im = ImageOps.fit(im, (384, 384))
	except OSError:
		print(f'Failed to process {post["id"]}, dropping')
		return None

	feature = {
		'score': _int64_feature(post['score']),
		'image': _bytes_feature(im.tobytes()),
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))

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