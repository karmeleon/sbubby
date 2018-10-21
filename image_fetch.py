import argparse
import collections
import functools
from io import BytesIO
import json
from multiprocessing import Pool
import os
import os.path
from urllib.parse import urlparse, urlunparse

from PIL import Image
import requests
from tqdm import tqdm

import common

SKIP_FILE_NAME = 'skip.txt'
URL_WHITELIST_FILE_NAME = 'url_whitelist.txt'

def main():
	parser = argparse.ArgumentParser(description='Hoover up some reddit posts in JSON format and download their images.')
	parser.add_argument('file', help='.json file to read from, in https://github.com/karmeleon/reddit-scraper format.')
	
	args = parser.parse_args()

	if not os.path.isfile(args.file):
		print('{} does not exist, exiting.'.format(args.file))
		exit(1)
	
	with open(args.file, 'r', encoding="utf-8") as f:
		json_data = json.load(f)
	
	posts = [(post['id'], post['url'], post['subreddit']) for post in json_data]
	
	with open(URL_WHITELIST_FILE_NAME, 'r') as file:
		url_whitelist = {line.rstrip('\n') for line in file}

	img_path = os.path.abspath('images')

	if not os.path.isdir(img_path):
		os.mkdir(img_path)

	# get just the filename (without path or extension)
	existing_images = common.get_filenames_in_directory_without_extension(img_path)

	bound_download_image = functools.partial(download_image, existing_images, url_whitelist, img_path)

	# default dict for counting posts downloaded from each sub, defaults to 0
	counts = collections.defaultdict(int)

	# fire up a thread pool to download + save the images in parallel
	with Pool(processes=os.cpu_count() * 2) as pool:
		results = pool.imap_unordered(bound_download_image, posts, 10)
		for result in tqdm(results, total=len(posts)):
			if result is None:
				continue
			counts[result] += 1
	
	print('Downloaded:')
	for sub, count in counts.items():
		print('{}: {}'.format(sub, count))


def download_image(skip_images, url_whitelist, img_path, post_data):
	post_id, url, subreddit = post_data
	# don't download anything we've already downloaded or know to be dead
	if post_id in skip_images:
		return None
	
	parsed_url = urlparse(url)

	# skip any blacklisted domains
	if parsed_url.netloc not in url_whitelist:
		return None
	
	image = common.fetch_image(url)

	if image is None:
		skip_post(post_id)

	# process the image down to being smol
	image = common.process_image(image)

	# dump it as a jpg
	image.save(os.path.join(img_path, '{}.jpg'.format(post_id)))

	return subreddit

def skip_post(post_id):
	# Keep a note to skip a post in future runs (only do this if the post has been deleted,
	# not just because the fetch failed -- it could mean we need a new function to handle the url)
	with open(SKIP_FILE_NAME, 'a') as file:
		file.write('{}\n'.format(post_id))


if __name__ == '__main__':
	main()
