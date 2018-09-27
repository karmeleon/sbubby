import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
from io import BytesIO
import json
import os
import os.path
from urllib.parse import urlparse, urlunsplit

from PIL import Image
import requests

SKIP_FILE_NAME = 'skip.txt'
ERROR_FILE_NAME = 'error_ids.txt'
ERROR_URL_FILE_NAME = 'error_urls.txt'

def main():
	parser = argparse.ArgumentParser(description='Hoover up some reddit posts in JSON format and download their images.')
	parser.add_argument('file', help='.json file to read from, in https://github.com/karmeleon/reddit-scraper format.')
	
	args = parser.parse_args()

	if not os.path.isfile(args.file):
		print(f'{args.file} does not exist, exiting.')
		exit(1)
	
	with open(args.file, 'r', encoding="utf-8") as f:
		json_data = json.load(f)
	
	posts = [(post['id'], post['url']) for post in json_data]

	dead_ids = set()
	# see which IDs we should skip
	for filename in [SKIP_FILE_NAME, ERROR_FILE_NAME]:
		if os.path.isfile(filename):
			with open(filename, 'r') as file:
				dead_ids = dead_ids | {line.rstrip('\n') for line in file}

	img_path = os.path.abspath('images')

	if not os.path.isdir(img_path):
		os.mkdir(img_path)

	# get just the filename (without path or extension)
	existing_images = {
		os.path.splitext(os.path.split(f)[1])[0]
		for f in os.listdir(img_path)
		if os.path.isfile(os.path.join(img_path, f))
	}

	bound_download_image = functools.partial(download_image, existing_images | dead_ids, img_path)

	# fire up a thread pool to download + save the images in parallel
	with ThreadPoolExecutor(max_workers=4) as pool:
		pool.map(bound_download_image, posts)


def download_image(skip_images, img_path, post_data):
	post_id, url = post_data
	# don't download anything we've already downloaded or know to be dead
	if post_id in skip_images:
		return
	
	parsed_url = urlparse(url)

	# A map of netlocs to functions that can transform the link into a raw image URL
	PARSERS = {
		'imgur.com': imgur_imageify,
	}

	if parsed_url.netloc in PARSERS:
		# If we have a specialized handler for a URL, use it to transform the URL
		direct_image_url = PARSERS[parsed_url.netloc](parsed_url)
	else:
		# Otherwise, cross our fingers and hope the URL is a direct image link
		direct_image_url = url

	try:
		r = requests.get(direct_image_url, timeout=10)
		# dump it as a png
		image = Image.open(BytesIO(r.content))
		# check to see if it's actually worth saving
		if image.size == (130, 60):
			print(f'{direct_image_url} is (probably) a deleted reddit image post :(')
			skip_post(post_id)
			return
		if image.size == (161, 81):
			print(f'{direct_image_url} is (probably) a deleted imgur post :(')
			skip_post(post_id)
			return

		image.save(os.path.join(img_path, f'{post_id}.jpg'))
		print(f'downloaded {direct_image_url} for post id {post_id}')
	except Exception as e:
		print(f'failed to download {direct_image_url} ({type(e)})')
		note_errored_post(post_id, url)


def imgur_imageify(url):
	# change 'imgur.com' to 'i.imgur.com' and add '.png' to the url
	url[1] = 'i.imgur.com'
	url[2] += '.png'
	return urlunsplit(url)


def skip_post(post_id):
	# Keep a note to skip a post in future runs (only do this if the post has been deleted,
	# not just because the fetch failed -- it could mean we need a new function to handle the url)
	with open(SKIP_FILE_NAME, 'a') as file:
		file.write(f'{post_id}\n')


def note_errored_post(post_id, url):
	# Keep a note that a post has a URL that errored out while downloading to skip it next time,
	# and also save the URL for future analysis
	with open(ERROR_FILE_NAME, 'a') as file:
		file.write(f'{post_id}\n')
	with open(ERROR_URL_FILE_NAME, 'a') as file:
		file.write(f'{url}\n')


if __name__ == '__main__':
	main()
