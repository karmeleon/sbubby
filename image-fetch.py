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

COUNT = 0

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

	img_path = os.path.abspath('images')

	if not os.path.isdir(img_path):
		os.mkdir(img_path)

	# get just the filename (without path or extension)
	existing_images = {os.path.splitext(os.path.split(f)[1])[0] for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))}

	bound_download_image = functools.partial(download_image, existing_images, img_path)

	# fire up a thread pool to download + save the images in parallel
	with ThreadPoolExecutor(max_workers=4) as pool:
		pool.map(bound_download_image, posts)


def download_image(existing_images, img_path, post_data):
	post_id, url = post_data
	# don't download anything we've already downloaded
	if post_id in existing_images:
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
		r = requests.get(direct_image_url)
		# dump it as a png
		image = Image.open(BytesIO(r.content))
		# check to see if it's actually worth saving
		if image.size == (130, 60):
			print(f'{direct_image_url} is (probably) a deleted reddit image post :(')
			return
		if image.size == (161, 81):
			print(f'{direct_image_url} is (probably) a deleted imgur post :(')
			return

		image.save(os.path.join(img_path, f'{post_id}.jpg'))
		print(f'downloaded {direct_image_url} for post id {post_id}')
	except Exception as e:
		print(f'failed to download {direct_image_url} ({type(e)})')


def imgur_imageify(url):
	# change 'imgur.com' to 'i.imgur.com' and add '.png' to the url
	url[1] = 'i.imgur.com'
	url[2] += '.png'
	return urlunsplit(url)

if __name__ == '__main__':
	main()