import os.path
from urllib.parse import urlparse, urlunparse

from io import BytesIO
from PIL import Image

import requests

IMAGE_SIZE = 256

def get_filenames_in_directory_without_extension(path):
	return {
		os.path.splitext(os.path.split(f)[1])[0]
		for f in os.listdir(path)
		if os.path.isfile(os.path.join(path, f)) and is_file_non_empty(os.path.join(path, f))
	}

def is_file_non_empty(file):
	try:
		size = os.path.getsize(file)
		if size > 0:
			return True
	except:
		pass
	return False

def process_image(image):
	"""
	Crop, resize, and pad an image to a given square size.
	:param image: a PIL image
	:returns: a PIL image
	"""
	im = image.convert('RGB')
	# Resize the image down, then paste it on a square canvas to pad it to 1:1
	im.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
	canvas = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
	upper_left = (
		int((IMAGE_SIZE - im.size[0]) / 2),
		int((IMAGE_SIZE - im.size[1]) / 2),
	)
	canvas.paste(im, upper_left)

	return canvas

def fetch_image(url):
	"""
	:param url: a string url of an image to download
	:returns: a PIL Image object, or None if the download or parse failed for some reason
	"""
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
		r = requests.get(direct_image_url, timeout=5)
		# load the image
		image = Image.open(BytesIO(r.content))
		# check to see if it's actually worth saving
		if image.size == (130, 60):
			return None
		if image.size == (161, 81):
			return None
	except Exception:
		# stuff breaks sometimes, we don't really care
		return None
	
	# return which sub we got it from to keep track of how many images of each sub we actually download
	return image

def imgur_imageify(url):
	# change 'imgur.com' to 'i.imgur.com' and add '.png' to the url
	url._replace(netloc='i.imgur.com', path='{}.png'.format(url.path))
	return urlunparse(url)
