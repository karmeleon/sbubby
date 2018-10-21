import os.path

from PIL import Image

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
