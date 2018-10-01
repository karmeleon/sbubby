import os.path

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