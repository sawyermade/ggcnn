import urllib.request, os, tqdm, subprocess

def extract_tar_gz(fpath_list):
	print('Extracting...')
	for fpath in tqdm.tqdm(fpath_list):
		subprocess.run(['tar', 'zxf', fpath])
		subprocess.run(['rm', fpath])

def download_url_list(url_list, download_dir='.'):
	print('Downloading...')
	fpath_list = []
	for url in tqdm.tqdm(url_list):
		fname = url.split('/')[-1]
		fpath = os.path.join(download_dir, fname)
		fpath_list.append(fpath)
		urllib.request.urlretrieve(url, fpath)

	return fpath_list

def main():
	# download urls from http://pr.cs.cornell.edu/grasping/rect_data/data.php
	url_list = [
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data01.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data02.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data03.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data04.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data05.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data06.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data07.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data08.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data09.tar.gz',
		'http://pr.cs.cornell.edu/grasping/rect_data/temp/data10.tar.gz'
	]

	# Downloads and returns local file path list of downloaded files
	fpath_list = download_url_list(url_list)

	# Extracts files from tar.gz fpath_list and deletes them
	extract_tar_gz(fpath_list)

if __name__ == '__main__':
	main()