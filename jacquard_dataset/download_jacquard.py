import urllib.request, os, tqdm, subprocess

def extract_zip(fpath_list):
	print('Extracting...')
	for fpath in tqdm.tqdm(fpath_list):
		subprocess.run(['unzip', fpath])
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
	url_list = [
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_0.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_1.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_2.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_3.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_4.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_5.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_6.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_7.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_8.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_9.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_10.zip',
		'https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_11.zip'
	]

	# Downloads and returns local file path list of downloaded files
	fpath_list = download_url_list(url_list)

	# Extracts and cleans up zips
	extract_zip(fpath_list)

if __name__ == '__main__':
	main()