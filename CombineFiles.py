""" This script combines .csv files from BilayerHeightMeasurements.py into one .csv file
that is fed into MixtureModel.py"""

import os
import pandas as pd

def MrSifter(root_dir):

	"""This function returns the file path of all the tif files within root_dir."""

	fnames = []
	for root, dirs, files in os.walk(root_dir):

		## Loop through the files
		## Check for .tiff extension
		## save if it has one

		for file in files:
			if file.endswith("_thickresults.csv"): # "_NewResults.csv" for 5050sample, "_optiresults.csv" for 8020
				fnames.append(os.path.join(root, file))

		## YOU HAVE BEEN SIFTED

	return fnames

if __name__ == '__main__':
	
	root_dir = "C:\\Users\\caitl\\Documents\\Tomograms\\ThickTomo\\80Sample\\"

	files = MrSifter(root_dir)

	dfs = []
	for file in files:

		df = pd.read_csv(file)
		dfs.append(df)

	master = pd.concat(dfs)

	master.to_csv(root_dir + '80Sample.csv')



