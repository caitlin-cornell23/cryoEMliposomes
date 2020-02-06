""" This script performs the same function as ColorMapMask.py version 1, with some operational changes. The script
loads a .csv file that contains all of the measurement data from the BilayerHeightMeasurement.py script. KDE's are 
computed from the data for Lo, Ld, and mixed samples. A color map is plotted on the original image by taking the
bilayer thickness value from the sample and computing a 'likelihood ratio' that the measurement falls in either
the Lo or the Ld distributions."""

## For file paths, etc.
import os

## For data manipulation, I/O, and
## visualization.
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

## For image manipulation
import cv2
from skimage.feature import canny

## For custom color map
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

## For KDE construction
from scipy.stats import gaussian_kde

## Set up the plotting environment
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["image.cmap"] = "bone"

## Set some colors as well, used in the plots
## throughout.
red = '#cc0000'
blue = '#0039e6'
white = '#ffffff'

def MrSifter(root_dir):

	"""This function returns the file path of all the tif files within root_dir."""

	fnames = []
	for root, dirs, files in os.walk(root_dir):

		## Loop through the files
		## Check for .tiff extension
		## save if it has one

		for file in files:
			if file.endswith("_NewResults.csv"): # "_NewResults.csv" for 5050sample, "_optiresults.csv" for 8020
				fnames.append(os.path.join(root, file))

		## YOU HAVE BEEN SIFTED

	return fnames

def InteractiveCanny(image):

	""" This function allows the user to interactively change the values of sigma
	in the gaussian filter portion of the Canny edge detection algorithm. The Canny filtered
	binary image is returned."""

	## Compute the Canny Filter
	Filtered = canny(image, sigma=4)
	plt.imshow(Filtered)
	plt.show()

	## Inquire if the sigma value is correct
	Question = int(input('Input 1 if the filter is good, 0 if it is bad: '))

	## Decide whether to move on or test different values of sigma
	if Question !=0:
		New_sigma = float(input('Enter the original value of sigma: '))
	else:
		New_sigma = float(input('Enter the new value of sigma: '))

	## Generate a new filtered image
	Filtered = canny(image, sigma=New_sigma)

	return Filtered

def TemplateArray(image):

	""" This function takes an array of binary filtered images and their cut-off
	values and returns a mapped template array in polar coordinates."""

	## Create the cut-off version of the image
	plt.imshow(image)
	plt.show()
	cut_off_x = int(input('Enter the cut_off_x number: '))
	cut_off_y = int(input('Enter the cut_off_y number: '))
	cut_filter = image[:cut_off_y, :cut_off_x]
	plt.imshow(cut_filter)
	plt.show()

	## Input the image resolution in nm per pixel
	res = 0.191 # 0.191 for Glacios and 0.254 for TF20

	## Create the XY array of the same shape as the image
	x, y = np.arange(cut_off_x), np.arange(cut_off_y)
	X,Y = np.meshgrid(x,y)

	## Store the pixel locations in image space
	pixel_x = X*cut_filter
	pixel_y = Y*cut_filter
	pixel_x = pixel_x[pixel_x != 0]
	pixel_y = pixel_y[pixel_y != 0]

	## Re-position the origin of the template array to be in the center
	## of the filter image
	X = cut_filter*X # Change for right side image, # *X for right side, * X-cut_off_x for left
	Y = cut_filter*(Y-(0.5*cut_off_y))

	## Convert the XY coordinates to polar coordinates (theta, r)
	## Invert the theta coordinate so that it goes from 0 to pi
	fr = (np.sqrt(X**2 + Y**2)*cut_filter)*res
	ft = abs(cut_filter*np.arctan2(X,Y))

	## Isolate the radii and theta values for the edges and store them
	## in an array outside the loop
	fr_edge = fr[fr !=0]
	ft_edge = ft[fr !=0] # use fr so that both are flattened the same way

	## Create a pandas dataframe to store the values
	df = pd.DataFrame({"fr":fr_edge, "theta":ft_edge, 
					   "pixel_x":pixel_x, "pixel_y":pixel_y})

	## Create an array of values for bins in theta
	theta_bins = np.linspace(0, np.pi, 30)

	## Apply bins to the values of theta and add to the dataframe
	df["theta_cut"] = pd.cut(df.theta, theta_bins)

	## Make labels for the bins that are the middle value in the bin range
	theta_labels = 0.5*(theta_bins[:-1]+theta_bins[1:])
	df["theta_labels"] = pd.cut(df.theta,theta_bins, labels=theta_labels)

	## Convert the theta labels to float values instead of categories
	df.theta_labels = df.theta_labels.astype(float)

	## Find the average between the two edges so that we
	## can distinguish between the two by seeing their position
	## with respect to the average value
	fr_avg = df.groupby("theta_labels")["fr"].mean()

	## Define the top edge of the two edges by putting an
	## addition to the data frame that checks the radius
	## values against the average radius values in the bings
	## and if they are above those values, places it in the top
	df["top_edge"] = df["fr"] > fr_avg.loc[df["theta_labels"]].values

	# Create a new data frame with only the values for the top edge
	top_df = df.loc[df["top_edge"] == True]

	## Define the bottom edge and create a new dataframe with only those values
	bottom_df = df.loc[df["top_edge"] == False]

	return top_df, bottom_df, cut_filter

def GetKDEs():

	## Define the root directory
	root_dir = "C:\\Users\\caitl\\Documents\\Tomograms\\HalfSample\\"
	## For 80/20 sample, "C:\\Users\\caitl\\Documents\\Tomograms\\"
	## For 50/50 sample, "C:\\Users\\caitl\\Documents\\Tomograms\\HalfSample\\"

	## Return a list of filenames
	fnames = MrSifter(root_dir)

	## Store the filenames in a list
	Lo = []
	Ld = []

	## Loop through the files and sort the .csv files into dataframes for each sample category
	for name in fnames:

		## Parse the file name to get the parent folder
		sample = name[name.rfind("HalfSample\\")+11:name.rfind("_Height")+7]
		## For 80/20 sample, sample = name[name.rfind("Tomograms\\")+10:name.rfind("_Height")+7]
		## For 50/50 sample, sample = name[name.rfind("HalfSample\\")+11:name.rfind("_Height")+7]
		
		if sample == "50Lo_Height":
			## For 80/20 sample, sample == "Lo_Height"
			## For 50/50 sample, sample == "50Lo_Height"
			Lo.append(name)
		if sample == "50Ld_Height":
			## For 80/20 sample, sample == "Ld_Height"
			## For 50/50 sample, sample == "50Ld_Height"
			Ld.append(name)

	## Loop through the files in each category and import the data

	# Set up the empty lists
	Lo_df = []
	Ld_df = []
	for name in Lo:
		## Import the data
		Lo_df.append(pd.read_csv(name, header=0, index_col=None))
	for name in Ld:
		## Import the data
		Ld_df.append(pd.read_csv(name, header=0, index_col=None))

	## Concatenate each of the category dataframes
	Lo_df = pd.concat(Lo_df, axis=0)
	Lo_thickness = Lo_df["optim_thickness"] 
	## Cut out non-physical values
	Lo_thickness = Lo_thickness[(Lo_thickness >= 2) & (Lo_thickness <= 6)] 

	Ld_df = pd.concat(Ld_df, axis=0)
	Ld_thickness = Ld_df["optim_thickness"]
	## Cut out non-physical values
	Ld_thickness = Ld_thickness[(Ld_thickness >= 2) & (Ld_thickness <= 6)]


	## Compute KDEs for Height, Lo, and Ld
	Lo_kernel = gaussian_kde(Lo_thickness, bw_method='scott')
	Ld_kernel = gaussian_kde(Ld_thickness, bw_method='scott')

	return Lo_kernel, Ld_kernel

if __name__ == '__main__':
	
	## Load in an image to analyze
	Image = cv2.imread("C:\\Users\\caitl\\Documents\\Tomograms\\HalfSample\\50Both_Height\\Right\\Sample_046_7.tif", 0)

	## Perform a Canny Filter on the image
	Filtered = InteractiveCanny(Image)

	## Create a dataframe containing r and theta information about the 
	## edge detected in the Canny Filter
	top_df, bottom_df, cut_filter = TemplateArray(Filtered)

	## Compute the thickness by finding the minimum distance from a point
	## on the bottom to any point on the top
	thickness = []
	pairs = []
	centers = []
	for i, row in bottom_df.iterrows():
		bx = (row["fr"], row["theta"])
		delta = np.sqrt(top_df["fr"].values**2 +bx[0]**2-2.*top_df["fr"].values
						*bx[0]*np.cos(top_df["theta"].values-bx[1]))
		min_index = np.argmin(delta)
		tx = (top_df["pixel_x"].values[min_index],top_df["pixel_y"].values[min_index])
		thickness.append(delta[min_index])
		pairs.append([(row["pixel_x"],row["pixel_y"]),tx])
		centers.append((0.5*(tx[0]+row["pixel_x"]), 0.5*(tx[1]+row["pixel_y"])))
	thickness = pd.Series(thickness,index=bottom_df["theta"], name="thickness")
	#thickness_avg = thickness.rolling(window=2, center=True).mean()
	centers = np.array(centers)

	################### Construct KDEs ########################

	## Get the KDEs
	Lo_kde, Ld_kde = GetKDEs()

	## Probability density of being Lo phase or Ld phase
	Lo_density = Lo_kde(thickness.values)
	Ld_density = Ld_kde(thickness.values)

	## Percentage of probability density ratio 
	#  percentage = Lo_density/(Lo_density + Ld_density)

	## Probability of Lo phase given distance
	## From MixtureModel.py, the mixing ratio (alpha) is 0.43 for Ratio 2 and 0.20 for Ratio 5
	prob_Lo = (Lo_density * 0.20) / ((0.20 * Lo_density) + (1- 0.20) * Ld_density) 
	percentage = prob_Lo * 100

	plt.imshow(cut_filter)
	for p in pairs:
		bx, by = p[0]
		tx, ty = p[1]
		plt.plot([bx,tx],[by,ty],c="C3")
		
	fig, axes = plt.subplots(1,3, figsize=(32,14))

	################### Continuous color map ########################

	# Create a customized cmap
	basic_cols=[blue, white,  red]
	newcmp=LinearSegmentedColormap.from_list('mycmap', basic_cols)
	# RdBuBig = matplotlib.cm.get_cmap('RdBu', 512)
	# newcmp = matplotlib.colors.ListedColormap(RdBuBig(np.linspace(0.2, 0.9, 256)))

	#Normalize the colors based on the likelihood ratio
	norm = matplotlib.colors.Normalize(vmin=percentage.min(),vmax=percentage.max(),clip=True)
	#Map the colormap colors to the normalized values
	mapper = matplotlib.cm.ScalarMappable(norm=norm,cmap=newcmp)
	#Create an array of RGBA values for the colors
	colors = [mapper.to_rgba(prd) for prd in percentage]

	# Use contourf to provide colorbar info, then clear figure
	Z = [[0,0],[0,0]]
	steps = 0.1
	levels = np.arange(percentage.min(), percentage.max(),steps)
	CS3 = plt.contourf(Z, levels, cmap=newcmp)

	################### Plot ########################

	# Plot the thickness values with the likelihood ratio
	axes[0].plot(centers[:,1],thickness.values,lw=2,color="grey", alpha=0.5)
	axes[0].scatter(centers[:,1],thickness.values, marker="o",color=colors, s=20**2, edgecolor="#000000")
	axes[0].set_ylim(1.0,4.0,1)
	axes[0].set(xlabel="Pixel position",ylabel="Thickness (nm)")
	axes[0].set_title("Filtered Thickness Values")
	
	# Plot the original image (cropped)
	Image = Image[:197, :61]
	axes[1].imshow(Image, cmap='bone')
	axes[1].set_title("Original Image")

	# Plot the filtered image with the likelihood_ratio superimposed on top
	axes[2].imshow(cut_filter)
	#plt.scatter(bottom_df["pixel_x"].values,bottom_df["pixel_y"].values,color=colors,marker="s",s=10**2)
	axes[2].scatter(centers[:,0],centers[:,1],color=colors,marker="s",s=20**2, linestyle='None')
	axes[2].set_title("Filtered Image")
	plt.colorbar(CS3)
	
	
	plot_file = "C:\\Users\\caitl\\Documents\\Tomograms\\HalfSample\\50Both_Height\\Right\\"
	plt.savefig(plot_file + "Sample_046_7.pdf")
	plt.show()
