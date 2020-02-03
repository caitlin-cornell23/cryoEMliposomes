""" This script imports 0 angle images from Tomograms of lipid bilayers. A Canny edge detection algorithm is applied to the image to find the
edges of the bilayer. The process of determining sigma for the gaussian filter in the Canny algorithm is semi-
iterative. The edge image is a boolean array that is then converted to a binary array. The binary array is converted
to XY. These XY coordinates are converted to polar coordinates and the origin is shifted to the center of the image. 
The radii and theta values for each point along an edge are stored in a pandas dataframe. The program differentiates 
between the two edges by taking an average between the two and setting a boundary for above and below the average value. 
The theta values are binned so that radii are an average value within each bin. Once the inner and outer leaflet
are determined, the script calculates the difference between a pixel on the inner leaflet with every pixel
on the outer leaflet and chooses the minimum distance. The final height measurements are averaged by
taking a rolling average over two pixel values. This information is stored in a dataframe."""


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Canny filter
from skimage.feature import canny
# Denoise 
from skimage.restoration import denoise_nl_means, estimate_sigma
# Equalize histogram
from skimage.exposure import equalize_hist


## Set global matplotlib parameters
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (20,18)
plt.rcParams["image.cmap"] = "bone"

def MrSifter(root_dir):

	'''This function returns the file path of all the tif files within root_dir.'''

	fnames = []
	for root, dirs, files in os.walk(root_dir):

		## Loop through the files
		## Check for .tif extension
		## save if it has one

		for file in files:
			if file.endswith(".tif"):
				fnames.append(os.path.join(root, file))

	return fnames


def InteractiveCanny(Image):
	''' This function allows the user to interactively change the values of sigma in the 
	gaussian filter portion of the Canny edge detection algorithm. '''

	Filtered = canny(Image, sigma=4) # 4 for tomos and 6 for projections
	plt.imshow(Filtered)
	plt.show()

	## Inquire if the sigma value is correct
	Question = int(input('Input 1 if the filter i good, 0 if it is bad: '))

	## Decide whether to move on or test different values of sigma
	if Question !=0:
		New_sigma = float(input('Enter the original value of sigma: '))
	else:
		New_sigma = float(input('Enter new value of sigma: '))

	## Generate a new filtered image
	Filtered = canny(Image, sigma=New_sigma)

	## Show the new image
	plt.imshow(Filtered)
	plt.show()
	plt.close()

	return Filtered

def TemplateArray(Image):

	""" This function takes an array of binary filtered images and returns
	a mapped template array in polar coordinates."""

	## Input the image resolution in nm per pixel 
	res = 0.191 # on TF20, 0.254

	## Create an XY array of the same shape as the image
	x,y = np.arange(Image.shape[1]), np.arange(Image.shape[0])
	X,Y = np.meshgrid(x, y)

	## Reposition the origin of the template array to be in the center of the filtered image
	X = Image * X #-Image.shape[1] # X for right, its Image*X-Image.shape[1] for left
	Y = Image *(Y-(0.5*Image.shape[0])) # This gives 1/2 the yaxis length

	## Convert the XY coordinates to polar coordinates (theta, r)
	## Invert the theta coordinate so thtat it goes from 0 to pi
	fr = (np.sqrt(X**2 + Y**2)*Image)*res
	ft = abs(Image*np.arctan2(X,Y)) # normally, arctan2 takes Y,X but this is inverted

	## Isolate the radii and theta values for the edges and store them
	## in an array outside the loop
	fr_edge = fr[fr !=0]
	ft_edge = ft[fr !=0] # us fr so that both are flattened in the same way

	## Create a pandas dataframe to store the values
	df = pd.DataFrame({"fr":fr_edge, "theta":ft_edge})

	## Create an array of values for bins in theta
	theta_bins = np.linspace(0, np.pi, 30)

	## Apply bins to the values of theta and add to the dataframe
	df["theta_cut"] = pd.cut(df.theta, theta_bins)

	## Make labels for the bins that are the middle value in the bin range
	theta_labels = 0.5*(theta_bins[:-1]+theta_bins[1:])
	df["theta_labels"] = pd.cut(df.theta, theta_bins, labels=theta_labels)

	## Convert the theta labels to float values instead of categories
	df.theta_labels = df.theta_labels.astype(float)

	## Find the average between the two edges so that we 
	## can distinguish between the two by seeing their position
	## with respect to the average value
	fr_avg = df.groupby("theta_labels")["fr"].mean()

	## Define the top edge of the two edges by putting an addition to the data frame
	## that checks the radius values against the average radius values in the bins
	## and if they are above those values, places it in the top
	df["top_edge"] = df["fr"] > fr_avg.loc[df["theta_labels"]].values 

	## Create a new data frame with only the values for the top edge
	top_df = df.loc[df["top_edge"] == True]

	## Define the bottom edge and create a new dataframe with only those values
	bottom_df = df.loc[df["top_edge"] == False]

	return top_df, bottom_df



if __name__ == '__main__':
	

	## Set the root directory
	root_dir = "C:\\Users\\caitl\\Documents\\Tomograms\\HalfSample\\50Lo_Height\\Right\\"

	## Return a list of file names
	fnames = MrSifter(root_dir)

	## Loop through the images
	for name in fnames:
		Image = cv2.imread(name, 0) # 0 is a key to import the image as greyscale

		## Parse the file name to get the image name
		sample = name[name.rfind("Right\\")+6:name.rfind(".tif")] # change depending on names (left is 5, right is 6)
		print(f"File name = {sample}")

		## Select an ROI on the image
		roi = cv2.selectROI(Image)

		## Close the selection window
		cv2.destroyWindow('roi')

		## Crop the image
		imCrop = Image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

		## Display the cropped image
		cv2.imshow("Cropped Image", imCrop)
		cv2.waitKey(0)

		## Enhance the contrast by equalizing the histogram
		# imCrop = cv2.equalizeHist(imCrop)

		# ## Estimate the noise standard deviation from the image for Denoising
		# sigma_est = np.mean(estimate_sigma(imCrop, multichannel=False))
		# print(f"estimated noise standard deviation = {sigma_est}")

		# ## Denoise the image using a slow algorith.
		# ## Spatial Gaussian weighting is applied to patches when computing distances.
		# ## H is the cut-off distance. A higher H means more permissive of accepting patches.
		# ## It results in a smoother image, but can blur features. 
		# Denoise = denoise_nl_means(imCrop, patch_size = 5, patch_distance = 6,
		#  						   h=1.2 * sigma_est, fast_mode=False, multichannel=False)

		# ## Equalize the histogram of the denoised image
		# ImageEnhanced = equalize_hist(Denoise)

		## Perform a Canny Filter algorithm
		Filtered = InteractiveCanny(imCrop)

		# figs, axes = plt.subplots(1, 4, figsize=(28,10))

		# axes[0].imshow(imCrop)
		# axes[0].set_title("Contrast enhanced")

		# axes[1].imshow(Denoise)
		# axes[1].set_title("Denoised")

		# axes[2].imshow(ImageEnhanced)
		# axes[2].set_title("Image Contrast again")

		# axes[3].imshow(Filtered)
		# axes[3].set_title("Edge Detected")
		# plt.show()

		## Create a dataframe containing r and theta information about the edge
		## detected in the Canny filter
		top_df, bottom_df = TemplateArray(Filtered)

		## Compute the thicknesses by finding the minimum distance from a point
		## on the bottom to any point on the top
		optim_thickness = []
		for bx in zip(bottom_df["fr"], bottom_df["theta"]):
			delta = np.sqrt(top_df["fr"].values**2 + bx[0]**2-2.*top_df["fr"].values * bx[0]*np.cos(top_df["theta"].values - bx[1]))
			optim_thickness.append(delta.min())
		optim_thickness = pd.Series(optim_thickness, index=bottom_df["theta"], name="thickness")

		## Compute a rolling average of the optim_thickness values to smooth out
		## pixel to pixel variations
		optim_thickness = optim_thickness.rolling(window=2, center=False).mean()

		## Calculate the averages for the top and bottom edges and store
		top_avg = top_df.groupby("theta_labels")["fr"].mean()
		bottom_avg = bottom_df.groupby("theta_labels")["fr"].mean()

		## Calculate the thickness of the bilayer
		thickness = top_avg - bottom_avg

		# Rename the vector to the correct name and store
		thickness.rename("thickness", inplace=True)

		## Plot the two different thickness measurements
		fig, axes = plt.subplots(figsize=(10,8))
		axes.plot(thickness.index,thickness.values,lw=2,c="k", label="binned theta method")
		axes.plot(optim_thickness,color="C3",marker="o",markersize=13, label="optimization method")
		axes.set_xlabel("Theta")
		axes.set_ylabel("Bilayer Thickness (nm)")
		axes.legend()
		plt.tight_layout()
		plot_file = name[:name.rfind(".tif")]
		#plt.savefig(plot_file + "_optim.pdf")
		plt.show()
		plt.close()

		## Plot the radii vs theta figure
		figs, axes = plt.subplots(1,1, figsize = (10,8))
		axes.plot(top_df.theta, top_df.fr, marker="o", Linestyle="None", markersize=15, alpha=0.3, color= '#004445')
		axes.plot(bottom_df.theta, bottom_df.fr, marker="o", Linestyle="None", markersize=15, alpha=0.3, color = '#6FB98F' )
		axes.plot(top_df.theta_labels, top_df.fr, marker="s", Linestyle="None", markersize=12, alpha=0.5, color='#2C7873')
		axes.plot(bottom_df.theta_labels, bottom_df.fr, marker="s", Linestyle="None", markersize=15, alpha=0.5, color = '#6FB98F' )
		axes.set_xlabel("Theta (rad)")
		axes.set_ylabel("Radii (nm)")
		plt.tight_layout()

		## Save the figure with the image title
		plt.savefig(plot_file + ".pdf")
		plt.show()
		plt.close()

		## Save the optimum thickness to a .csv file
		df_opti = pd.DataFrame({"optim_thickness":optim_thickness})
		df_opti.to_csv(name[:name.rfind(".tif")] + "_NewResults.csv")


