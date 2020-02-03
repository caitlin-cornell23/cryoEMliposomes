""" MixtureModel.py

This script uses Lo and Ld thickness measurements to approximate the probability
of observing certain bilayer thicknesses in each phase.

Those approximations form a 2 component mixture model that we fit to mixed samples,
infering the fraction of each sample in each phase. 

Data comes from processed imagery. """

## For file paths, etc.
import os

## For data manipulation, I/O, and
## visualization.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Set up the plotting environment
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["lines.linewidth"] = 2.0


## Set some colors as well, used in the plots
## throughout.
light_blue = "#90AFC5"
blue = "#336B87"
dark_blue = "#2A3132"
red = "#763626"
lightest_green = "#C4E2CC"
light_green = '#92C7AC'
dark_green = '#3C6571'

## For model fitting and KDE construction
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

def LogPosterior(alpha,p_t_Lo,p_t_Ld,
				 a=1,b=1):

	""" This is the log of the posterior probability distribution,

					p(alpha | {t in sample}, p_Lo(t), p_Ld(t))

		with a beta distribution, Beta(a,b), as the prior. This can be numerically
		integrated to give an exact posterior or it can be optimized via scipy.minimize
		to compute a gaussian approximation to alpha. 

		Note: p_t_Lo and p_t_Ld should be numpy arrays computed by evaluating the 
			  respective KDEs at each of the sample heights. """

	## Start by computing the contribution of the prior. This comes
	## from the probability density of the beta distribution. See
	## https://en.wikipedia.org/wiki/Beta_distribution for more
	## details.
	log_prior = (a-1)*np.log(alpha) + (b-1)*np.log(1-alpha)

	## Now compute the likelihood.
	log_like = np.log(alpha*p_t_Lo + (1-alpha)*p_t_Ld).sum()

	return log_prior + log_like

def SamplePosterior(alpha,posterior,num_samples=5000):

	""" This is wrapper on np.random.choice which uses the alpha values and their
		associated posterior probabilities to draw samples. This technically treats
		alpha as a discrete parameter, so the spacing between alphas has to be very small,
		i.e. alpha should be a long numpy array. """

	return np.random.choice(alpha,replace=True,size=(num_samples,),p=posterior)

if __name__ == "__main__":

	## Get the data from CSV prepared by Caitlin
	print("\nGetting the data...")
	data_dir = os.path.join("C:\\Users\\caitl\\Documents\\PythonScripts\\TEM","mixture1")
	Lo_samples = pd.read_csv(os.path.join(data_dir,"Lo.csv"),
							 header=None,
							 usecols=[1],
							 dtype={1:np.float64},
							 names=["thickness"])["thickness"]
	Ld_samples = pd.read_csv(os.path.join(data_dir,"Ld.csv"),
							 header=None,
							 usecols=[1],
							 dtype={1:np.float64},
							 names=["thickness"])["thickness"]
	mixed_samples = pd.read_csv(os.path.join(data_dir,"Height.csv"),
								header=None,
								usecols=[1],
								dtype={1:np.float64},
								names=["thickness"])["thickness"]

	## Construct KDE approximations. We use the Lo and
	## Ld approximation for fitting. The mixed_sample KDE
	## is just for visualizing the results.
	print("Computing KDEs...")
	p_Lo = gaussian_kde(Lo_samples.values)
	p_Ld = gaussian_kde(Ld_samples.values)
	p_mixed = gaussian_kde(mixed_samples.values)

	## Evaluate the Lo and Ld distributions at each of
	## the mixed points.
	p_t_Lo = p_Lo(mixed_samples.values)
	p_t_Ld = p_Ld(mixed_samples.values)

	## Compute the log poserior accross alpha points. This is done
	## by evaluating the function above and then exponentiating with
	## a scale factor to keep everything numerically stable. Finally,
	## the posterior is normalized by dividing by the total (i.e. a 
	## right-hand rule numerical integration).
	print("Calculating posterior distribution...")
	alpha = np.linspace(1e-6,1-1e-6,1000)
	posterior = np.zeros(alpha.shape)
	normalization = 0
	stability_constant = LogPosterior(0.5,p_t_Lo,p_t_Ld)
	for i, a in enumerate(alpha):
		posterior[i] = np.exp(LogPosterior(a,p_t_Lo,p_t_Ld)\
							  -stability_constant)
		normalization += posterior[i]
	posterior = posterior/normalization

	## Compute the mean and standard deviation by integrating
	## the posterior times alpha and the posterior time (alpha-E[alpha])**2
	mean_alpha = (alpha*posterior).sum()
	var_alpha = (posterior*(alpha-mean_alpha)**2).sum()
	std_alpha = np.sqrt(var_alpha)
	print("Alpha estimate = {} +/- {}".format(mean_alpha,2.*std_alpha))

	## Plot the results
	fig, axes = plt.subplots(1,2, figsize=(9,5))

	## Set up an x axis that covers a wide range
	## of thicknesses
	x_low = min([Lo_samples.min(),
				 Ld_samples.min(),
				 mixed_samples.min()])
	x_high = max([Lo_samples.max(),
				  Ld_samples.max(),
				  mixed_samples.max()])
	x = np.linspace(0.9*x_low,1.1*x_high,500)

	## Plot the KDE approximations
	axes[0].plot(x,p_Ld(x), color=lightest_green)
	axes[0].fill_between(x,0,p_Ld(x),
					  color=lightest_green,alpha=0.5)
	axes[0].plot(x,p_mixed(x), color=light_green)
	axes[0].fill_between(x,0,p_mixed(x),
					  color=light_green,alpha=0.5)
	axes[0].plot(x,p_Lo(x), color=dark_green)
	axes[0].fill_between(x,0,p_Lo(x),
					  color=dark_green,alpha=0.5)

	## Finish up the details, axes limits, etc.
	axes[0].set(ylim=(0,2.25),xlabel="Measured thickness",
			 ylabel=r"$p(t\,|\,$sample type$)$")
	axes[0].set_xlim(1,5)
	axes[0].set_yticks([])
	fig.tight_layout()
	
	# ## Plot the posterior distribution
	# axes[1].fill_between(alpha,0,posterior,
	# 				  color=red,alpha=0.8)
	# axes[1].errorbar(mean_alpha,1.025*posterior.max(),xerr=2.*std_alpha,
	# 			  color="k",lw=1,marker="o",ls="None")
	# axes[1].set(xlim=(0,1),ylim=(0,None),
	# 		 xlabel=r"$\alpha$",
	# 		 ylabel=r"$p(\alpha\,|\,$dataset$)$")
	# axes[1].set_yticks([])
	# fig.tight_layout()

	## Make a plot of the KDE fit 
	print("Computing fit and uncertainty...")
	alpha_samples = SamplePosterior(alpha,posterior,
									num_samples=10000)
	p_Lo_x = p_Lo(x)
	p_Ld_x = p_Ld(x)
	fit_samples = np.zeros((len(alpha_samples),len(x)))
	for i, a in enumerate(alpha_samples):
		fit_samples[i] = a*p_Lo_x + (1-a)*p_Ld_x

	## Compute the CI's via np.percentile (just some summary)
	## statistics of the sample
	mean = mean_alpha*p_Lo_x + (1-mean_alpha)*p_Ld_x
	low = np.percentile(fit_samples,2.5,axis=0)
	high = np.percentile(fit_samples,97.5,axis=0)
	
	## Plot the fit
	axes[1].fill_between(x,low,high,color=dark_blue,alpha=0.5)
	axes[1].plot(x,mean,color=dark_blue,alpha=1)
	axes[1].plot(x,p_mixed(x),color=light_green,alpha=1)
	axes[1].fill_between(x, 0, p_mixed(x), color=light_green, alpha=0.5)
	axes[1].set(ylim=(0,2.25),
			 xlabel="Measured thickness",
			 ylabel=r"$p(t\,|\,$sample type$)$")
	axes[1].set_xlim(1,5)
	axes[1].set_yticks([])
	fig.tight_layout()
	#fig.savefig(os.path.join("C:\\Users\\caitl\\Documents\\PythonScripts\\TEM","FigureMixture1_Nov18.pdf"))

	## Finish up
	plt.show()
