"""UtilsPlot

Attributes:
    colors (TYPE): Colors for plotting
    plotParams (TYPE): Default plotting parameters
"""
#### Python imports

#### Library imports
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
# from IPython.core import debugger
# breakpoint = debugger.set_trace
import matplotlib
matplotlib.use('TkAgg')

#### Local imports
import Utils

#### Default matplotlib preferences
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plotParams = {
				'font.size': 12,
				'figure.dpi': 80,
				'figure.autolayout': True,
				'figure.titleweight': 'bold',
				'savefig.dpi': 200,
				'axes.titlesize': 10, # main title
				'axes.labelsize': 8, # x and y titles
				'axes.titleweight': 'bold', # x and y titles
				'axes.labelweight': 'bold', # x and y titles
				'grid.linestyle': '--',
				'grid.linewidth': 1,
				'text.usetex': False,
				'xtick.labelsize': 8,
				'xtick.minor.visible': True,
				'ytick.labelsize': 8,
				'ytick.minor.visible': True,
				'lines.linewidth': 1,
				'lines.markersize': 4.0,
				'legend.fontsize': 8,
				'legend.shadow': True,
				}

#mpl.use('Qt4Agg', warn=False) ## Needed to allow drawing with matplotlib during debug mode
plt._INSTALL_FIG_OBSERVER = True

mpl.rcParams.update(plotParams)
plt.ion()


def PlotCodingScheme(ModFs, DemodFs):
	"""PlotCodingScheme: Create a 1x3 figure with modulation, demodulation, and the correlation.
	
	Args:
	    modF (numpy.ndarray): Modulation functions. N x K matrix.
	    demodF (numpy.ndarray): Demodulation functions. N x K matrix
	
	Returns:
	    plt.figure: Figure handle
	    plt.axis: Axis handle
	"""
	#### Assume the following constants
	totalEnergy = 1.
	tau = 1.
	averagePower = totalEnergy / tau
	#### Reshape to ensure needed dimensions
	## Assume that the number of elements is larger than the number of coding pairs, i.e. rows>cols
	if(ModFs.shape[0] < ModFs.shape[1]): ModFs = ModFs.transpose()
	if(DemodFs.shape[0] < DemodFs.shape[1]): DemodFs = DemodFs.transpose()
	#### Verify Inputs
	assert(ModFs.shape == DemodFs.shape), "Input Error - PlotCodingScheme: ModFs and \
	DemodFs should be the same dimensions."
	#### Set some parameters
	(N,K) = ModFs.shape
	avgPower = torch.sum(ModFs[:,0])/N 
	#### Set default values
	t = torch.linspace(0, tau, N)
	phase = torch.linspace(0, 2*np.pi,N)
	#### Reshape to ensure same dimensions
	t = t.reshape((N,))
	#### Get Correlation functions
	CorrFs = Utils.GetCorrelationFunctions(ModFs=ModFs,DemodFs=DemodFs)
	#sio.savemat('np_vector.mat', {'vect':CorrFs})
	#### Plot Decomposition
	## Clear current plot
	plt.clf()
	## Get current figure
	fig = plt.gcf()
	## Add subplots and get axis array
	for i in range(K):
		# breakpoint()
		fig.add_subplot(K,3,3*i + 1)
		fig.add_subplot(K,3,3*i + 2)
		fig.add_subplot(K,3,3*i + 3)
	axarr = fig.get_axes()
	## Make all plots
	## Calculate Avg power. 
	avgPower = torch.sum(ModFs[:,0]) / N
	avgPower = [avgPower for i in range(0, N)]

	## Plot ObjCorrF first so that stars don't cover the corrFs.
	for i in range(0, K):
		labelInfo = str(i)
		axarr[3*i + 0].plot(t.numpy(), ModFs[:,i].detach().numpy(), label='Md-'+labelInfo,linewidth=2, color=colors[i])
		axarr[3*i + 1].plot(t.numpy(), DemodFs[:,i].detach().numpy(), label='Dmd-'+labelInfo,linewidth=2, color=colors[i])
		axarr[3*i + 2].plot(phase.numpy(), CorrFs[:,i].detach().numpy(), label='Crr-'+labelInfo,linewidth=2, color=colors[i])
		axarr[3*i + 0].plot(t.numpy(), avgPower, '--', label='AvgPower', linewidth=3, color=colors[i])
		## Set axis labels
		axarr[3*i + 0].set_xlabel('Time')
		axarr[3*i + 1].set_xlabel('Time')
		axarr[3*i + 2].set_xlabel('Phase')
		axarr[3*i + 0].set_ylabel('Instant Power')
		axarr[3*i + 1].set_ylabel('Exposure')
		axarr[3*i + 2].set_ylabel('Magnitude')

	## Set Titles
	axarr[0].set_title('Modulation')
	axarr[1].set_title('Demodulation')
	axarr[2].set_title('Correlation')
	# ## Set ylimit so that we can see the legend
	# axarr[0].set_ylim([0,1.2*np.max(ModFs)])
	# axarr[1].set_ylim([0,1.2*np.max(DemodFs)])
	# axarr[2].set_ylim([0,1.2*np.max(CorrFs)])	

	plt.savefig('test.png')

	return (fig, axarr)


