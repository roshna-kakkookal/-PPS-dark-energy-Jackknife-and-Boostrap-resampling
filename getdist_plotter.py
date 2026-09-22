
# Code to plot the corner plot of flat $\Lambda$ CDM model

import getdist.plots
from getdist import loadMCSamples, MCSamples
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import norm, f
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


flcdm = loadMCSamples('polychord_brout_chains_for_plotting/polychord_broutflatLCDM')  # folder which includes .txt and .paramnames

g = getdist.plots.get_subplot_plotter()

# Update font sizes in the plotter's settings
g.settings.axes_labelsize = 16    # Size of parameter names (axis labels)
g.settings.axes_fontsize = 16      # Size of tick labels (numbers on axes)
g.settings.legend_fontsize = 16    # Size of legend text
model_params = ["Omegam", "h", "M"]
g.settings.axis_tick_max_labels = 20
g.settings.linewidth = 1.5 
g.settings.legend_frame = False

g.triangle_plot([flcdm],model_params, filled=False, colour = 'black', legend_labels=None, param_limits={"Omegam": (0.25, 0.41), "h": (0.68, 0.82), "M": (-19.43, -19.05)})


ax = g.subplots[2,2]
ax.set_xticks([-19.35, -19.15])

#################################################

### functions for GLS

def deltachisquare(theta1, theta2, bf1, bf2, curvprime):
    delta = np.array([theta1-bf1,theta2-bf2])
    return np.matmul(np.matmul(delta.T,curvprime),delta)


#flat LCDM
parnum = 3
bestfit = np.array([0.33277768, 0.73467665, -19.24828116])
curvmat = np.array([[10812.75864179, 70704.54287207, -24030.05418411], [70704.54287207, 647902.48321677,-219166.28488175], [-24030.05418411, -219166.28488175, 75296.14959051]])
sigma = np.array([0.0180015, 0.01010854, 0.02943014])



#######################################


### Add contour of GLS
custom_levels = [2.279,5.991]

for j in range(1,parnum):
    i=0
    while(i<j):
        bestfit_i = bestfit[i]
        bestfit_j = bestfit[-parnum+j]
        curvprime = curvmat[[i,-parnum+j],:][:,[i,-parnum+j]]
        ax = g.subplots[j,i]
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
        Z_list = []
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                Z_list.append(deltachisquare(X[m,n],Y[m,n],bestfit_i,bestfit_j,curvprime))
        Z_arr =np.array(Z_list)
        Z = Z_arr.reshape(X.shape)
        ax.contour(X,Y,Z,levels=custom_levels, colors='red', linestyles='-', linewidths=1)
        i+=1

for i in range(parnum):
    ax = g.subplots[i,i]
    bf = bestfit[i]
    s = sigma[i]
    x_min, x_max = ax.get_xlim()
    X = np.linspace(x_min, x_max, 100)
    Z= norm.pdf(X,bf,s)#*np.sqrt(2*np.pi)*s
    ax.plot(X,Z/(np.max(Z)),ls='-',color='red',linewidth=1)


##########################################

# Add contour for Jackknife

# flcdm

# delete-1
parnum = 3
bestfit = np.array([0.29719553, 0.74266665, -19.23739817])
curvmat = np.array([[11335.01946191, 67863.5665913, -23215.81645396],
 [67863.5665913, 621959.24598378, -211202.7187944],
 [-23215.81645396, -211202.7187944, 71857.59662062]])
sigma = np.array([0.01639855, 0.02936804, 0.08747405])
numsamples = 1701
custom_levels = [f.ppf(0.68, 2, numsamples-2), f.ppf(0.95, 2, numsamples-2)]

print(custom_levels)

for j in range(1,parnum):
    i=0
    while(i<j):
        bestfit_i = bestfit[i]
        bestfit_j = bestfit[-parnum+j]
        curvprime = curvmat[[i,-parnum+j],:][:,[i,-parnum+j]]
        ax = g.subplots[j,i]
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
        Z_list = []
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                Z_list.append(deltachisquare(X[m,n],Y[m,n],bestfit_i,bestfit_j,curvprime))
        Z_arr =np.array(Z_list)
        Z = Z_arr.reshape(X.shape)
        ax.contour(X,Y,Z,levels=custom_levels, colors='green', linestyles='-', linewidths=1)
        i+=1

for i in range(parnum):
    ax = g.subplots[i,i]
    bf = bestfit[i]
    s = sigma[i]
    x_min, x_max = ax.get_xlim()
    X = np.linspace(x_min, x_max, 100)
    Z= norm.pdf(X,bf,s)#*np.sqrt(2*np.pi)*s
    ax.plot(X,Z/(np.max(Z)),ls='-',color='green',linewidth=1)




###############################################

# Add contour of bootstrap


from scipy.stats import gaussian_kde
# path of the param_list of bootstrap
bootstrap_wbc = np.loadtxt("/home/nitk217ph005/pantheon_analysis/Pantheon/10jan2026/flatLCDMbootstrap_param_list.txt")

bias = np.array([1.32769545e-02, -1.81539542e-03, 3.71429640e-05])
bootstrap = bootstrap_wbc - bias
custom_levels =[0.393, 0.86]
parnum = 3



for j in range(1, parnum):
    for i in range(j):
        ax = g.subplots[j, i]
        x = bootstrap[:, i]
        y = bootstrap[:, j]
        sns.kdeplot(x=x, y=y, ax=ax,
                              color='blue', linestyles=['-'], bw_adjust=1.5, levels=custom_levels)
for i in range(parnum):
    ax = g.subplots[i, i]
    x = bootstrap[:, i]
    kde = gaussian_kde(x)
    X = np.linspace(x.min(), x.max(), 1000)
    Y = kde(X)
    Y_values= Y/np.max(Y)
    sns.kdeplot(x=x, ax=ax, color='blue', linewidth=1, bw_adjust=1.5)
    line = ax.lines[-1]
    y_data = line.get_ydata()
    line.set_ydata(y_data / np.max(y_data))
    ax.relim()
    ax.autoscale_view()


########################################################

# Add legend and save

handles = [Line2D([0], [0], color='black', lw=1, label=r'Bayesian'),
    Line2D([0], [0], color='red', lw=1, label=r'GLS'),
    Line2D([0], [0], color='green', lw=1, label=r'Jackknife'),
    Line2D([0], [0], color='blue', lw=1, label=r'Bootstrap')]

g.fig.legend(handles=handles, loc='upper right', fontsize=14)


g.export('tri_allmethods_flcdm.pdf')
plt.close(g.fig)
del g

