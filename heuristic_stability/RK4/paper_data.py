# In this python script we convert the data obtained from
#  the stability script to match the data used in the paper

# The definition of the Courant number for the data is:
# $Courant_data = \frac{\sqrt{u^{2} + v^{2}} \Delta t}{\delta}$, where
# $\delta = \Delta x = \Delta y$.


# In the paper we take the summation of the courant in all directions.
# for $\delta = \Delta x = \Delta y$ and $u=v=1$ we use
# $Courant_paper = \frac{Courant_data}{\sqrt{2}}$
# and Reh_paper = \frac{Reh_data}{\sqrt{2}}

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family': 'serif',
        'weight': 'normal',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('lines', lw=1)
matplotlib.rc('text', usetex=False)
plt.rcParams['mathtext.fontset'] = 'stix'
f, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
f.set_size_inches([5.25, 4.75])

ax.grid(which='minor', alpha=0.19)
colors = ['k', '#2E7990', '#A43F45', '#915F56', '#C49B4C']
markers = ['o', 's', 'D', 'v','*']

# load data
stability_data = np.loadtxt('./RK4-3-8-stability.txt',delimiter=",",usecols=[0,1])
#convert data to match the main results section
# stability_paper = stability_data/np.sqrt(2)

#convert data to match the unified formulation section
stability_paper = stability_data*1
# Reh conversion
stability_paper[:,1] = stability_data[:,1]/np.sqrt(2)/2
# Courant conversion
stability_paper[:,0] = stability_data[:,0]*np.sqrt(2)

#plot
Reh = stability_paper[2:,1]
Courant = stability_paper[2:,0]
ax.loglog(Reh, Courant, color=colors[1], linewidth=1.5, marker=markers[0], markersize=4.5, label="RK4 Taylor Vortex")
## plot the analytical stability boundaries
# inner_bound = lambda Pe: Pe/2.0/1.45
# outer_bound = lambda Pe: 2.0/np.sqrt(2)*np.ones_like(Pe)

## plot the unified stability formulation boundaries
inner_bound = lambda Pe: 2.09*Pe
outer_bound = lambda Pe: 2.84*np.ones_like(Pe)

ax.loglog(Reh, inner_bound(Reh),'--',color=colors[2],linewidth=1.5,label="Inner Bound")
ax.loglog(Reh, outer_bound(Reh),'-.',color=colors[3],linewidth=1.5,label="Outer Bound")


ax.set_ylabel('Courant', fontsize=12)
ax.set_xlabel('Cell Reynolds Number', fontsize=12)

# ax.set_ylim([4e-2,2])
plt.legend(ncol=1,loc='lower right',fontsize=10,frameon=False)

plt.tight_layout()
plt.show()

