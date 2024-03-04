import warnings
warnings.filterwarnings('ignore')
from numpy import *
import numpy
from pylab import *
from matplotlib.colors import LogNorm
import os
import time
from scipy.signal import *
import nds2
import pickle

import sys
print()
print(sys.executable)
print('Hello')


raise
import sys
sys.path.append('/Users/simon/Dropbox/Python/pycrime/')
from pycrime import *




matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'figure.figsize': (10,6)})

def save_fig(fig_id, tight_layout=True):
    path = fig_id + '.png'
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=100)
def save_fig_pdf(fig_id, tight_layout=True):
    path = fig_id + '.pdf'
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=100)
def pcolormesh_logscale(T, F, S):
    pcolormesh(T, F, S, norm=LogNorm(vmin=S.min(), vmax=S.max()))
    
