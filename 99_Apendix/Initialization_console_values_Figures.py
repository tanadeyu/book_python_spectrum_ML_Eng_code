import os
os.system('cls') # Windows (Console clear)
#os.system('clear') # Linux

import sys
sys.modules[__name__].__dict__.clear()  # Variables clear

import matplotlib.pyplot as plt
plt.close('all') # Figure close
