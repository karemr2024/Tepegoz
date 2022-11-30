from scipy.io import savemat
import numpy as np
import glob
import os

def np2mat():
    npzFiles = glob.glob("*.npy", root_dir= "C:\Tepegoz\Images\Temp_np")
    for f in npzFiles:
        fm = os.path.splitext(f)[0]+'.mat'
        d = np.load(f)
        savemat(fm, d)
        print('generated ', fm, 'from', f)

