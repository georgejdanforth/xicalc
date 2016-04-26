import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from global_vars import *


def plot_xi(xi, pole, absolute=True):
    """
    Plot xi using pyplot
    """
    # Set up x-axis
    R = pow(10, LNR_MIN + ((np.arange(NB_LNR) + 0.5) * LNR_BW))

    # Spline xi
    if absolute:
        tck = interp.splrep(R, np.absolute(xi))
    else:
        tck = interp.splrep(R, xi)
    R_new = LNR_MIN + ((np.arange(NB_LNR * 20) + 0.5) * pow(10, LNR_BW))
    xi_spl = interp.splev(R_new, tck, der=0)

    # Plot log scale if absolute==True, otherwise plot linear
    if absolute:
        plt.plot(R_new, np.abs(xi_spl), color='k')
        plt.yscale('log')
        plt.axis((30, 200, 1e-4, 0.1))
        if pole == 0:
            plt.ylabel(r'$|\xi_0(r)|$', fontsize=12)
        elif pole == 2:
            plt.ylabel(r'$|\xi_2(r)|$', fontsize=12)
        elif pole == 4:
            plt.ylabel(r'$|\xi_4(r)|$', fontsize=12)
    else:
        plt.plot(R_new, xi_spl, color='k')
        plt.axis((30, 200, -0.001, 0.005))
        if pole == 0:
            plt.ylabel(r'$\xi_0(r)$', fontsize=12)
        elif pole == 2:
            plt.ylabel(r'$\xi_2(r)$', fontsize=12)
        elif pole == 4:
            plt.ylabel(r'$\xi_4(r)$', fontsize=12)
    plt.xlabel(r'$r \; [Mpc/h]$', fontsize=14)
    plt.show()


def main():

    # Get list of DD filenames in directory
    all_files = [f for f in os.listdir() if f.endswith('.DD')]
    n_files = len(all_files)

    # Initialize lists of xi for each file
    all_xi0 = []
    all_xi2 = []
    all_xi4 = []

    for f in all_files:

        # Read in DD file
        DD = np.loadtxt(f)
        npair = np.sum(DD)

        # Initialize arrays for xi
        xi0 = np.zeros(NB_LNR)
        xi2 = np.zeros(NB_LNR)
        xi4 = np.zeros(NB_LNR)

        for ir in range(NB_LNR):

            # Compute RR analytically
            r_min = pow(10, LNR_MIN + ir * LNR_BW)
            r_max = pow(10, LNR_MIN + (ir + 1) * LNR_BW)
            vol = (4 / 3) * np.pi * ((r_max**3) - (r_min**3))
            RR = ((vol / VOL_TOT) * npair) / NB_MU  # Fraction RR per bin in mu

            for imu in range(NB_MU):

                mu = MU_MIN + (imu + 0.5) * MU_BW

                xi0[ir] += (DD[ir][imu] - RR)
                xi2[ir] += (DD[ir][imu] - RR)*0.5*(3*(mu**2)-1)
                xi4[ir] += (DD[ir][imu] - RR)*0.125*(35*(mu**4)-30*(mu**2)+3)

            xi0[ir] /= (RR * NB_MU)
            xi2[ir] /= (RR * NB_MU)
            xi4[ir] /= (RR * NB_MU)

        # Append xis to lists
        all_xi0.append(xi0)
        all_xi2.append(xi2)
        all_xi4.append(xi4)

    # Average across all xi
    xi0_av = np.sum(np.array(all_xi0), axis=0) / n_files
    xi2_av = np.sum(np.array(all_xi2), axis=0) / n_files
    xi4_av = np.sum(np.array(all_xi4), axis=0) / n_files

    # Plot xi
    plot_xi(xi0_av, 0, absolute=True)
    plot_xi(xi2_av, 2, absolute=True)
    plot_xi(xi4_av, 4, absolute=True)


if __name__ == '__main__':
    main()
