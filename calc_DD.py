import sys
import time
import numpy as np
from multiprocessing import Pipe, Process

from global_vars import *


ST = time.time()


class Worker(Process):

    """
    Subclass for multiprocessing. Process for calculating DD in parallel.
    Accepts range of values to roll R and a child pipe to send completed
    DD histogram to parent process.
    """

    def __init__(self, ID, chunk, pipe):
        super().__init__()
        self.ID = ID  # Process ID
        self.chunk = chunk  # Range of values for which to roll data
        self.pipe = pipe  # Pipe to send DD back to master process
        self.DD = np.zeros((NB_LNR, NB_MU), dtype=np.int64)  # Initialize DD
        self.high = list(self.chunk)[-1]  # Maximum roll
        self.low = list(self.chunk)[0] - 1  # Minimum roll
        self.length = self.high - self.low  # Roll range

    def run(self):
        print("Worker " + str(self.ID) + " started")

        for i in self.chunk:
            self.DD += calc_DD(i)  # Calculate DD for roll by i
            if i % 1000 == 0:  # Print completion percentage
                pct = str(((i - self.low) / self.length) * 100)[:5] + "% "
                print("Worker " + str(self.ID) + ": " + pct + "Time: ", end="")
                fmt_time(time.time() - ST)

        self.pipe.send(self.DD)  # Send DD histogram when completed
        self.pipe.close()
        print("Worker " + str(self.ID) + " finished. Time: ", end="")
        fmt_time(time.time() - ST)


def fmt_time(sec):
    """
    Helper function for printing completion percentage and time.
    """
    h = int(sec / 3600)
    m = int((sec - (3600 * h)) / 60)
    s = sec - (3600 * h) - (60 * m)
    formatted = str(h) + ":" + str(m) + ":" + str(s)[:5]
    print(formatted)


def divide(it):
    """
    Divide NGAL into CORES equal ranges to pass to individual processes
    for rolling data.
    Returns a list of range objects.
    """
    remainder = it % CORES
    chunk_size = (it - remainder) // CORES
    chunks = []

    for i in range(CORES):
        if i == CORES - 1:
            # append largest chunk last
            chunks.append(range(i*chunk_size+1, (i+1)*chunk_size+1+remainder))
        else:
            chunks.append(range(i*chunk_size+1, (i+1)*chunk_size+1))

    return chunks


def calc_DD(i):
    """
    Compute DD from input coordinate data.
    """
    center = 0.5 * (R + np.roll(R, i, axis=0))  # Center of pair separation
    vdiff = np.roll(R, i, axis=0) - R  # Vector along pair separation
    vdiff[np.where(vdiff > 1500)] -= 3000  # Wrap-around corrections
    vdiff[np.where(vdiff < -1500)] += 3000
    diff = np.einsum('ij, ij->i', vdiff, vdiff)**0.5  # Magnitude of separation
    lnr = np.log10(diff)

    # Compute mu from center and separation
    dot = np.einsum('ij, ij->i', vdiff, center)
    center = np.einsum('ij, ij->i', center, center)**0.5
    mu = dot / (center * diff)

    # Create (NB_R x NB_MU) DD histogram
    return np.histogram2d(lnr, mu, bins=(NB_LNR, NB_MU),
                          range=((LNR_MIN, LNR_MAX), (MU_MIN, MU_MAX))
                          )[0].astype(np.int64)


def main():

    fin_name = "gal_3D_xyz." + sys.argv[1] + ".txt"
    fout_name = fin_name + ".DD"

    # Read in coordinates
    CP = np.loadtxt(fin_name, usecols=(0, 1, 2))
    CV = np.loadtxt(fin_name, usecols=(3, 4, 5))
    NGAL = len(CP)

    # Apply RSD
    global R
    R = CP * (1 + (np.sum(CP * CV) / np.sum(CP ** 2)))

    # Determine number of rolls to perform on data
    global EVEN, IT
    IT = NGAL // 2
    EVEN = NGAL % 2 == 0
    if not EVEN:
        IT += 1

    # Calculate roll chunks for each process
    chunks = divide(IT)

    # Initialize pipes for processes
    pipes = []
    for i in range(CORES):
        pipes.append(Pipe())

    # Initialize worker processes
    processes = []
    for i in range(CORES):
        process = Worker(i, chunks[-i - 1], pipes[i][1])
        process.start()
        processes.append(process)
    del chunks

    # Receive individual DDs from workers
    DD_final = np.zeros((NB_LNR, NB_MU), dtype=np.int64)
    """
    done = 0
    while done != CORES:
        for pipe in pipes:
            if pipe[0].poll():
                DD_final += pipe[0].recv()
                done += 1
    """

    for pipe in pipes:
        DD_final += pipe[0].recv()

    # Join processes
    for p in processes:
        p.join()

    # Save DD file
    np.savetxt(fout_name, DD_final, fmt='%d')

    print("All done. Time: ", end="")
    fmt_time(time.time() - ST)

if __name__ == '__main__':
    main()
