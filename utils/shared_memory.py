""" Python multiprocessing with shared memory example.

This example demonstrate workaround for the GIL problem. Workaround uses
processes instead of threads and RawArray allocated from shared memory.

See also:
    [1] http://docs.python.org/2/library/multiprocessing.html
    [2] http://folk.uio.no/sturlamo/python/multiprocessing-tutorial.pdf
    [3] http://www.bryceboe.com/2011/01/28/the-python-multiprocessing-queue-and-large-objects/

"""

import time
import ctypes
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

def generateNodes(N):
    """ Generate random 3D nodes
    """

    return np.random.rand(N, 3)

def spCalcDistance(nodes):
    """ Single process calculation of the distance function.
    """

    p = nodes
    q = nodes.T

    # components of the distance vector
    Rx = p[:, 0:1] - q[0:1]
    Ry = p[:, 1:2] - q[1:2]
    Rz = p[:, 2:3] - q[2:3]

    # calculate function of the distance
    L = np.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
    D = L * L * L / 12 + L * L / 6

    return D

def mpCalcDistance_Worker(nodes, queue, arrD):
    """ Worker process for the multiprocessing calculations
    """

    nP = nodes.shape[0]
    nQ = nodes.shape[0]

    D = np.reshape(np.frombuffer(arrD), (nP, nQ))

    while True:
        job = queue.get()
        if job == None:
            break

        start = job[0]
        stop = job[0] + job[1]

        # components of the distance vector
        p = nodes[start:stop]
        q = nodes.T

        Rx = p[:, 0:1] - q[0:1]
        Ry = p[:, 1:2] - q[1:2]
        Rz = p[:, 2:3] - q[2:3]

        # calculate function of the distance
        L = np.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
        D[start:stop, :] = L * L * L / 12 + L * L / 6

        queue.task_done()
    queue.task_done()

def mpCalcDistance(nodes):
    """ Multiple processes calculation of the distance function.
    """

    # allocate shared array
    nP = nodes.shape[0]
    nQ = nodes.shape[0]

    arrD = mp.RawArray(ctypes.c_double, nP * nQ)

    # setup jobs
    #nCPU = mp.cpu_count()
    nCPU = 2
    nJobs = nCPU * 36

    q = nP / nJobs
    r = nP % nJobs

    jobs = []
    firstRow = 0
    for i in range(nJobs):
        rowsInJob = q
        if (r > 0):
            rowsInJob += 1
            r -= 1
        jobs.append((firstRow, rowsInJob))
        firstRow += rowsInJob

    queue = mp.JoinableQueue()
    for job in jobs:
        queue.put(job)
    for i in range(nCPU):
        queue.put(None)

    # run workers
    workers = []
    for i in range(nCPU):
        worker = mp.Process(target = mpCalcDistance_Worker,
                            args = (nodes, queue, arrD))
        workers.append(worker)
        worker.start()

    queue.join()

    # make array from shared memory
    D = np.reshape(np.frombuffer(arrD), (nP, nQ))
    return D

def compareTimes():
    """ Compare execution time single processing versus multiple processing.
    """
    nodes = generateNodes(3000)

    t0 = time.time()
    spD = spCalcDistance(nodes)
    t1 = time.time()
    print("single process time: {:.3f} s.".format(t1 - t0))

    t0 = time.time()
    mpD = mpCalcDistance(nodes)
    t1 = time.time()
    print("multiple processes time: {:.3f} s.".format(t1 - t0))

    err = np.linalg.norm(mpD - spD)
    print("calculate error: {:.2e}".format(err))

def showTimePlot():
    """ Generate execution time plot single processing versus multiple processing.
    """

    N = range(100, 4000, 4)
    spTimes = []
    mpTimes = []
    rates = []
    for i in N:
        print(i)
        nodes = generateNodes(i)

        t0 = time.time()
        spD = spCalcDistance(nodes)
        t1 = time.time()
        sp_tt = t1 - t0
        spTimes.append(sp_tt)

        t0 = time.time()
        mpD = mpCalcDistance(nodes)
        t1 = time.time()
        mp_tt = t1 - t0
        mpTimes.append(mp_tt)

        rates.append(sp_tt / mp_tt)

    plt.figure()
    plt.plot(N, spTimes)
    plt.plot(N, mpTimes)
    plt.xlabel("N")
    plt.ylabel("Execution time")

    plt.figure()
    plt.plot(N, rates)
    plt.xlabel("N")
    plt.ylabel("Rate")
    plt.show()

def main():
    compareTimes()
    #showTimePlot()

if __name__ == '__main__':
    main()