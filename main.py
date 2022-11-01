from math import ceil
import sys
import argparse
from mpi4py import MPI
import numpy as np

def calulate_value(V_prev: tuple[int, int, int, int]):
    return sum(V_prev)/4

def test_layers_diff(prev_layer: np.ndarray, curr_layer: np.ndarray, epsilon: float) -> bool:
    # TODO: consider division
    for i in prev_layer.shape[0]:
        for j in prev_layer.shape[1]:
            if abs(prev_layer[i][j]-curr_layer[i][j]) > epsilon:
                return False # if diff is too large return False
    return True

def add_boundary_condition(arr: np.ndarray, gap_size: int):
    size = arr.shape[0]
    gap_start = (size-gap_size)//2
    gap_end = size-gap_start
    arr[gap_start:gap_end, gap_start:gap_end] = 1
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0

def get_chunk(arr: np.ndarray, rank: int, size: int):
    grid_size = arr.shape[0]-2

    height = grid_size//size

    start = rank*height
    end = start+height

    if rank == size-1: end = grid_size 

    return arr[start:end+2, :]


def main(**kwargs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        print(kwargs)

    # prev_layer = np.ones(kwargs['grid_size'], kwargs['grid_size'])
    # add_boundary_condition(prev_layer)

    # curr_layer = np.zeros(kwargs['grid_size'], kwargs['grid_size'])
    # add_boundary_condition(curr_layer)

    grid_size: int = kwargs['grid_size']

    itemsize = MPI.DOUBLE.Get_size() 
    if rank == 0: 
        nbytes = (grid_size+2) ** 2 * itemsize 
    else:
        nbytes = 0

    # on rank 0, create the shared block
    # on rank 1 get a handle to it (known as a window in MPI speak)
    
    # win_prev = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    # win_curr = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

    # # create a numpy array whose data points to the shared mem
    # buf_prev, itemsize = win_prev.Shared_query(0)
    # buf_curr, itemsize = win_curr.Shared_query(0)

    # prev_layer = np.ndarray(buffer=buf_prev, dtype='d', shape=(grid_size+2, grid_size+2))
    # add_boundary_condition(prev_layer)

    # curr_layer = np.ndarray(buffer=buf_curr, dtype='d', shape=(grid_size+2, grid_size+2))
    # add_boundary_condition(curr_layer)

    win = MPI.Win.Allocate_shared(nbytes*2, itemsize, comm=comm)

    # create a numpy array whose data points to the shared mem
    buf, itemsize = win.Shared_query(0)

    prev_layer = np.ndarray(buffer=buf, dtype='d', shape=(grid_size+2, grid_size+2))
    add_boundary_condition(prev_layer)

    curr_layer = np.ndarray(buffer=buf, dtype='d', offset=nbytes, shape=(grid_size+2, grid_size+2))
    add_boundary_condition(curr_layer)

    calc_finished = False
    # Skąd rank != 0 wiedzą jaki jest calc_finished
    # while(rank != 0 or not calc_finished):
    while(not calc_finished):
        prev_layer = np.copy(curr_layer)
        # calcs here

        prev_chunk = get_chunk(prev_layer, rank, size)
        curr_chunk = get_chunk(curr_layer, rank, size)

        for i in range(1, curr_chunk.shape[0]):
            for j in range(1, curr_chunk.shape[1]):
                curr_chunk[i, j] = calulate_value((prev_chunk[i-1, j], prev_chunk[i+1, j], prev_chunk[i, j-1], prev_chunk[i, j+1]))

        comm.Barrier() 
        if rank == 0:
            calc_finished = test_layers_diff(prev_layer=prev_layer, curr_layer=curr_layer, epsilon=kwargs['epsilon'])
            # kontynuacja z 91 lini: tutaj jakiś sygnał do pozostałych wątków żeby wypierdalać




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'Program Name',
        description = 'Description.',
    )

    parser.add_argument('filename')
    parser.add_argument('-n', '--n_threads', type=int, default=1)
    parser.add_argument('-g', '--grid_size', type=int, default=20)
    parser.add_argument('--gap_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--external_voltage', type=float, default=1.)

    main(**vars(parser.parse_args(sys.argv)))

"""

## ## ## ## ## ## ul ## ##
## ## ## ## ## ## ## ## ##
## ## ## ## ## ## dl ## ##
## ## ur ## ## ## ## ## ##
## ## ## ## ## ## ## ## ##
## ## dr ## ## ## ## ## ##
## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ##

"""