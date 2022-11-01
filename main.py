from math import ceil
import sys
import argparse
from mpi4py import MPI
import numpy as np


def calculate_value(V_prev: tuple[int, int, int, int]):
    return sum(V_prev)/4


def layers_diff(prev_layer: np.ndarray, curr_layer: np.ndarray) -> bool:
    # TODO: consider division
    return np.max(np.abs(prev_layer-curr_layer))


def add_boundary_condition(arr: np.ndarray, gap_size: int, external_voltage: float):
    size = arr.shape[0]
    gap_start = (size-gap_size)//2
    gap_end = size-gap_start
    arr[gap_start:gap_end, gap_start:gap_end] = external_voltage
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0


def get_chunk(grid_size: int, rank: int, size: int) -> tuple[int, int]:
    height = grid_size//3

    length = ceil(grid_size*height/size)

    start = length*rank
    end = start+length

    if rank == size-1:
        end = grid_size*height

    return start, end


def main(**kwargs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        print(kwargs)

    grid_size: int = kwargs['grid_size']

    itemsize = MPI.DOUBLE.Get_size()
    if rank == 0:
        nbytes = (grid_size+2) ** 2 * itemsize
    else:
        nbytes = 0

    win = MPI.Win.Allocate_shared(nbytes*2, itemsize, comm=comm)

    # create a numpy array whose data points to the shared mem
    buf, itemsize = win.Shared_query(0)

    prev_layer = np.ndarray(buffer=buf, dtype='d',
                            shape=(grid_size+2, grid_size+2))
    add_boundary_condition(
        prev_layer, kwargs['gap_size'], kwargs['external_voltage'])

    curr_layer = np.ndarray(buffer=buf, dtype='d',
                            offset=nbytes, shape=(grid_size+2, grid_size+2))
    add_boundary_condition(
        curr_layer, kwargs['gap_size'], kwargs['external_voltage'])

    should_stop = False

    # grid_size = 18, gap = 6
    gap_start = (grid_size+2-kwargs['gap_size'])//2
    gap_end = grid_size+2-gap_start

    while (should_stop):
        prev_layer = np.copy(curr_layer)

        chunk_start, chunk_end = get_chunk(grid_size, rank, size)

        for i in range(chunk_start, chunk_end):
            for j in range(1, 4):
                x = 1+i % grid_size
                y = j+3*(i//grid_size)
                if x in range(gap_start, gap_end) and y in range(gap_start, gap_end):
                    continue
                curr_layer[x, y] = calculate_value(
                    (prev_layer[x-1, y], prev_layer[x+1, y], prev_layer[x, y-1], prev_layer[x, y+1]))

        comm.Barrier()

        if rank == 0:
            should_stop = layers_diff(
                prev_layer=prev_layer, curr_layer=curr_layer) < kwargs['epsilon']

            for i in range(1, size):
                comm.send(obj=should_stop, dest=i)
        else:
            should_stop = comm.recv(source=0)

        comm.Barrier()

    if rank != 0:
        exit(0)
    print(curr_layer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Program Name',
        description='Description.',
    )

    parser.add_argument('filename')
    parser.add_argument('-g', '--grid_size', type=int, default=20)
    parser.add_argument('--gap_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--external_voltage', type=float, default=1.)

    main(**vars(parser.parse_args(sys.argv)))

"""

mpiexec -n <thread_num> python3 main.py

"""
