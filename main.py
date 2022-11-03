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
    grid_size = grid_size-2  # exclude boundary

    height = grid_size//3

    length = ceil(grid_size*height/size)

    start = length*rank
    end = start+length

    if rank == size-1:
        end = grid_size*height

    return start, end


def get_rank_for_cell(x: int, y: int, grid_size: int, size: int, gap_size: int):
    # x,y in boundry guards
    if x not in range(1, grid_size-1) or y not in range(1, grid_size-1):
        return -1

    if x in range((grid_size-gap_size)//2, (grid_size+gap_size)//2) and y in range((grid_size-gap_size)//2, (grid_size+gap_size)//2):
        return -1

    # exclude boundary
    grid_size = grid_size-2
    x, y = x-1, y-1

    height = grid_size//3

    length = ceil(grid_size*height/size)

    pos = x+(y//3*grid_size)  # possibly x, y should be reversed

    return pos//length


def hash_cell_pos(x, y, grid_size):
    return x*grid_size+y


def main(**kwargs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    grid_size = kwargs['grid_size']
    gap_size = kwargs['gap_size']

    curr_layer = np.zeros((grid_size, grid_size))
    add_boundary_condition(
        curr_layer, gap_size, kwargs['external_voltage'])

    prev_layer = np.zeros((grid_size, grid_size))
    add_boundary_condition(
        prev_layer, gap_size, kwargs['external_voltage'])

    chunk_start, chunk_end = get_chunk(grid_size, rank, size)

    inner_grid_size = grid_size-2
    print(f'Job {rank} working on chunk: ({chunk_start}, {chunk_end})')
    # send inital messages to neighbors
    for i in range(chunk_start, chunk_end):
        for j in range(1, 4):
            x = 1+i % inner_grid_size
            y = j+3*(i//inner_grid_size)

            # get tuple (neighbor rank, cells x, cells y)

            neighbor_ranks = [
                (get_rank_for_cell(x_, y_, grid_size, size, gap_size), x_, y_) for x_, y_ in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            ]

            # send messages with the calculated values to neighbors in different ranks
            for rank_, x_, y_ in neighbor_ranks:
                if rank_ not in [rank, -1]:
                    comm.isend(curr_layer[x, y], dest=rank_,
                               tag=hash_cell_pos(x, y, grid_size))

    for t in range(kwargs['iterations']):
        for i in range(chunk_start, chunk_end):
            for j in range(1, 4):
                x = 1+i % inner_grid_size
                y = j+3*(i//inner_grid_size)

                # check self rank_
                if x in range((grid_size-gap_size)//2, (grid_size+gap_size)//2) and y in range((grid_size-gap_size)//2, (grid_size+gap_size)//2): continue

                # get tuple (neighbor rank, cells x, cells y)
                neighbor_ranks = [
                    (get_rank_for_cell(x_, y_, grid_size, size, gap_size), x_, y_) for x_, y_ in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                ]

                # get values for neighbor cells
                neighbor_values = [
                    prev_layer[x_, y_] if rank_ in [rank, -1] else comm.recv(source=rank_, tag=hash_cell_pos(x_, y_, grid_size)) for rank_, x_, y_ in neighbor_ranks
                ]


                curr_layer[x, y] = calculate_value(neighbor_values)

                # send messages with the calculated values to neighbors in different ranks
                for rank_, x_, y_ in neighbor_ranks:
                    if rank_ not in [rank, -1]:
                        comm.isend(curr_layer[x, y], dest=rank_,
                                tag=hash_cell_pos(x, y, grid_size))
        # update layers
        prev_layer = np.copy(curr_layer)
    print(curr_layer)
    np.save(file=f'before_merge_{rank}.npy', arr=curr_layer)

    # wait for all to finish before assembling final results
    comm.Barrier()

    if rank == 0:
        # list of results from all processes
        results = [curr_layer]

        # collect results from other workers
        for i in range(1, size):
            data = np.empty(shape=(grid_size, grid_size))
            comm.Recv(data, source=i, tag=grid_size**2)
            results.append(data)

        # for each cell get value from associated worker and copy it to curr_layer as result
        for x in range(1, grid_size):
            for y in range(1, grid_size):
                rank = get_rank_for_cell(
                    x, y, grid_size, size, gap_size)

                if rank not in [-1, 0]:
                    curr_layer[x, y] = results[rank][x, y]
        print('\n---------------------RESULT---------------------\n')
        print(curr_layer)
        np.save(file=f'result.npy', arr=curr_layer)
    else:
        # send result for collection
        comm.Send(curr_layer, dest=0, tag=grid_size**2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Program Name',
        description='Description.',
    )

    np.set_printoptions(precision=3, linewidth=400)

    parser.add_argument('filename')
    parser.add_argument('-g', '--grid_size', type=int, default=20)
    parser.add_argument('--gap_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--external_voltage', type=float, default=1.)

    main(**vars(parser.parse_args(sys.argv)))
