#!/usr/bin/env python
from mpi4py import MPI
import socket
import math

n = 100

def is_prime(x, arr):
    return all([x%el!=0 for el in arr])

B = range(2, math.ceil(math.sqrt(n)))

B = [el for el in B if is_prime(el, B[:(el-2)])]

is_p = lambda x: is_prime(x, B)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("my rank is: %d, out of %d processes at node %s"%(rank, size, socket.gethostname()))

partition_len = math.ceil((n-math.ceil(math.sqrt(n)))/size)
A_start = math.ceil(math.sqrt(n))+rank*partition_len
A_end = math.ceil(math.sqrt(n))+min((1+rank)*partition_len, n)

if A_start >= A_end:
    comm.send([], dest=0)

print(f'I\'m calculating partition <{A_start}, {A_end})')

A = range(A_start, A_end)

A_processed = [el for el in A if is_p(el)]

comm.Barrier()
full_table = comm.reduce(A_processed, op=MPI.SUM)
if rank == 0:
    print(full_table)
