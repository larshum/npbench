import parir
import torch


@parir.jit
def parir_kernel(TSTEPS, N, A):
    parir.label('outer')
    for x in range(1):
        for t in range(0, TSTEPS-1):
            for i in range(1, N-1):
                for k in range(1, N-1):
                    A[i,k] += (A[i-1,k-1] + A[i-1,k] + A[i-1,k+1] +
                               A[i,k+1] + A[i+1,k-1] + A[i+1,k] +
                               A[i+1,k+1])
                for j in range(1, N-1):
                    A[i,j] += A[i,j-1]
                    A[i,j] /= 9.0


def kernel(TSTEPS, N, A):
    parir_kernel(TSTEPS, N, A, parallelize={'outer': [parir.threads(2)]})
