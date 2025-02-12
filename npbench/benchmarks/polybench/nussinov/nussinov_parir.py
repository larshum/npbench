import parir
import torch

def match(b1, b2):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


@parir.jit
def parir_kernel(table, seq, N):
    # "Parallel" outermost loop to trick compiler
    for x in range(1):
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                if j-1 >= 0:
                    table[i,j] = max(table[i,j], table[i,j-1])
                if i+1 < N:
                    table[i,j] = max(table[i,j], table[i+1,j])
                if j-1 >= 0 and i+1 < N:
                    if i < j-1:
                        m = 1 if seq[i] + seq[j] == 3 else 0
                        table[i,j] = max(table[i,j], table[i+1,j-1] + m)
                    else:
                        table[i,j] = max(table[i,j], table[i+1,j-1])
                for k in range(i+1, j):
                    table[i,j] = max(table[i,j], table[i,k] + table[k+1,j])

def kernel(N, seq):
    table = torch.zeros((N, N), dtype=torch.int32, device=seq.device)
    parir_kernel(table, seq, N, parallelize={'x': [parir.threads(2)]})
    return table
