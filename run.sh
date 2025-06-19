#!/bin/bash

OUT="out/"
FRAMEWORKS=("cupy" "dace_gpu" "jax" "parir_cuda" "torch_gpu")
BENCHMARKS=("adi" "arc_distance" "azimint_naive" "cavity_flow" "channel_flow" "cholesky" "compute" "conv2d_bias" "correlation" "covariance" "crc16" "deriche" "durbin" "fdtd_2d" "floyd_warshall" "go_fast" "gramschmidt" "hdiff" "heat_3d" "jacobi_1d" "jacobi_2d" "lenet" "lu" "ludcmp" "mlp" "nbody" "nussinov" "resnet" "scattering_self_energies" "seidel_2d" "softmax" "spmv" "symm" "syr2k" "syrk" "trisolv" "trmm" "vadv")
PRESET=L

rm -f npbench.db
rm -f ${OUT}/*.out ${OUT}/*.err
mkdir -p ${OUT}

echo "Running ${#BENCHMARKS[@]} benchmarks using ${#FRAMEWORKS[@]} frameworks"

for f in ${FRAMEWORKS[@]}; do
  echo "Benchmarking framework $f"
  for b in ${BENCHMARKS[@]}; do
    python run_benchmark.py -f $f -b $b -p $PRESET >> ${OUT}/$f.out 2>> ${OUT}/$f.err
  done
done
python plot_results.py -p $PRESET -b torch_gpu
