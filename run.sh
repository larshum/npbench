#!/bin/bash

OUT="out"

function run_benchmarks {
  if [ -f npbench.db ]; then
    echo "Found existing benchmark results - plotting immediately without running the suite"
  else
    rm -f ${OUT}/*.out ${OUT}/*.err
    mkdir -p ${OUT}

    echo "Running ${#BENCHMARKS[@]} benchmarks using ${#FRAMEWORKS[@]} frameworks (${FRAMEWORKS[@]})"

    for f in ${FRAMEWORKS[@]}; do
      echo "Benchmarking framework $f"
      if [ $f = $BASELINE ]; then
        VALIDATION_STR=""
      else
        VALIDATION_STR="-x $VALIDATOR"
      fi
      for b in ${BENCHMARKS[@]}; do
        timeout 40m python run_benchmark.py -f $f -b $b -p $PRESET ${BASELINE_STR} >> ${OUT}/$f.out 2>> ${OUT}/$f.err
      done
    done
  fi
  python plot_results.py -p $PRESET -b $BASELINE
}

function bench_cuda {
  export FRAMEWORKS=("cupy" "dace_gpu" "jax" "parpy_cuda" "torch_cuda")
  export BENCHMARKS=("adi" "arc_distance" "azimint_naive" "cavity_flow" "channel_flow" "cholesky" "compute" "conv2d_bias" "correlation" "covariance" "crc16" "deriche" "durbin" "fdtd_2d" "floyd_warshall" "go_fast" "gramschmidt" "hdiff" "heat_3d" "jacobi_1d" "jacobi_2d" "lenet" "lu" "ludcmp" "nbody" "nussinov" "resnet" "scattering_self_energies" "seidel_2d" "softmax" "spmv" "symm" "syr2k" "syrk" "trisolv" "trmm" "vadv")
  export PRESET=L
  export BASELINE=torch_cuda
  export VALIDATOR=numpy
  run_benchmarks
}

function bench_metal {
  export FRAMEWORKS=("jax32" "torch_metal" "parpy_metal" "numpy32")
  # We skip the following tests because numpy32 fails to validate compared to
  # the default numpy version (using 64-bit floats):
  # - adi
  # - azimint_naive
  # - durbin
  # - jacobi_1d
  # - lu, ludcmp
  # - nbody
  # - vadv
  export BENCHMARKS=("arc_distance" "cavity_flow" "channel_flow" "cholesky" "compute" "conv2d_bias" "correlation" "covariance" "crc16" "deriche" "fdtd_2d" "floyd_warshall" "go_fast" "gramschmidt" "hdiff" "heat_3d" "jacobi_2d" "lenet" "nussinov" "resnet" "scattering_self_energies" "seidel_2d" "softmax" "spmv" "symm" "syr2k" "syrk" "trisolv" "trmm")
  export PRESET=L
  export BASELINE=numpy32
  export VALIDATOR=numpy32
  run_benchmarks
}

case $1 in
  "cuda")
    bench_cuda
    ;;
  "metal")
    bench_metal
    ;;
esac
