#!/bin/bash

OUT="out/"
frameworks="cupy dace_gpu jax parir torch"

rm -f npbench.db
mkdir -p ${OUT}

for f in $frameworks; do
  echo "Running benchmarks for framework $f"
  python run_framework.py -f $f -p L > ${OUT}/$f.out 2> ${OUT}/$f.err
done

python plot_results.py -p L -b torch
