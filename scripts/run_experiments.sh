#!/bin/bash

echo "=== Running benchmarks ==="

echo "Gradient Descent on Quadratic (A-scale=2.0):"
python scripts/run_benchmark.py --optimizer gradient_descent --problem quadratic \
    --lr 0.01 --n-iters 100 --dim 2 --A-scale 2.0 --b-scale 1.0

echo "Heavy Ball (m=0.9) on Quadratic (A-scale=2.0):"
python scripts/run_benchmark.py --optimizer heavy_ball --problem quadratic \
    --lr 0.01 --momentum 0.9 --n-iters 100 --dim 2 --A-scale 2.0 --b-scale 1.0

echo "Heavy Ball (m=0.5) on Quadratic (A-scale=2.0):"
python scripts/run_benchmark.py --optimizer heavy_ball --problem quadratic \
    --lr 0.01 --momentum 0.5 --n-iters 100 --dim 2 --A-scale 2.0 --b-scale 1.0

echo "Gradient Descent on Quadratic (A-scale=20.0):"
python scripts/run_benchmark.py --optimizer gradient_descent --problem quadratic \
    --lr 0.01 --n-iters 100 --dim 2 --A-scale 20.0 --b-scale 1.0

echo "Heavy Ball (m=0.9) on Quadratic (A-scale=20.0):"
python scripts/run_benchmark.py --optimizer heavy_ball --problem quadratic \
    --lr 0.01 --momentum 0.9 --n-iters 100 --dim 2 --A-scale 20.0 --b-scale 1.0

echo "Heavy Ball (m=0.5) on Quadratic (A-scale=20.0):"
python scripts/run_benchmark.py --optimizer heavy_ball --problem quadratic \
    --lr 0.01 --momentum 0.5 --n-iters 100 --dim 2 --A-scale 20.0 --b-scale 1.0

echo "Gradient Descent on Rosenbrock:"
python scripts/run_benchmark.py --optimizer gradient_descent --problem rosenbrock \
    --lr 0.001 --n-iters 500 --dim 2

echo "Heavy Ball (m=0.9) on Rosenbrock:"
python scripts/run_benchmark.py --optimizer heavy_ball --problem rosenbrock \
    --lr 0.001 --momentum 0.9 --n-iters 500 --dim 2

echo "Heavy Ball (m=0.5) on Rosenbrock:"
python scripts/run_benchmark.py --optimizer heavy_ball --problem rosenbrock \
    --lr 0.001 --momentum 0.5 --n-iters 500 --dim 2

echo ""
echo "=== Comparing results ==="

echo "Quadratic (A-scale=2.0) comparison:"
python scripts/compare_methods.py --problem quadratic_dim2_A2.0_b1.0

echo "Quadratic (A-scale=20.0) comparison:"
python scripts/compare_methods.py --problem quadratic_dim2_A20.0_b1.0

echo "Rosenbrock comparison:"
python scripts/compare_methods.py --problem rosenbrock_dim2

echo ""
echo "Done. Results saved to results/"