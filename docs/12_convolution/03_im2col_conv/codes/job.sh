#!/bin/bash
make clean
make

./conv2ddemo  128 3   225 225 32  3   3   2   2   0   0
./conv2ddemo  49  128 35  35  384 3   3   2   2   0   0
./conv2ddemo  16  128 105 105 256 3   3   2   2   0   0
./conv2ddemo  128 3   230 230 64  7   7   2   2   0   0
./conv2ddemo  2   3   838 1350    64  7   7   2   2   0   0
./conv2ddemo  256 256 28  28  256 2   2   2   2   0   0
./conv2ddemo  128 3   225 225 32  3   3   1   1   0   0
