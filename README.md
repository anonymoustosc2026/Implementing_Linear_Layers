# Depth-Priority Greedy + SAT Search for CNOT Circuits

This program searches for low-depth in-place CNOT implementations of binary linear layers.  
It combines several greedy search strategies with a depth-first SAT-based local refinement.
This program uses the CaDiCaL SAT solver.

Example compilation command:

g++ -O3 -std=c++17 sxor_depth_priority_satfix.cpp \
    /mnt/f/cadical-master/build/libcadical.a \
    -o sxor_search


Input format

The input matrix is read from standard input.

The expected format is:

1
n n
a_00 a_01 ... a_0,n-1
a_10 a_11 ... a_1,n-1
...
a_n-1,0 ... a_n-1,n-1


Basic command:
./sxor_search greedy_rounds sat_rounds min_window max_window extra_gate_allow \
    pool_per_depth conflict_limit output_file \
    [perm_rounds] [block_rounds] [random_perm_count] [refine_depth_slack] [rng_seed] \
    < matrix.txt

Example:

./sxor_search 50000 3 2 5 8 20 200000 result.txt \
    2000 50000 64 0 12345 < AES.txt

Parameters
greedy_rounds:	Number of greedy searches on the original target matrix.

sat_rounds:	Number of SAT refinement rounds for each selected candidate.

min_window:	Minimum window length for SAT local refinement.

max_window:	Maximum window length for SAT local refinement.

extra_gate_allow:	Extra CNOT gates allowed in SAT refinement when trying to reduce depth.

pool_per_depth:	Maximum number of candidates kept for each depth.

conflict_limit:	CaDiCaL conflict limit for each SAT call.

output_file:	File used to save the best circuit found.

perm_rounds:	Number of greedy searches on permuted equivalent targets.

block_rounds:	Number of AES block-cyclic prefix seed searches. Mainly useful for 32-bit AES MixColumn-like matrices.

random_perm_count:	Number of random permutations added to the structure-aware permutation set.

refine_depth_slack:	SAT refinement is applied to candidates with depth at most best_depth + refine_depth_slack.

rng_seed:	Optional fixed random seed for reproducible experiments.

