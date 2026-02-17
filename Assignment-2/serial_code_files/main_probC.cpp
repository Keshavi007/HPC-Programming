#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init.h"
#include "utils.h"

#define CLK CLOCK_MONOTONIC

int main() {

    struct timespec start_alg, end_alg;

    int minProbSize = 1 << 2;   // 2^2 = 4
    int maxProbSize = 1 << 12;  // 2^12 = 4096

    double** m1;
    double** m2;
    double** result;

    printf("ProblemSize, BlockSize, AlgoTime, MFLOPS\n");

    for (int Np = minProbSize; Np <= maxProbSize; Np *= 2) {

        init_matrices(Np, &m1, &m2, &result);

        // Choose block size (safe default)
        int Bsize = (Np >= 32 ? 32 : Np);

        // Reset result matrix
        for (int i = 0; i < Np; i++)
            for (int j = 0; j < Np; j++)
                result[i][j] = 0.0;

        clock_gettime(CLK, &start_alg);

        block_matrix_multiplication(m1, m2, result, Bsize, Np);

        clock_gettime(CLK, &end_alg);

        double alg_time =
            (end_alg.tv_sec - start_alg.tv_sec) +
            (end_alg.tv_nsec - start_alg.tv_nsec) * 1e-9;

        double flops = 2.0 * Np * Np * Np;
        double mflops = (flops / alg_time) / 1e6;

        printf("%d, %d, %.9lf, %.2lf\n",
               Np, Bsize, alg_time, mflops);

        fflush(stdout);

        free_matrices(Np, m1, m2, result);
    }

    return 0;
}
