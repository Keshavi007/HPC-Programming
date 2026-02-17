#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init.h"
#include "utils.h"

#define CLK CLOCK_MONOTONIC

int main()
{
    struct timespec start, end;

    int minProbSize = 1 << 2;
    int maxProbSize = 1 << 12;

    double **m1, **m2, **mt, **result;

    printf("ProblemSize, AlgoTime, MFLOPS\n");

    for (int N = minProbSize; N <= maxProbSize; N *= 2) {

        init_matrices(N, &m1, &m2, &result);

        /* Allocate transpose */
        mt = (double**)malloc(N * sizeof(double*));
        for (int i = 0; i < N; i++)
            mt[i] = (double*)malloc(N * sizeof(double));

        /* Transpose outside timing */
        transpose(m2, mt, N);

        clock_gettime(CLK, &start);
        transposed_matrix_multiplication(m1, mt, result, N);
        clock_gettime(CLK, &end);

        double time =
            (end.tv_sec - start.tv_sec) +
            (end.tv_nsec - start.tv_nsec) * 1e-9;

        double flops = 2.0 * N * N * N;
        double mflops = (flops / time) / 1e6;

        printf("%d, %.9lf, %.2lf\n", N, time, mflops);

        free_matrices(N, m1, m2, result);
        for (int i = 0; i < N; i++) free(mt[i]);
        free(mt);
    }

    return 0;
}
