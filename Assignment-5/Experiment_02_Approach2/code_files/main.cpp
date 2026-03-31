#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
long long NUM_Points;
int Maxiter;
double dx, dy;

static const int grid_nx[] = {250, 500, 1000};
static const int grid_ny[] = {100, 200, 400};
static const int num_grids = 3;

int main() {
    printf("\n\n========================================\n");
    printf("EXPERIMENT 02 - Immediate Only\n");
    printf("========================================\n");

    NUM_Points = 14000000LL;
    Maxiter = 10;

    static const int thread_counts[] = {1, 2, 4, 8, 16};
    static const int num_thread_sets = 5;

    for (int g = 0; g < num_grids; g++) {
        NX = grid_nx[g];
        NY = grid_ny[g];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        char fname[64];
        sprintf(fname, "exp2_immediate_grid%d.csv", g + 1);

        FILE *fp = fopen(fname, "w");
        if (!fp) {
            fprintf(stderr, "Cannot open %s\n", fname);
            return 1;
        }

        fprintf(fp, "Threads,Iter,Interp,Mover,Total\n");
        printf("\n=== Grid %d: NX=%d, NY=%d | N=%lld ===\n", g + 1, NX, NY, NUM_Points);

        double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *)malloc(NUM_Points * sizeof(Points));
        Points *points_init = (Points *)malloc(NUM_Points * sizeof(Points));

        if (!mesh_value || !points || !points_init) {
            fprintf(stderr, "malloc failed in immediate run\n");
            return 1;
        }

        srand(12345);
        initializepoints(points_init);

        for (int tc = 0; tc < num_thread_sets; tc++) {
            int nthreads = thread_counts[tc];
            printf("  Threads = %d\n", nthreads);

            memcpy(points, points_init, NUM_Points * sizeof(Points));

            for (int iter = 0; iter < Maxiter; iter++) {
                memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

                double t0 = omp_get_wtime();
                interpolation(mesh_value, points);
                double t1 = omp_get_wtime();

                if (nthreads == 1)
                    mover_serial_immediate(points, dx, dy);
                else
                    mover_immediate_parallel(points, dx, dy, nthreads);

                double t2 = omp_get_wtime();

                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n",
                        nthreads, iter + 1, t1 - t0, t2 - t1, t2 - t0);

                printf("    T=%2d | Iter %2d | Imm(%.4f)\n",
                       nthreads, iter + 1, t2 - t1);
            }
        }

        free(mesh_value);
        free(points);
        free(points_init);
        fclose(fp);

        printf("  Written: %s\n", fname);
    }

    printf("\nImmediate experiment complete.\n");
    return 0;
}