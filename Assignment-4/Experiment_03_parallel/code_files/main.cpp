#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
long long NUM_Points;
int Maxiter;
double dx, dy;

int main() {

    // Fixed parameters
    NX = 1000;
    NY = 400;
    NUM_Points = 14000000LL;
    Maxiter = 10;

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    double *mesh_value = (double*) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points*) malloc(NUM_Points * sizeof(Points));

    // Initialize once
    initializepoints(points);

    printf("Iteration,Interp,Move,Total\n");

    for (int iter = 0; iter < Maxiter; iter++) {

        double start_interp = omp_get_wtime();
        interpolation(mesh_value, points);
        double end_interp = omp_get_wtime();

        double start_move = omp_get_wtime();
        mover_parallel(points, dx, dy);
        double end_move = omp_get_wtime();

        double interp_time = end_interp - start_interp;
        double move_time = end_move - start_move;
        double total_time = interp_time + move_time;

        printf("%d,%lf,%lf,%lf\n",
               iter+1, interp_time, move_time, total_time);
    }

    free(mesh_value);
    free(points);

    return 0;
}