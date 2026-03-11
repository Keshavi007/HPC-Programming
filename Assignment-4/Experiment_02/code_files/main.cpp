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

    Maxiter = 10;

  
    NUM_Points = 100000000LL;

    // Loop over 3 configurations
    for (int config = 1; config <= 3; config++) {

        if (config == 1) {
            NX = 250;  NY = 100;
        }
        else if (config == 2) {
            NX = 500;  NY = 200;
        }
        else {
            NX = 1000; NY = 400;
        }

        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh_value = (double*) calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points*) malloc(NUM_Points * sizeof(Points));

        // Initialize particles 
        initializepoints(points);

        double total_interp_time = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {

            double start = omp_get_wtime();
            interpolation(mesh_value, points);
            double end = omp_get_wtime();

            total_interp_time += (end - start);
        }

        printf("%d,%lf\n", config, total_interp_time);

        free(mesh_value);
        free(points);
    }

    return 0;
}