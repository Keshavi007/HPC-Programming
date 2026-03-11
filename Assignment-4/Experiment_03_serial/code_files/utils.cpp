#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {

    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    for (int p = 0; p < NUM_Points; p++) {

        double x = points[p].x;
        double y = points[p].y;

        // Determine cell index
        int i = (int)(x / dx);
        int j = (int)(y / dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        double x0 = i * dx;
        double y0 = j * dy;

        // Local weights
        double wx = (x - x0) / dx;
        double wy = (y - y0) / dy;

        int index = j * GRID_X + i;

        // Distribute weight to 4 surrounding grid nodes
        mesh_value[index]                 += (1 - wx) * (1 - wy);
        mesh_value[index + 1]             += wx * (1 - wy);
        mesh_value[index + GRID_X]        += (1 - wx) * wy;
        mesh_value[index + GRID_X + 1]    += wx * wy;
    }
}

// Mover (Serial)
unsigned int seed = 1234 + omp_get_thread_num();
void mover_serial(Points *points, double deltaX, double deltaY) {

    for (long long i = 0; i < NUM_Points; i++) {

        double new_x, new_y;

        do {
            double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
            double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

            new_x = points[i].x + rx * deltaX;
            new_y = points[i].y + ry * deltaY;

        } while (new_x < 0.0 || new_x > 1.0 ||
                 new_y < 0.0 || new_y > 1.0);

        points[i].x = new_x;
        points[i].y = new_y;
    }
}

// Mover (Parallel - 4 threads)
void mover_parallel(Points *points, double deltaX, double deltaY) {

#pragma omp parallel num_threads(4)
    {
        unsigned int seed = 1234 + omp_get_thread_num();

#pragma omp for
        for (long long i = 0; i < NUM_Points; i++) {

            double new_x, new_y;

            do {
                double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
                double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

                new_x = points[i].x + rx * deltaX;
                new_y = points[i].y + ry * deltaY;

            } while (new_x < 0.0 || new_x > 1.0 ||
                     new_y < 0.0 || new_y > 1.0);

            points[i].x = new_x;
            points[i].y = new_y;
        }
    }
}
// Write mesh to file
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}