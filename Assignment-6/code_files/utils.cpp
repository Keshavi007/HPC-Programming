#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Serial interpolation 
void interpolation(double *mesh_value, Points *points) {
    
    memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int p = 0; p < NUM_Points; p++) {
        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        double Xi = i * dx;
        double Yj = j * dy;

        double lx = x - Xi;
        double ly = y - Yj;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx       * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx        * ly;

        int base = j * GRID_X + i;

        mesh_value[base] += w00;
        mesh_value[base + 1] += w10;
        mesh_value[base + GRID_X] += w01;
        mesh_value[base + GRID_X + 1] += w11;
    }
}

void interpolation_parallel(double *mesh_value, Points *points) {

    memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel
    {
        double *local_mesh = (double *) calloc(GRID_X * GRID_Y, sizeof(double));

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {

            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;

            double Xi = i * dx;
            double Yj = j * dy;

            double lx = x - Xi;
            double ly = y - Yj;

            double w00 = (dx - lx) * (dy - ly);
            double w10 = lx       * (dy - ly);
            double w01 = (dx - lx) * ly;
            double w11 = lx        * ly;

            int base = j * GRID_X + i;

            local_mesh[base] += w00;
            local_mesh[base + 1] += w10;
            local_mesh[base + GRID_X] += w01;
            local_mesh[base + GRID_X + 1] += w11;
        }

        // Combine results
        #pragma omp critical
        {
            for (int i = 0; i < GRID_X * GRID_Y; i++) {
                mesh_value[i] += local_mesh[i];
            }
        }

        free(local_mesh);
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