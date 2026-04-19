#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include <omp.h>


double min_val, max_val;


void interpolation(double *mesh_value, Points *points) {

    if (omp_get_max_threads() <= 1) {
        serial_interpolation(mesh_value, points);
        return;
    }

    memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel
    {
        double *local_mesh = (double *) calloc(GRID_X * GRID_Y, sizeof(double));

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {
            
            if (points[p].is_void) continue;
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

void normalization(double *mesh_value) {

    if (omp_get_max_threads() <= 1) {
        serial_normalization(mesh_value);
        return;
    }

    min_val = mesh_value[0];
    max_val = mesh_value[0];

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh_value[i] < min_val) min_val = mesh_value[i];
        if (mesh_value[i] > max_val) max_val = mesh_value[i];
    }

    double range = max_val - min_val;

    if (range == 0.0) return;

    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }
}



void mover(double *mesh_value, Points *points) {

    if (omp_get_max_threads() <= 1) {
        serial_mover(mesh_value, points);
        return;
    }

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++) {

        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX ) i = NX - 1;
        if (j >= NY ) j = NY - 1;

        double Xi = i * dx;
        double Yj = j * dy;

        double lx = x - Xi;
        double ly = y - Yj;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx       * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx        * ly;

        int base = j * GRID_X + i;

        double Fi =
            w00 * mesh_value[base] +
            w10 * mesh_value[base + 1] +
            w01 * mesh_value[base + GRID_X] +
            w11 * mesh_value[base + GRID_X + 1];

        points[p].x += Fi * dx;
        points[p].y += Fi * dy;

        if (points[p].x < 0.0 || points[p].x > 1.0 ||
            points[p].y < 0.0 || points[p].y > 1.0) {
            points[p].is_void = true;
        }
    }
}


void denormalization(double *mesh_value) {

    if (omp_get_max_threads() <= 1) {
        serial_denormalization(mesh_value);
        return;
    }

    double range = max_val - min_val;

    if (range == 0.0) return;

    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = ((mesh_value[i] + 1.0) * 0.5) * range + min_val;
    }
}



// count particles that went beyond the domain
long long int void_count(Points *points) {

    long long int voids = 0;
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }
    return voids;
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

// Serial implementations
void serial_interpolation(double *mesh_value, Points *points) {

    memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int p = 0; p < NUM_Points; p++) {

        if (points[p].is_void) continue;

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

// Normalization: scale grid to [-1, 1]
void serial_normalization(double *mesh_value) {

    min_val = mesh_value[0];
    max_val = mesh_value[0];

    // find min/max
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh_value[i] < min_val) min_val = mesh_value[i];
        if (mesh_value[i] > max_val) max_val = mesh_value[i];
    }

    double range = max_val - min_val;

    if (range == 0.0) return;

    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }
}

// Reverse interpolation (Mover)
void serial_mover(double *mesh_value, Points *points) {

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int p = 0; p < NUM_Points; p++) {

        if (points[p].is_void) continue;

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

        double Fi =
            w00 * mesh_value[base] +
            w10 * mesh_value[base + 1] +
            w01 * mesh_value[base + GRID_X] +
            w11 * mesh_value[base + GRID_X + 1];

        // update positions
        points[p].x += Fi * dx;
        points[p].y += Fi * dy;

        // mark void particles
        if (points[p].x < 0.0 || points[p].x > 1.0 ||
            points[p].y < 0.0 || points[p].y > 1.0) {
            points[p].is_void = true;
        }
    }
}


// Denormalization
void serial_denormalization(double *mesh_value) {

    double range = max_val - min_val;

    if (range == 0.0) return;

    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = ((mesh_value[i] + 1.0) * 0.5) * range + min_val;
    }
}


