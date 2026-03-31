#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {

    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    for (long long p = 0; p < NUM_Points; p++) {

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

// Assignment 4 - for comparison (without insertion and deletion)

// Serial Mover
void mover_serial(Points *points, double deltaX, double deltaY) {
    unsigned int seed = 1234 + (unsigned int)omp_get_thread_num();

    for (long long i = 0; i < NUM_Points; i++) {
        double new_x, new_y;
        do {
            double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaX - deltaX;
            double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaY - deltaY;
            new_x = points[i].x + rx;
            new_y = points[i].y + ry;
        } while (new_x < 0.0 || new_x > 1.0 ||
                 new_y < 0.0 || new_y > 1.0);

        points[i].x = new_x;
        points[i].y = new_y;
    }
}

// Parallel Mover
void mover_parallel(Points *points, double deltaX, double deltaY, int num_threads) {
#pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed = 1234 + (unsigned int)omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long i = 0; i < NUM_Points; i++) {
            double new_x, new_y;
            do {
                double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaX - deltaX;
                double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaY - deltaY;
                new_x = points[i].x + rx;
                new_y = points[i].y + ry;
            } while (new_x < 0.0 || new_x > 1.0 ||
                     new_y < 0.0 || new_y > 1.0);

            points[i].x = new_x;
            points[i].y = new_y;
        }
    }
}

// Assignment - 5

// Serial Mover Immediate Approach
void mover_serial_immediate(Points *points, double deltaX, double deltaY) {

    unsigned int seed = 1234 + omp_get_thread_num();

    for (long long i = 0; i < NUM_Points; i++) {

        double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

        double new_x = points[i].x + rx * deltaX;
        double new_y = points[i].y + ry * deltaY;

        if (new_x < 0.0 || new_x > 1.0 ||
            new_y < 0.0 || new_y > 1.0) {

            // delete and insert new particle
            points[i].x = (double) rand_r(&seed) / RAND_MAX;
            points[i].y = (double) rand_r(&seed) / RAND_MAX;

        }
        else {

            points[i].x = new_x;
            points[i].y = new_y;

        }
    }
}

// Parallel Mover Immediate Approach
void mover_immediate_parallel(Points *points, double deltaX, double deltaY,int num_threads) {

#pragma omp parallel num_threads(num_threads)
{
    unsigned int seed = 1234 + omp_get_thread_num();

#pragma omp for
    for (long long i = 0; i < NUM_Points; i++) {

        double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

        double new_x = points[i].x + rx * deltaX;
        double new_y = points[i].y + ry * deltaY;

        if (new_x < 0.0 || new_x > 1.0 ||
            new_y < 0.0 || new_y > 1.0) {

            points[i].x = (double) rand_r(&seed) / RAND_MAX;
            points[i].y = (double) rand_r(&seed) / RAND_MAX;

        }
        else {

            points[i].x = new_x;
            points[i].y = new_y;
        }
    }
}
}

// Serial Mover Deferred Approach
void mover_serial_deferred(Points *points, double deltaX, double deltaY) {
    unsigned int seed = 42;
    long long delete_count = 0;

    // 1 Move all particles
    // If particle exits domain, mark it by setting x = -1.0
    // (sentinel value outside valid [0,1] range)
    for (long long i = 0; i < NUM_Points; i++) {
        double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaX - deltaX;
        double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaY - deltaY;

        double new_x = points[i].x + rx;
        double new_y = points[i].y + ry;

        if (new_x < 0.0 || new_x > 1.0 ||
            new_y < 0.0 || new_y > 1.0) {
            // Mark as deleted using sentinel
            points[i].x = -1.0;
            points[i].y = -1.0;
            delete_count++;
        } else {
            points[i].x = new_x;
            points[i].y = new_y;
        }
    }

    // 2 Shift — push voids to the end Using two-pointer approach
    
    long long left  = 0;
    long long right = NUM_Points - 1;

    while (left < right) {
        // Advance left until we find a void slot
        while (left < right && points[left].x != -1.0) left++;
        // Retreat right until we find a valid particle
        while (left < right && points[right].x == -1.0) right--;

        if (left < right) {
            // Swap: bring valid particle to front, void goes to back
            Points temp    = points[left];
            points[left]   = points[right];
            points[right]  = temp;
            left++;
            right--;
        }
    }

    // 3: Batch insert into tail void slots
    // The last delete_count slots are now the void slots
    for (long long k = NUM_Points - delete_count; k < NUM_Points; k++) {
        points[k].x = (double) rand_r(&seed) / RAND_MAX;
        points[k].y = (double) rand_r(&seed) / RAND_MAX;
    }
}

// Parallel Mover Deferred Approach
void mover_deferred_parallel(Points *points, double deltaX, double deltaY, int num_threads) {

    // 1 Parallel move & mark
    long long delete_count = 0;

#pragma omp parallel num_threads(num_threads) reduction(+:delete_count)
    {
        unsigned int seed = 1234 + (unsigned int)omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long i = 0; i < NUM_Points; i++) {
            double rx = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaX - deltaX;
            double ry = ((double) rand_r(&seed) / RAND_MAX) * 2.0 * deltaY - deltaY;

            double new_x = points[i].x + rx;
            double new_y = points[i].y + ry;

            if (new_x < 0.0 || new_x > 1.0 ||
                new_y < 0.0 || new_y > 1.0) {
                // Mark as deleted using sentinel
                points[i].x = -1.0;
                points[i].y = -1.0;
                delete_count++;   // reduction handles thread safety
            } else {
                points[i].x = new_x;
                points[i].y = new_y;
            }
        }
    } // end parallel 1

    // 2: Shift — push voids to end (serial)
    long long left  = 0;
    long long right = NUM_Points - 1;

    while (left < right) {
        while (left < right && points[left].x  != -1.0) left++;
        while (left < right && points[right].x == -1.0) right--;

        if (left < right) {
            Points temp   = points[left];
            points[left]  = points[right];
            points[right] = temp;
            left++;
            right--;
        }
    }

    // 3 - Parallel re-insertion into tail void slots
#pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed = 5678 + (unsigned int)omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long k = NUM_Points - delete_count; k < NUM_Points; k++) {
            points[k].x = (double) rand_r(&seed) / RAND_MAX;
            points[k].y = (double) rand_r(&seed) / RAND_MAX;
        }
    } // end parallel 3
}

// Particles Distribution in 1x1 Grid x and y coordinates
void save_particles_sample(Points *points, long long sample_size)
{
    FILE *fp = fopen("particle_sample.csv","w");

    if(fp==NULL){
        printf("Error creating particle_sample.csv\n");
        return;
    }

    if(sample_size > NUM_Points)
        sample_size = NUM_Points;

    for(long long i=0;i<sample_size;i++)
    {
        fprintf(fp,"%lf,%lf\n",points[i].x,points[i].y);
    }

    fclose(fp);
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