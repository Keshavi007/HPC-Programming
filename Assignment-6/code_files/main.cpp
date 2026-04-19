#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

static inline double wtime() { return omp_get_wtime(); }

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // ---- Open input file ----
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        return 1;
    }

    // ---- Read parameters ----
    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // ---- Allocate memory ----
    double *mesh_serial   = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    double *mesh_parallel = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points        = (Points *)calloc(NUM_Points, sizeof(Points));

    // ---- Preload all data ----
    double *all_x = (double *)malloc((size_t)Maxiter * NUM_Points * sizeof(double));
    double *all_y = (double *)malloc((size_t)Maxiter * NUM_Points * sizeof(double));

    for (int iter = 0; iter < Maxiter; iter++) {
        read_points(file, points);
        for (int p = 0; p < NUM_Points; p++) {
            all_x[iter * NUM_Points + p] = points[p].x;
            all_y[iter * NUM_Points + p] = points[p].y;
        }
    }
    fclose(file);

    // ---- SERIAL baseline ----
    double total_serial = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {
        for (int p = 0; p < NUM_Points; p++) {
            points[p].x = all_x[iter * NUM_Points + p];
            points[p].y = all_y[iter * NUM_Points + p];
        }

        memset(mesh_serial, 0, GRID_X * GRID_Y * sizeof(double));

        double t0 = wtime();
        interpolation(mesh_serial, points);
        total_serial += wtime() - t0;
    }

    double avg_serial = total_serial / Maxiter;

    // ---- Create CSV filename per config ----
    char filename[256];
    sprintf(filename, "results_NX%d_NY%d_P%d_labpc.csv", NX, NY, NUM_Points);

    FILE *csv = fopen(filename, "w");
    if (!csv) {
        printf("Error creating CSV file\n");
        return 1;
    }

    fprintf(csv, "NX,NY,Points,Iterations,Threads,"
                     "Avg_Time_Serial(s),Avg_Time_Parallel(s),Speedup\n");

    // ---- Thread loop ----
    int thread_list[] = {2, 4, 8, 16};
    int num_cases = 4;

    for (int t = 0; t < num_cases; t++) {

        int num_threads = thread_list[t];
        omp_set_num_threads(num_threads);

        double total_parallel = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {

            for (int p = 0; p < NUM_Points; p++) {
                points[p].x = all_x[iter * NUM_Points + p];
                points[p].y = all_y[iter * NUM_Points + p];
            }

            double t0 = wtime();
            interpolation_parallel(mesh_parallel, points);
            total_parallel += wtime() - t0;
        }

        double avg_parallel = total_parallel / Maxiter;
        double speedup = avg_serial / avg_parallel;

        printf("Threads: %d | Time: %.6f | Speedup: %.2f\n",
               num_threads, avg_parallel, speedup);

        fprintf(csv, "%d,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",
            NX, NY, NUM_Points, Maxiter, num_threads,
            avg_serial, avg_parallel, speedup);
    }

    fclose(csv);

    // Save final mesh
    save_mesh(mesh_parallel);

    // Cleanup
    free(mesh_serial);
    free(mesh_parallel);
    free(points);
    free(all_x);
    free(all_y);

    return 0;
}