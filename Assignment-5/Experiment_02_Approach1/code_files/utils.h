#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

void interpolation(double *mesh_value, Points *points);
void save_mesh(double *mesh_value);

void mover_serial(Points *points, double deltaX, double deltaY);
void mover_parallel(Points *points, double deltaX, double deltaY,int num_threads);

void mover_serial_immediate(Points *points, double deltaX, double deltaY);
void mover_immediate_parallel(Points *points, double deltaX, double deltaY,int num_threads);

void mover_serial_deferred(Points *points, double deltaX, double deltaY);
void mover_deferred_parallel(Points * points,double deltaX,double deltaY, int num_threads);

void save_particles_sample(Points *points, long long sample_size);
#endif