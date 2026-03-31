#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
long long NUM_Points;
int Maxiter;
double dx, dy;

int main()
{
    Maxiter = 10;

    long long particle_counts[] =
    {100,10000,1000000,100000000,1000000000};

    int num_cases = 5;

    char method[] = "deferred";  

    for(int config=1; config<=3; config++)
    {
        
        if(config==1){
            NX=250; NY=100;
        }
        else if(config==2){
            NX=500; NY=200;
        }
        else{
            NX=1000; NY=400;
        }

        GRID_X = NX+1;
        GRID_Y = NY+1;

        dx = 1.0/NX;
        dy = 1.0/NY;

        char filename[100];
        sprintf(filename,
        "exp1_%s_grid%d_labpc.csv",
        method, config);

        FILE *csv = fopen(filename,"w");

        fprintf(csv,
        "config,NX,NY,particles,interp_time,mover_time,total_time,PPC,per_particle_time\n");

        printf("\n=== Running Grid %d (%dx%d) ===\n",config,NX,NY);

        for(int p=0;p<num_cases;p++)
        {
            NUM_Points = particle_counts[p];

            printf("Particles: %lld\n",NUM_Points);

            double *mesh =
            (double*)calloc(GRID_X*GRID_Y,sizeof(double));

            Points *points =
            (Points*)malloc(NUM_Points*sizeof(Points));

            initializepoints(points);

            double interp_time = 0;
            double mover_time = 0;

            for(int iter=0;iter<Maxiter;iter++)
            {
                double start,end;

                // Interpolation
                start = omp_get_wtime();
                interpolation(mesh,points);
                end = omp_get_wtime();

                interp_time += end-start;

                // Mover
                start = omp_get_wtime();

                // mover_serial_immediate(points,dx,dy);
                mover_serial_deferred(points,dx,dy);

                end = omp_get_wtime();

                mover_time += end-start;
            }

            double total_time = interp_time + mover_time;

            double PPC =
            (double)NUM_Points/(NX*NY);

            double per_particle_time =
            total_time/NUM_Points;

            fprintf(csv,
            "%d,%d,%d,%lld,%.12e,%.12e,%.12e,%.12e,%.12e\n",
            config,NX,NY,NUM_Points,
            interp_time,
            mover_time,
            total_time,
            PPC,
            per_particle_time);

            free(mesh);
            free(points);
        }

        fclose(csv);

        printf("Saved CSV: %s\n", filename);

        // Particle sample (per grid)

        NUM_Points = particle_counts[num_cases-1];

        Points *points_sample =
        (Points*)malloc(NUM_Points*sizeof(Points));

        initializepoints(points_sample);

        char sample_fname[64];
        sprintf(sample_fname, "particle_sample_config%d.csv", config);
        save_particles_sample(points_sample, 100000);

        // Rename the output file to the config-specific name
        rename("particle_sample.csv", sample_fname);
        printf("Saved particle sample: %s\n", sample_fname);

        free(points_sample);
    }

    printf("\nExperiment 1 completed successfully\n");

    return 0;
}