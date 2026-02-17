#include "utils.h"

void matrix_multiplication(double** A, double** B, double** C, int N, int type)
{
    if (type == 1) {
        for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
        for (int k=0;k<N;k++)
            C[i][j] += A[i][k]*B[k][j];
    }
    else if (type == 2) {
        for (int i=0;i<N;i++)
        for (int k=0;k<N;k++)
        for (int j=0;j<N;j++)
            C[i][j] += A[i][k]*B[k][j];
    }
    else if (type == 3) {
        for (int j=0;j<N;j++)
        for (int i=0;i<N;i++)
        for (int k=0;k<N;k++)
            C[i][j] += A[i][k]*B[k][j];
    }
    else if (type == 4) {
        for (int j=0;j<N;j++)
        for (int k=0;k<N;k++)
        for (int i=0;i<N;i++)
            C[i][j] += A[i][k]*B[k][j];
    }
    else if (type == 5) {
        for (int k=0;k<N;k++)
        for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            C[i][j] += A[i][k]*B[k][j];
    }
    else if (type == 6) {
        for (int k=0;k<N;k++)
        for (int j=0;j<N;j++)
        for (int i=0;i<N;i++)
            C[i][j] += A[i][k]*B[k][j];
    }
}

void transpose(double** m, double** mt, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mt[j][i] = m[i][j];
}

void transposed_matrix_multiplication(double** m1, double** mt, double** result, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += m1[i][k] * mt[j][k];
            }
            result[i][j] = sum;
        }
    }
}

void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N)
{
    for (int ii=0; ii<N; ii+=B)
    for (int jj=0; jj<N; jj+=B)
    for (int kk=0; kk<N; kk+=B)

        for (int i=ii; i<ii+B && i<N; i++)
        for (int j=jj; j<jj+B && j<N; j++)
        for (int k=kk; k<kk+B && k<N; k++)
            result[i][j] += m1[i][k]*m2[k][j];
}
