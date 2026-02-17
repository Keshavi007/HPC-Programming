#ifndef UTILS_H
#define UTILS_H

void matrix_multiplication(double** A, double** B, double** C, int N, int type);

void transpose(double** m, double** mt, int N);
void transposed_matrix_multiplication(double** m1, double** mt, double** result, int N);

void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N);

#endif
