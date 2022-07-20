#ifndef SPKMEANS_H_
#define SPKMEANS_H_
double **kmeans(double ** centeroids, double ** vectors, int const length, int const numberOfVectors, int const K, double const EPSILON, int const MAX_ITER);
double **wam(double **matrix, int numberOfVectors, int length);
double ** ddg(double **matrix, int numberOfVectors, int length);
double **lnorm(double ** vectors, int numberOfVectors, int length);
double ** jacobi(double **vectors, int dim);
double **createBlockMatrix(int size, int numberOfRows, int numberOfColumns);
void freeBlock(double **ptr);
void errorMsg(int code);
void printMatrix(double **array, int numberOfRows, int numberOfColumns);
#endif
