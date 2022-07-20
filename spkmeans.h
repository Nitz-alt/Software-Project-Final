#ifndef SPKMEANS_H_
#define SPKMEANS_H_
double **kmeans(double ** centeroids, double ** vectors, int const length, int const numberOfVectors, int const K, double const EPSILON, int const MAX_ITER);
double **wam(double **matrix, size_t numberOfVectors, size_t length);
double ** ddg(double **matrix, size_t numberOfVectors, size_t length);
double **lnorm(double ** vectors, size_t numberOfVectors, size_t length);
double ** jacobi(double **vectors, size_t dim);
double **createBlockMatrix(size_t size, size_t numberOfRows, size_t numberOfColumns);
void freeBlock(double **ptr);
void errorMsg(int code);
void printMatrix(double **array, size_t numberOfRows, size_t numberOfColumns);
#endif