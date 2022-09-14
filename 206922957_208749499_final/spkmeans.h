#ifndef SPKMEANS_H_
#define SPKMEANS_H_
double **kmeans(double **centeroids, double **vectors, int const length, int const numberOfVectors, int const K);
double **wam(double **matrix, int numberOfVectors, int length);
double ** ddg(double **matrix, int numberOfVectors, int length);
double **lnorm(double ** vectors, int numberOfVectors, int length);
double ** jacobi(double **vectors, int dim);
double **createBlockMatrix(size_t size, int numberOfRows, int numberOfColumns);
void freeBlock(double **ptr);
void errorMsg(int code);
void printMatrix(double **array, int numberOfRows, int numberOfColumns);
double **normalSpectralClustering(double **vectors, int numberOfVectors, int length, int *numberOfClusters);
#endif
