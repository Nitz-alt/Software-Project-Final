#ifndef SPKMEANS_H_
#define SPKMEANS_H_


double **kmeans(double ** , double ** , int const , int const , int const , double const , int const );
double **wam(double **, size_t , size_t);
double ** ddg(double **, size_t , size_t);
double **lnorm(double ** , size_t , size_t);
double ** jacobi(double **, size_t);
#endif