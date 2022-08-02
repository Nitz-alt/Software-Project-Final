#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"
#include <string.h>


/**
 * @brief Create a Python List object
 * 
 * @param vectors vecotrs array to turn list to
 * @param length length of each vector
 * @param numberOfVectors number of vectors
 * @return PyObject* 
 */
static PyObject* createPythonList(double ** vectors, int length, int numberOfVectors){
    int i,j, k;
    PyObject *list = PyList_New(numberOfVectors);
    PyObject *vector;
    for (i = 0; i < numberOfVectors; i++){
        vector = PyList_New((int)length);
        if (vector == NULL){
            for (k = 0; k < i; k++){
                Py_DecRef(PyList_GetItem(list, k));
            }
            return NULL;
        }
        for (j = 0; j < length; j++){
            PyList_SetItem(vector, j, PyFloat_FromDouble(vectors[i][j]));
        }
        PyList_SetItem(list, i, vector);
    }
    return list;
}

/**
 * @brief Converting a python list of floats to any array of doubles.
 * Assumes the list is a matrix and that each sublist is of same length as others.
 * 
 * @param list Python list
 * @param numberOfVectors number of vectors (rows) in the matrix
 * @param length Lenght of the vectors
 * @return double** 
 */
double **convertPythonListToArray(PyObject *list, int numberOfVectors, int length){
    PyObject *subList, *item;
    int i,j;
    double **vectors = createBlockMatrix(sizeof(double), numberOfVectors, length);
    if (vectors == NULL){
        return NULL;
    }
    for (i = 0; i < numberOfVectors; i++){
        /* Getting row of matrix */
        subList = PyList_GetItem(list, i);
        /* Error checking if the row is not a list */
        if (!PyList_Check(subList)){
            freeBlock(vectors);
            return NULL;
        }
        for (j = 0; j < length; j++){
            /* Getting item from row */
            item = PyList_GetItem(subList, j);
            /* Error checking if the value is indeed a float */
            if (!PyFloat_Check(item)){
                freeBlock(vectors);
                return NULL;
            }
            vectors[i][j] = PyFloat_AsDouble(item);
            /* If an error occured PyErr... returns a pointer to the error type (i.e. not equal NULL) and null succeeded
            Error checking for sucess */
            if (PyErr_Occurred()){
                freeBlock(vectors);
                return NULL;
            }
        }
    }
    return vectors;
}

PyObject* _kmeans(PyObject *self, PyObject *args){
    PyObject *pyListCenteroids, *pyListVectors, *pyResult;
    int numberOfVectors, lengthOfVectors, K;
    double **centeroids, **vectors, **result;
    int i = 1;
    /* Parsing arguments */
    if (!PyArg_ParseTuple(args, "OOiii", &pyListCenteroids, &pyListVectors, &numberOfVectors, &lengthOfVectors, &K)){
        errorMsg(1);
        return NULL;
    }
    centeroids = convertPythonListToArray(pyListCenteroids, K, lengthOfVectors);
    if (centeroids == NULL){
        return NULL;   
    }
    vectors = convertPythonListToArray(pyListVectors, numberOfVectors, lengthOfVectors);
    if (vectors == NULL){
        return NULL;
    }
    result = kmeans(centeroids, vectors, lengthOfVectors, numberOfVectors, K);
    if (result == NULL){
        freeBlock(centeroids);
        freeBlock(vectors);
        return NULL;
    }
    printMatrix(result, K, lengthOfVectors);
    pyResult = createPythonList(result, lengthOfVectors, K);
    freeBlock(centeroids);
    freeBlock(vectors);
    freeBlock(result);
    return pyResult;
}

/**
 * @brief Excecution routine for WAM, DDG and LNORM
 * 
 * @param args 
 * @param goal goal function from C library
 * @return PyObject* appropriate matrix according to goal.
 */
PyObject *dataPointsOperation(PyObject *args, double ** (*goal)(double **, int, int)){
    PyObject *list, *PyResult;
    double **matrix, **result;
    int numberOfVectors, lengthOfVectors;
    if (!PyArg_ParseTuple(args, "Oii", &list, &numberOfVectors, &lengthOfVectors)){
        PyErr_PrintEx(0);
        errorMsg(1);
        return NULL;
    }
    if (!PyList_Check(list)){
            errorMsg(1);
            return NULL;
    }
    if (numberOfVectors <= 0 || lengthOfVectors <= 0){
        errorMsg(1);
        return NULL;
    }
    matrix = convertPythonListToArray(list, numberOfVectors, lengthOfVectors);
    if (!matrix){
        errorMsg(1);
        return NULL;
    }
    result = (*goal)(matrix, numberOfVectors, lengthOfVectors);
    if (result == NULL){
        errorMsg(1);
        return NULL;
    }
    PyResult = createPythonList(result, numberOfVectors, numberOfVectors);
    freeBlock(matrix);
    freeBlock(result);
    return PyResult;
}



/**
 * @brief Weighted Adjacency Matrix calculation 
 * 
 * @param self module
 * @param args arguments from python
 * @return PyObject* 
 */
PyObject* _wam(PyObject *self, PyObject *args){
    return dataPointsOperation(args, &wam);
}
/**
 * @brief DDG matrix calculation 
 * 
 * @param self module
 * @param args arguments from python
 * @return PyObject* 
 */
PyObject* _ddg(PyObject *self, PyObject *args){
    return dataPointsOperation(args, &ddg);
}
/**
 * @brief lnorm matrix calculation 
 * 
 * @param self module
 * @param args arguments from python
 * @return PyObject* 
 */
PyObject* _lnorm(PyObject *self, PyObject *args){
    return dataPointsOperation(args, &lnorm);
}


/**
 * @brief Jacobi algorithm
 * 
 * @param self 
 * @param args 
 * @return PyObject* 
 */
PyObject* _jacobi(PyObject *self, PyObject *args){
    PyObject *list, *PyResult;
    double **matrix, **result;
    int dim;
    if (!PyArg_ParseTuple(args, "Oi", &list, &dim)){
        errorMsg(1);
        return NULL;
    }
    if (!PyList_Check(list)){
        errorMsg(1);
        return NULL;
    }
    if (dim <= 0){
        errorMsg(1);
        return NULL;
    }
    matrix = convertPythonListToArray(list, dim, dim);
    if (!matrix){
        errorMsg(1);
        return NULL;
    }
    result = jacobi(matrix, dim);
    PyResult = createPythonList(result, dim, dim + 1);
    freeBlock(matrix);
    freeBlock(result);
    return PyResult;
}

PyObject *_normalSpectralClustering(PyObject *self, PyObject *args){
    int K, numberOfVectors, lengthOfVectors;
    PyObject *list, *pyResult;
    double **matrix, **result;
    if (!PyArg_ParseTuple(args, "Oiii", &list, &numberOfVectors, &lengthOfVectors, &K)){
        errorMsg(1);
        return NULL;
    }
    if (!PyList_Check(list)){
            errorMsg(1);
            return NULL;
    }
    if (numberOfVectors <= 0 || lengthOfVectors <= 0){
        errorMsg(1);
        return NULL;
    }
    matrix = convertPythonListToArray(list, numberOfVectors, lengthOfVectors);
    if (!matrix){
        errorMsg(1);
        return NULL;
    }
    result = normalSpectralClustering(matrix, numberOfVectors, lengthOfVectors, &K);
    if (result == NULL){
        errorMsg(1);
        return NULL;
    }
    pyResult = createPythonList(result, K, numberOfVectors);
    freeBlock(matrix);
    freeBlock(result);
    return Py_BuildValue("(Oi)", pyResult, K);
}


static PyMethodDef spkmeansMethods[] = {
    {"kmeans", (PyCFunction)_kmeans, METH_VARARGS, PyDoc_STR("Calculates centroids for K-Means classification.\nParameters: centroids list\nVectors list\nLength of vectors\nNumber of vectors\nK\nReturn: Clusters array\n")},
    {"wam", (PyCFunction) _wam, METH_VARARGS, PyDoc_STR("Calculates the Weighted Adjacency Matrix.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: Weighted Adjacency Matrix\n")},
    {"ddg", (PyCFunction) _ddg, METH_VARARGS, PyDoc_STR("Calculates  Diagonal Degree Matrix.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: Diagonal Degree Matrix\n")},
    {"lnorm", (PyCFunction) _lnorm, METH_VARARGS, PyDoc_STR("Calculates the Normalized Graph Laplacian.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: Normalized Graph Laplacian\n")},
    {"jacobi", (PyCFunction) _jacobi, METH_VARARGS, PyDoc_STR("Calculates the Normalized Graph Laplacian.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: A matrix. First row cotains the eigenvalues and others rows the eigenvectors\n")},
    {"normalSpectralClustering", (PyCFunction) _normalSpectralClustering, METH_VARARGS, PyDoc_STR("T matrix of the Normalized Spectral Clustering method.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: A tuple containing The matrix and K\n")},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef spkmeansModule = {
    PyModuleDef_HEAD_INIT,
    "spkmeans",
    NULL,
    -1,
    spkmeansMethods
};


PyMODINIT_FUNC PyInit_spkmeans(void){
    PyObject *m;
    m = PyModule_Create(&spkmeansModule);
    if(!m){
        return NULL;
    }
    return m;
}
