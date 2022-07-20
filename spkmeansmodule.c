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
    int i,j;
    PyObject *list = PyList_New(numberOfVectors);
    PyObject *vector;
    for (i = 0; i < numberOfVectors; i++){
        vector = PyList_New((int)length);
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
    return NULL;
}

/**
 * @brief Checks input validity 
 * 
 * @param operation goal code
 * @param list Matrix or data points
 * @param numberOfVectors number of rows in the matrix
 * @param lengthOfVectors length of a row in the matrix
 * @return int 0 if not valid
 */
int inputVadility(const char *operation, PyObject *list, int numberOfVectors, int lengthOfVectors){
    if (!strcmp(operation, "wam") || !strcmp(operation, "ddg") || !strcmp(operation, "lnorm")){
        if (!PyList_Check(list)){
            return 0;
        }
        if (numberOfVectors <= 0 || lengthOfVectors <= 0){
            return 0;
        }
        return 1;
    }
    if (!strcmp(operation, "jacobi")){
        return 1;
    }
    return 1;
}

/**
 * @brief Weighted Adjacency Matrix calculation 
 * 
 * @param self module
 * @param args arguments from python
 * @return PyObject* 
 */
PyObject* _wam(PyObject *self, PyObject *args){
    PyObject *list, *PyResult;
    double **matrix, **result;
    int numberOfVectors, length;
    if (!PyArg_ParseTuple(args, "Oii", &list, &numberOfVectors, &length)){
        PyErr_PrintEx(0);
        errorMsg(1);
        return NULL;
    }
    if (!inputVadility("wam", list, numberOfVectors, numberOfVectors)){
        errorMsg(1);
        return NULL;
    }
    matrix = convertPythonListToArray(list, numberOfVectors, length);
    if (!matrix){
        errorMsg(1);
        return NULL;
    }
    result = wam(matrix, numberOfVectors, length);
    PyResult = createPythonList(result, length, numberOfVectors);
    freeBlock(matrix);
    freeBlock(result);
    return PyResult;
}
PyObject* _ddg(PyObject *self, PyObject *args){
    return NULL;
}
PyObject* _lnorm(PyObject *self, PyObject *args){
    return NULL;
}
PyObject* _jacobi(PyObject *self, PyObject *args){
    return NULL;
}

static PyMethodDef spkmeansMethods[] = {
    {"kmeans", (PyCFunction)_kmeans, METH_VARARGS, PyDoc_STR("Calculates centroids for K-Means classification.\nParameters: centroids list\nVectors list\nLength of vectors\nNumber of vectors\nK\nEpsilon\nMax iterations\nReturn: Clusters array\n")},
    {"wam", (PyCFunction) _wam, METH_VARARGS, PyDoc_STR("Calculates the Weighted Adjacency Matrix.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: Weighted Adjacency Matrix\n")},
    {"ddg", (PyCFunction) _ddg, METH_VARARGS, PyDoc_STR("Calculates  Diagonal Degree Matrix.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: Diagonal Degree Matrix\n")},
    {"lnorm", (PyCFunction) _lnorm, METH_VARARGS, PyDoc_STR("Calculates the Normalized Graph Laplacian.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: Normalized Graph Laplacian\n")},
    {"jacobi", (PyCFunction) _jacobi, METH_VARARGS, PyDoc_STR("Calculates the Normalized Graph Laplacian.\nParameters:\n\tData Points matrix\n\tNumber of data points\n\tDimension of point\nReturn: A matrix. First row cotains the eigenvalues and others rows the eigenvectors\n")},
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
