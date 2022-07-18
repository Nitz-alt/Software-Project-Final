#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

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
PyObject* _kmeans(PyObject *self, PyObject *args){
    return NULL;
}
PyObject* _wam(PyObject *self, PyObject *args){
    return NULL;
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

static struct PyModuleDef kmeansspModule = {
    PyModuleDef_HEAD_INIT,
    "spkmeans",
    NULL,
    -1,
    spkmeansMethods
};


PyMODINIT_FUNC PyInit_spkmeans(void){
    PyObject *m;
    m = PyModule_Create(&kmeansspModule);
    if(!m){
        return NULL;
    }
    return m;
}
