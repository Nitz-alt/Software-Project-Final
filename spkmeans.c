#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "spkmeans.h"


#define CONVERGENCE 1E-5
#define EPSILON 0
#define MAX_ITER 300

void copyMatrix(double **destination, double **origin, int rows, int cols);
/*
    Paramters:
        x - Vector 1
        y - Vector 2
    Return:
    double - Eldian distance from x to y
*/
double eucleadDist(double x[], double y[], int length){
    int i;
    double dist = 0;
    for (i=0;i < length; i++){
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return dist;
}

/*
    Parameters:
        dst - Destination array
        origin - Array to copy values from
        size - number of elements in the arrays
*/
void copyToArray(double dst[], double origin[], int size){
    int i;
    for (i = 0; i < size; i++){
        dst[i] = origin[i];
    }
}

/*
    Paramters:
        ar - array to print
        size - size of the array
    Return:
        void
*/
void printAr(double *ar, int size){
    int i = 0;
    char suffix = ',';
    for(;i < size; i++){
        if (i == size-1){
            suffix = '\n';
        }
        printf("%.4f%c", ar[i], suffix);
    }
}

/*
    Prints a matrix
    Paramters:
        array - matrix to print
        numberOfRows - number of rows
        numberOfColumns - number of columns
    Return:
*/
void printMatrix(double **array, int numberOfRows, int numberOfColumns){
    int i;
    for (i=0; i < numberOfRows; i++){
        printAr(array[i], numberOfColumns);
    }
}

/*
    Paramters:
        number - Number represented with a string
    Return:
        Integer of the number
*/
int convertToInt(char number[]){
    char c;
    int i=0;
    while ((c = number[i++]) != '\0'){
        if (c < '0' || c > '9' || c == '.'){
            return 0;
        }
    }
    return atoi(number);
}

void errorMsg(int code){
    /* Free memory */
    if (code == 0){
        printf("Invalid input");
    }
    if (code == 1)
        printf("An Error Has Occurred");
    exit(1);
}

/*
    Paramters:
        filename - name of a file
    Return:
        Checks if the file is a .txt or .csv file
*/
int checkTextFormat(char *fileName){
    int i = 0; 
    char c;
    while ((c = fileName[i++]) != '\0');
    i--;

    if (i < 4) return 0;
    fileName = fileName + (i-4);
    if (!strcmp(fileName, ".txt") || !strcmp(fileName, ".csv")) return 1;
    return 0;
}

/* 
    Parameters:
        coordinate - Specific coordinate to check to which cluster it needs to be assigned
        centeroids = Centeroids list
        K - number of Centeroids
        Length - Length of the vectors
    Return:
        int - Returns the index of the cluster the the coordinates need to be assigned to
*/
int ArgMin(double coordiante[], double * centeroids[], int const K, int length){
    double min_dist;
    double tmp_min_dist;
    int min_dist_index = 0;
    int i;
    /* Sets min dist to first distance */
    min_dist = eucleadDist(coordiante, centeroids[0], length);
    for (i=0; i< K; i++){
        tmp_min_dist = eucleadDist(coordiante, centeroids[i], length);
        if (tmp_min_dist < min_dist){
            min_dist = tmp_min_dist;
            min_dist_index = i;
        }
    }
    return min_dist_index;
}

/*
    Paramters:
        centeroids - Centrdoids array
        clusters - Clusters array
        clusterSizes - Array containing sizes of the clusters. clusterSizes[i] = len(clusters[i])
        length - Lenght of the vectors
        K - number of centeroids
    Return:
        double - maximum delta of the centeroids
*/
double UpdateCenteroids(double **centeroids, double ** clusters[], int clusterSizes[], int const length, int const K){
    double max_epsilon = 0;
    int i;
    double sum=0;
    double dist;
    double **old_centeroids = (double **) malloc(K * sizeof(double *));
    for (i=0; i < K; i++){
        old_centeroids[i] = malloc(sizeof(double) * length);
        copyToArray(old_centeroids[i], centeroids[i], length);
    }
    for (i=0; i < K; i++){
        /* For each cluster */
        int j;
        for (j = 0; j < length; j++){
            /* Cordinate in vector */
            int m;
            for (m = 0; m < clusterSizes[i]; m++){
                /* For each vector in the cluster*/
                sum += clusters[i][m][j];
            }
            sum = sum / (double) clusterSizes[i];
            centeroids[i][j] = sum;
            sum = 0;
        }
    }

    for (i = 0; i < K; i++){
        dist = eucleadDist(old_centeroids[i], centeroids[i], length);
        max_epsilon = dist > max_epsilon ? dist : max_epsilon;
        free(old_centeroids[i]);
    }
    free(old_centeroids);
    return sqrt(max_epsilon);
}


/*  
    Frees an of all pointers in the array and the array
    Paramters:
        ptr - an array of pointers to memory.
    Return:
*/
void freeArray(double ***array, int len){
    int i;
    for (i = 0; i < len; i++){
        free(array[i]);
    }
    free(array);
}



/*  
    kmeans algortihm.
    Paramters:
        centeroids - initial centroids array.
        vectors - All vectors to classify
        length - length of the vectors
        numberOfVectors - number of the vectors
        K - number of clusters
    Return:
        returns the final centroids.
*/
double **kmeans(double **initialCenteroids, double **vectors, int const length, int const numberOfVectors, int const K){
    int iter_num = 1;
    double max_epsilon = EPSILON + 1;
    int clusterIndex;
    int i,j,t, memoryListIndex = 0;
    int *clusterSizes;
    double ***clusters;
    double ***memory;
    double **centeroids;
    double **result;
    /**
     * Memory size calculation (only of double **)
     * 1 - clusters list
     * K - actual clusters
     * 1 - centeroids row pointers
     * 1 - centeroids block pointer
     */
    printf("Entered kmeans\nlength=%d\nnumberOfVectors=%d\nK=%d\n", length, numberOfVectors, K);
    memory = (double ***) malloc(sizeof(double **) * K + 3);
    if (memory == NULL) return NULL;
    centeroids = (double **) createBlockMatrix(sizeof(double), K, length);
    if (centeroids == NULL){
        free(memory);
        return NULL;
    }
    memory[memoryListIndex++] = centeroids;
    memory[memoryListIndex++] = (double **) *centeroids;
    printf("Reached 1\n");
    /* Copying centeroids */
    copyMatrix(centeroids, initialCenteroids, K, length);
    /*Allocate cluster*/
    clusters = (double ***) malloc(sizeof(double **) * K);
    if (clusters == NULL){
        freeArray(memory, memoryListIndex);
        return NULL;
    }
    memory[memoryListIndex++] = (double **)clusters;
    printf("Reached 2\n");
    for (i = 0; i < K; i++){
        clusters[i] = (double **) malloc(numberOfVectors *  sizeof(double *));
        if (clusters[i] == NULL){
            freeArray(memory, memoryListIndex);
            return NULL;
        }
        memory[memoryListIndex++] = clusters[i];
    }
    printf("Reached 3\n");
    /*Resetting cluster sizes to 0*/
    clusterSizes = (int *) calloc(K, sizeof(int));
    if (clusterSizes == NULL){
        freeArray(memory, memoryListIndex);
        return NULL;
    }
    printf("Reached 4\n");
    while (iter_num < MAX_ITER && max_epsilon > EPSILON){
        /* Assigning vectors to clusters*/
        for(i = 0; i < numberOfVectors; i++){
            clusterIndex = ArgMin(vectors[i], centeroids, K, length);
            clusters[clusterIndex][clusterSizes[clusterIndex]] = vectors[i];
            clusterSizes[clusterIndex]++;
        }
        printf("Reached -1\n");
        max_epsilon = UpdateCenteroids(centeroids, clusters, clusterSizes, length, K);
        printf("Reached -2\n");
        iter_num++;
        /* Resetting clusters*/
        for (j = 0; j < K; j++){
            for (t = 0; t < clusterSizes[j]; t++){
                clusters[j][t] = NULL;
            }
            clusterSizes[j] = 0;
        }
    }
    printf("Reached 5\n");
    result = createBlockMatrix(sizeof(double), K, length);
    printf("Reached 6\n");
    copyMatrix(result, centeroids, K, length);
    printf("Reached 7\n");
    freeArray(memory, memoryListIndex);
    free(clusterSizes);
    return result;
}

/*
    Allocates a matrix of dimension numberOfRows X numberOfColumns.
    The memory is allocate as a block.
    Paramters:
        numberOfRows - number of rows to in the matrix to allocate
        numberOfColumns - number of columns in the matrix to allocate
    Return:
        returns the pointer to the memory that was allocated
        
*/
double **createBlockMatrix(size_t size, int numberOfRows, int numberOfColumns){
    int i;
    double *block, **matrix;
    block = (double *) malloc(size * numberOfRows * numberOfColumns);
    if (block == NULL) {return NULL;}
    matrix = (double **) malloc(size * numberOfRows);
    if (matrix == NULL) {return NULL;}
    for (i = 0; i < numberOfRows; i++){
        matrix[i] = block + i * numberOfColumns;
    }
    return matrix;
}

/*  
    Frees a memory allocated as a block
    Paramters:
        ptr - Pointer to the memory allocated as a block.
    Return:
*/
void freeBlock(double **ptr){
    free(*ptr);
    free(ptr);
}

/*  
    Frees an of array of memory blocks
    Paramters:
        ptr - an array of pointers to memory allocated as a blocks.
    Return:
*/
void freeArrayOfBlocks(double ***ptr, int len){
    int i;
    for (i = 0; i < len; i++){
        freeBlock(ptr[i]);
    }
    free(ptr);
}



/*
    Paramters:
        input - File to parse vectors from
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        A matrix of the vectors parsed from the file
*/
double **parseMatrix(FILE *input, int numberOfVectors, int length){
    int i,j;
    double *vector;
    double ** matrix = createBlockMatrix(sizeof(double), numberOfVectors, length);
    if (matrix == NULL) return NULL;

    for (i = 0; i < numberOfVectors; i++){
        vector = matrix[i];
        for (j = 0; j < length; j++){
            if (j != length - 1){
                if (fscanf(input, "%lf,", (vector + j)) != 1) errorMsg(1);
            }
            else{
                if (fscanf(input, "%lf\n", (vector + j)) != 1) errorMsg(1);
            }
        }
    }
    return matrix;
}


/*
    Paramters:
        Matrix - matrix of vectors to calculated the weighted matrix on
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        Weighted Adjacency Matrix
*/
double **wam(double **matrix, int numberOfVectors, int length){
    int i,j;
    double wij;
    double **weightedMatrix = (double **) createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    if (weightedMatrix == NULL) return NULL;
    for (i = 0; i < numberOfVectors; i++){
        weightedMatrix[i][i] = 0;
        for (j = i + 1; j < numberOfVectors; j++){
            wij = exp(-1 * sqrt(eucleadDist(matrix[i], matrix[j], length)) * 0.5);
            weightedMatrix[i][j] = wij;
            weightedMatrix[j][i] = wij;
        }
    }
    return weightedMatrix;
}

/* 
    Paramters:
        matrix - matrix of vectors to calculated the weighted matrix on
        length - length of each vector in the matrix
        row - row to sum
    Return:
        Sum of row in matrix
        
*/
double sumOfRow(double **matrix, int length, int row){
    int i;
    double sum = 0;
    for (i = 0; i < length; i++){
        sum += matrix[row][i];
    }
    return sum;
}

double ff(double v){
    return pow(v, -0.5);
}

/*
    applying a function to each entry in the matrix.
    Performed in place.
    Paramters:
        dst_matrix - matrix to apply function on
        numberOfRows - number of rows in matrix
        numberOfColumns - number of columns in matrix
        f - function to apply
    Return:
        void => the function applys in place
*/
void funcOnMatrix(double **dst_matrix , int numberOfRows, int numberOfColumns, double (*f)(double)){
    int i,j;
    for (i = 0 ; i < numberOfRows; i++){
        for (j = 0; j < numberOfColumns; j++){
            dst_matrix[i][j] = (*f)(dst_matrix[i][j]);
        }
    }
}

/* 
    Paramters:
        matrix - matrix of vectors to calculate its diagonal degree matrix
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        Diagonal Degree Matrix of matrix
        
*/
double ** ddg(double **matrix, int numberOfVectors, int length){
    int i, j;
    double **wamMatrix;
    double **diag_matrix = (double **) createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    if (diag_matrix == NULL) return NULL;
    wamMatrix = wam(matrix, numberOfVectors, length);
    if (wamMatrix == NULL){
        freeBlock(diag_matrix);
        return NULL;
    }
    
    for (i = 0; i < numberOfVectors; i++){
        for (j = 0; j < numberOfVectors; j++){
            if (j != i) diag_matrix[i][j] = 0;
            else diag_matrix[i][i] = sumOfRow(wamMatrix, numberOfVectors, i);
        }
    }
    return diag_matrix;
}


/*
    Returns the eye matrix of (size X size) dimensions 
    Paramters:
        size - dimension of matrix
    Return:
*/
double **eye(int size){
    int i,j;
    double **matrix = (double**) createBlockMatrix(sizeof(double), size, size);
    if (matrix == NULL) return NULL;
    for (i = 0 ; i < size; i++){
        for (j = 0; j < size; j++){
            if (i != j) matrix[i][j] = 0;
            else matrix[i][j] = 1;
        }
    }
    return matrix;
}


/*
    Performs matrix multiplication. sets result to dst matrix: dst = AxB
    Assumes correct dimesnsions of dst (i.e. dst dimnesions are (rowsA x colsB) and colsA == rowsB)
    Paramters:
        dst = AxB
        A - Matrix A
        rowsA - number of rows in matrix A
        colsA - number of columns in matrix A
        B - Matrix B
        rowsB - number of rows in matrix B
        colsB - number of columns in matrix B
    Return:
*/
void dot(double **dst, double **A, int rowsA, int colsA, double **B, int rowsB, int colsB){
    int i, j, k;
    double sum;
    rowsB = rowsB + 1;
    for ( i = 0 ; i < rowsA; i++){
        for (j = 0 ; j < colsB; j++){
            sum = 0;
            for (k = 0 ; k < colsA; k++){
                sum += A[i][k] * B[k][j];
            }
            dst[i][j] = sum;
        }
    }
}

/*
    Performs matrix substraction. sets result to dst matrix: dst = A - B
    Assumes correct dimesnsions of dst (i.e. dst dimensions are (rowsA x colsA) and rowsA == rowsB, colsA == colsB)
    Paramters:
        dst - destination matrix
        A - Matrix A
        rowsA - number of rows in matrix A
        colsA - number of columns in matrix A
        B - Matrix B
        rowsB - number of rows in matrix B
        colsB - number of columns in matrix B
    Return:
*/
void sub(double **dst, double **A, double **B, int rows, int columns){
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < columns; j++){
            dst[i][j] = A[i][j] - B[i][j];
        }
    }
}

/*
    Performs inplace transpose on a matrix.
    Paramters:
        A - Matrix to transpose
        rows - number of rows in matrix A
        cols - number of columns in matrix A
    Return:
*/
void transpose(double **A, int rows, int cols){
    int i,j;
    double temp;
    for (i = 0; i < rows; i++){
        for (j = i+1; j < cols; j++){
            temp = A[i][j];
            A[i][j] = A[j][i];
            A[j][i] = temp;
        }
    }
}



/*
    Returns the sign of a value.
    Paramters:
        value
    Return:
        1 for non-negetive values
        -1 for negetive values
*/
int sign(double value){
    if (value >=0) return 1;
    return -1;
}

/*
    Copies origin matrix to destination matrix.
    Assumes the destination matrix and the origin matrix are of the same dimensions.
    Paramters:
        destination - destination matrix
        origin - origin matrix
        rows - number of rows
        cols - number of columns
    Return:
*/
void copyMatrix(double **destination, double **origin, int rows, int cols){
    int i, j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            destination[i][j] = origin[i][j];
        }
    }
}


/*
    Calculates c and s according to expressions in the instructions.
    Paramters:
        matrix - Matrix to get values from
        i,j - indexes of the element with the largest absolute value in the matrix
        c - variable to assign result c in
        s - variable to assign result s in
    Return:
        assigns c and s to the variables passed
*/
void calcCS(double **matrix, int i, int j, double *c, double *s){
    double theta,t;
    theta = (matrix[j][j] - matrix[i][i])/(2 * matrix[i][j]);
    t = sign(theta) / (fabs(theta) + sqrt(pow(theta, 2) + 1));
    *c = 1/sqrt(pow(t, 2) + 1);
    *s = t * (*c);
}


/*
    initialises the matrix to a rotation matrix P(row, col) as in instructions. initialises inplace
    Paramters:
        rotationMatrix - Matrix to transform into a rotation matrix
        row,col - indexes of the element with the largest absolute value in the matrix
        c - value of c as in example
        s - value of s as in example
    Return:
*/
void InitRotationMatrix(double **rotationMatrix, int size, int row, int col, double c, double s){
    int i, j;
    double value;
    for (i = 0; i < size; i++){
        for (j = 0; j < size; j++){
            if (i != j){
                value = 0;
            }
            else{
                value = 1;
            }
            rotationMatrix[i][j] = value;
        }
    }
    rotationMatrix[row][row] = c;
    rotationMatrix[col][col] = c;
    rotationMatrix[row][col] = s;
    rotationMatrix[col][row] = -s;
}

/*
    Calculates off function of a matrix. Assumes the matrix is symmetrical.
    Paramters:
        matrix - Matrix to calculed off of.
        row,col - number of rows and columns in the matrix.
    Return:
        off(matrix)
*/
double off(double **matrix, int dim){
    int i,j;
    double sum = 0;
    for (i = 0; i < dim; i++){
        for (j = i + 1; j < dim; j++){
            sum += pow(matrix[i][j], 2);
        }
    }
    return sum * 2;
}



/*
    Calculates and prints the matrix's eigenvalues and eigenvectors
    Paramters:
        matrix - Matrix to run jacobi algorithm on. Assumes the matrix is symmetrical (Maybe needs checking in the main body)
        dim - Dimension of the matrix (i.e. the dimensions of the matrix are dim X dim)
    Return:
        A matrix of dimensions (dim + 1 X dim).  The first row is the eigenvalues and the other rows are the eigenvectors.
*/
double ** jacobi(double **vectors, int dim){
    double espilon = 1;
    int iterNum = 1, i, j, r;
    double maxValue;
    int maxIndexRow = 0, maxIndexCol = 1;
    double currentValue, c, s;
    double **matrix, **matrixPrime, **finalEigenvectors, **rotationMatrix, **tempEigenvectors;
    double **result;
    double offA, offA_prime, **temp;

    /* Memory allocation part */
    double ***memory = malloc(sizeof(double **) * 5);
    int memory_index = 0;
    if (memory == NULL) return NULL;

    matrix = createBlockMatrix(sizeof(double), dim, dim);
    memory[memory_index++] = matrix;

    matrixPrime = createBlockMatrix(sizeof(double), dim, dim);
    memory[memory_index++] = matrixPrime;

    finalEigenvectors = createBlockMatrix(sizeof(double), dim,  dim);
    memory[memory_index++] = finalEigenvectors;

    rotationMatrix = createBlockMatrix(sizeof(double), dim, dim);
    memory[memory_index++] = rotationMatrix;

    tempEigenvectors = createBlockMatrix(sizeof(double), dim, dim);
    memory[memory_index++] = tempEigenvectors;

    if (matrix == NULL || matrixPrime == NULL || finalEigenvectors == NULL || rotationMatrix == NULL || tempEigenvectors == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }

    /* Algorithm part */
    /* Copying vectors to new matrix. Easier to maintain memory this way.*/
    copyMatrix(matrix, vectors, dim, dim);
    maxValue = fabs(matrix[0][1]);
    /*char suffix;*/
    while (espilon > CONVERGENCE && iterNum <= 100){
        /*Finding the off-diagonal element with the largest absolute value. The matrix is symmetric */
        maxValue = fabs(matrix[0][1]);
        maxIndexRow = 0;
        maxIndexCol = 1;
        for (i = 0; i < dim; i++){
            for (j = i + 1; j < dim; j++){
                currentValue = matrix[i][j];
                if (fabs(currentValue) > maxValue){
                    maxIndexRow = i;
                    maxIndexCol = j;
                    maxValue = fabs(currentValue);
                }
            }
        }

        /* matrix[maxIndexRow][maxIndexCol] is the element with the largest absolute value */
        calcCS(matrix, maxIndexRow, maxIndexCol, &c, &s);
        /* Calculating off(A) */
        offA = off(matrix, dim);
        /* Copying A to A prime */
        copyMatrix(matrixPrime, matrix, dim, dim);
        /* Calculating change of A to A' */
        /* Note that changes should be made symetrically  */
        for (r = 0; r < dim; r++){
            if (r != maxIndexRow && r != maxIndexCol){
                matrixPrime[r][maxIndexRow] = c * matrix[r][maxIndexRow] - s * matrix[r][maxIndexCol];
                matrixPrime[maxIndexRow][r] = c * matrix[r][maxIndexRow] - s * matrix[r][maxIndexCol];

                matrixPrime[r][maxIndexCol] = c * matrix[r][maxIndexCol] + s * matrix[r][maxIndexRow];
                matrixPrime[maxIndexCol][r] = c * matrix[r][maxIndexCol] + s * matrix[r][maxIndexRow];
                
            }
        }
        /* Note that maxIndexRow and maxIndexCol cannot be equal as they're coordinates of an off-diagonal element */
        matrixPrime[maxIndexRow][maxIndexRow] = c * c * matrix[maxIndexRow][maxIndexRow] + s * s * matrix[maxIndexCol][maxIndexCol] -2 * s * c * matrix[maxIndexRow][maxIndexCol];
        matrixPrime[maxIndexCol][maxIndexCol] = s * s * matrix[maxIndexRow][maxIndexRow] + c * c * matrix[maxIndexCol][maxIndexCol] + 2 * s * c * matrix[maxIndexRow][maxIndexCol];

        matrixPrime[maxIndexRow][maxIndexCol] = 0;
        matrixPrime[maxIndexCol][maxIndexRow] = 0;

        /* Calculating off(A') */
        offA_prime = off(matrixPrime, dim);
        /* Calculating epsilon */
        /*espilon = offA - offA_prime;*/
        espilon = offA - offA_prime;
        /* Setting A = A' according to expressions */
        temp = matrix;
        matrix = matrixPrime;
        matrixPrime = temp;
        /* Calculating eigenvalues */
        InitRotationMatrix(rotationMatrix, dim, maxIndexRow, maxIndexCol, c, s);
        if (iterNum != 1){
            copyMatrix(tempEigenvectors, finalEigenvectors, dim, dim);
            dot(finalEigenvectors, tempEigenvectors, dim, dim, rotationMatrix, dim, dim);
        }
        else{
            copyMatrix(finalEigenvectors, rotationMatrix, dim, dim);
        }
        iterNum++;
    }
    /* Copying results to new array */
    result = createBlockMatrix(sizeof(double), dim + 1, dim); /* Was dim + 1 on col (why?). Delete when done */
    if (result == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }

    for (i = 0; i < dim; i++){
        result[0][i] = matrix[i][i];
    }
    copyMatrix((result + 1), finalEigenvectors, dim, dim);

    freeArrayOfBlocks(memory, memory_index);
    return result;
}

double minusSqrt(double value){
    return pow(value, -0.5);
}


/**
 * @brief Calculates The Normalized Graph Laplacian
 * 
 * @param vectors data points
 * @param numberOfVectors number of data points
 * @param length length of data points
 * @return double** The Normalized Graph Laplacian
 */
double **lnorm(double ** vectors, int numberOfVectors, int length){
    double **diag, **weighted, **multLeft, **multRight, **eyeMatrix, **lnorm;
    int memory_index = 0, i;
    double ***memory = (double ***) malloc(sizeof(double **) * 6);
    if (memory == NULL) return NULL;

    
    weighted = wam(vectors, numberOfVectors, length);
    memory[memory_index++] = weighted;

    if (weighted == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }

    diag = ddg(vectors, numberOfVectors, length);
    memory[memory_index++] = diag;
    if (diag == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }
    for (i = 0; i < numberOfVectors; i++){
        diag[i][i] = pow(diag[i][i], -0.5);
    }
    
    multLeft = createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    memory[memory_index++] = multLeft;
    if (multLeft == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }
    dot(multLeft, diag, numberOfVectors, numberOfVectors, weighted, numberOfVectors, numberOfVectors);

    multRight = createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    memory[memory_index++] = multRight;
    if (multRight == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }
    dot(multRight, multLeft, numberOfVectors, numberOfVectors, diag, numberOfVectors, numberOfVectors);

    eyeMatrix = eye(numberOfVectors);
    memory[memory_index++] = eyeMatrix;
    if (eyeMatrix == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }

    lnorm = createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    if (lnorm == NULL){
        freeArrayOfBlocks(memory, memory_index);
        return NULL;
    }
    sub(lnorm, eyeMatrix, multRight, numberOfVectors, numberOfVectors);

    freeArrayOfBlocks(memory, memory_index);
    return lnorm;
}

/**
 * @brief Inplace transposition of a non squared matrix
 * 
 * @param matrix matrix to transpose
 * @param rows number of rows in matrix
 * @param cols number of columns in matrix
 * @return Returns a new matrix = transpose(matrix)
 */
double** transposeNonSquareMatrix(double **matrix, int rows, int cols){
    int i, j;
    double **tempMatrix = createBlockMatrix(sizeof(double), cols, rows);
    if (tempMatrix == NULL){
        return NULL;
    }
    /* Creating the transposed matrix (as the matrix is not square) */
    for (i = 0; i < cols; i++){
        for (j = 0; j < rows; j++){
            tempMatrix[i][j] = matrix[j][i];
        }
    }
    return tempMatrix;
}



/**
 * @brief A comparator to sort arrays by their first item
 * 
 * @param x Array 1
 * @param y Array 2
 * @return int
 */
int myCompare(const void *x, const void *y){
    double **xAr = (double **) x;
    double **yAr = (double **) y;
    double xValue = **xAr;
    double yValue = **yAr;
    if (xValue == yValue) return 0;
    return xValue > yValue ? -1 : 1;

}

/**
 * @brief Normal Spectral Clustering algorthim according to instructions
 * 
 * @param vectors Data points
 * @param numberOfVectors number of data points
 * @param length dimnesion of data point (length of the vectors)
 * @return double** Points in R^K to cluster (each row is a vector)
 */
double **normalSpectralClustering(double **vectors, int numberOfVectors, int length, int *numberOfClusters){
    double **lnormMatrix, **transposeMatrix, **T, **temp, *transposeBlock;
    int K=1, i, j;
    double argMax=0, arg, sum;
    /* Creating Lnorm matrix of X */
    lnormMatrix = lnorm(vectors, numberOfVectors, length);
    if (lnormMatrix == NULL) return NULL;
    /* Getting eigenvalues and eigenvectors */
    transposeMatrix = jacobi(lnormMatrix, numberOfVectors);
    temp = transposeNonSquareMatrix(transposeMatrix, numberOfVectors + 1, numberOfVectors);
    if (temp == NULL){
        freeBlock(transposeMatrix);
        return NULL;
    }
    freeBlock(transposeMatrix);
    transposeMatrix = temp;
    /*  explanation about the sorting. MyCompare compares arrays by their first item (in decreasing)
        so qsort sorts the rows in the matrix by their first item.
        As the first item (after transposing the jacobi matrix) are the eigenvalues, qsort sorts the rows by decreasing
        corresponding eigenvalues */
    transposeBlock = *transposeMatrix;
    qsort(transposeMatrix, numberOfVectors, sizeof(double *), &myCompare); /*qsort changes first object so when im trying to free a block using *ptr the pointer of
    the block is not correct. Solution ==> Save pointer of the block*/
    temp = transposeNonSquareMatrix(transposeMatrix, numberOfVectors, numberOfVectors + 1);
    if (temp == NULL){
        freeBlock(transposeMatrix);
        return NULL;
    }
    free(transposeBlock);
    free(transposeMatrix);
    transposeMatrix = temp;
    /* If number of clusters is zero we need to find K according to heuristic */
    if (*numberOfClusters  == 0){
        /* Finding K */
        for (i = 0 ; i < numberOfVectors/2; i++){
            arg = transposeMatrix[0][i] - transposeMatrix[0][i+1];
            if (arg > argMax){
                K = i+1;
                argMax = arg;
            }
        }
        *numberOfClusters = K;
    }
    K = *numberOfClusters;
    T = createBlockMatrix(sizeof(double), numberOfVectors, K);
    if (T == NULL){
        freeBlock(transposeMatrix);
        return NULL;
    }
    /* This copies only the first K columns without the first row in them (as the first row in transposeMatrix are the eigenvalues) */
    copyMatrix(T, transposeMatrix + 1, numberOfVectors, K);
    /* Normalizing values of T*/
    for (i = 0; i < numberOfVectors; i++){
        sum = 0;
        for (j = 0; j < K; j++){
            sum += pow(T[i][j], 2);
        }
        if (sum != 0){
            sum = pow(sum, 0.5);
            for (j = 0; j < K; j++){
                T[i][j] = T[i][j] / sum;
            }
        }
    }
    freeBlock(transposeMatrix);
    freeBlock(lnormMatrix);
    return T;
}

int main(int argc, char* argv[]){
    /*
    Gets goal and input file.
    Does not need kmeans as kmeans only used in python.
    TODO: * Implement wam 
          * Implement ddg 
          * Implement lnorm
          * Implement jacboi
    */
   char *input, *operation, c;
   double **vectors, **result;
   int length = 1, numberOfVectors=0;
   FILE *input_file;
   
    if (argc != 3) return 1;

    operation = argv[1];
    input = argv[2];

    if (!checkTextFormat(input)) errorMsg(0); /* File is not a txt or csv file */

    input_file = fopen(input, "r");
    if (input_file == NULL) errorMsg(0); /* File couldn't be open */

    /* Getting length of the vectors and number of vectors */
    while ((c = fgetc(input_file)) != '\n'){
        if (feof(input_file)){
            fclose(input_file);
            errorMsg(0);
            return 1;
        }
        if (c == ',') length++;
    }
    if (fseek(input_file, 0, SEEK_SET)){
        errorMsg(1);
        return 1;
    }
    while ((c = fgetc(input_file))){
        if (feof(input_file)) break;
        if (c == '\n') numberOfVectors++;
    }
    if (numberOfVectors == 0){
        errorMsg(0);
        return 1;
    }
    if (fseek(input_file, 0, SEEK_SET)){
        errorMsg(1);
        return 1;
    }

    /* Parsing vectors from input file */
    vectors = parseMatrix(input_file, numberOfVectors, length);
    if (vectors == NULL) errorMsg(1);

    /* Closing file */
    fclose(input_file);
    /* Jacobi */
    if (!strcmp(operation, "jacobi")){ 
        result = jacobi(vectors, numberOfVectors);
        if (result == NULL){
            freeBlock(vectors);
            errorMsg(1);
        }
        printMatrix(result, numberOfVectors + 1, numberOfVectors);
        freeBlock(vectors);
        freeBlock(result);
        return 0;
    }

    /* Wam */
    if (!strcmp(operation, "wam")){
        result = wam(vectors, numberOfVectors, length);
    }
    /* Ddg */
    else if (!strcmp(operation, "ddg")){
        result = ddg(vectors, numberOfVectors, length);
    }

    /* Lnorm */
    else if (!strcmp(operation, "lnorm")){
        result = lnorm(vectors, numberOfVectors, length);
    }

    if (result == NULL){
            freeBlock(vectors);
            errorMsg(1);
    }
    printMatrix(result, numberOfVectors, numberOfVectors);
    freeBlock(result);
    freeBlock(vectors);
    return 0;
}
