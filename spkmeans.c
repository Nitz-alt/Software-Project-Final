#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


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
void printAr(double *ar, size_t size){
    size_t i = 0;
    char suffix = ',';
    for(;i < size; i++){
        if (i == size-1){
            suffix = '\n';
        }
        printf("%.4f%c", ar[i], suffix);
    }
}

void printMatrix(double **array, size_t numberOfRows, size_t numberOfColumns){
    size_t i;
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
    size_t i=0;
    while ((c = number[i++]) != '\0'){
        if (c < '0' || c > '9' || c == '.'){
            return 0;
        }
    }
    return atoi(number);
}

void errorMsg(int code){
    /* Free memory */
    freeAllMemory();
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
    Allocates a matrix of dimension numberOfRows X numberOfColumns.
    The memory is allocate as a block.
    Paramters:
        numberOfRows - number of rows to in the matrix to allocate
        numberOfColumns - number of columns in the matrix to allocate
    Return:
        returns the pointer to the memory that was allocated
        
*/
double **createBlockMatrix(size_t size, size_t numberOfRows, size_t numberOfColumns){
    size_t i;
    double *block = (double *) malloc(size * numberOfRows * numberOfColumns);
    if (block == NULL) errorMsg(1);
    double **matrix = (double **) malloc(size * numberOfRows);
    if (matrix == NULL) errorMsg(1);
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
    Paramters:
        input - File to parse vectors from
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        A matrix of the vectors parsed from the file
*/
double **parseMatrix(FILE *input, size_t numberOfVectors, size_t length){
    size_t i,j;
    double ** matrix = createBlockMatrix(sizeof(double), numberOfVectors, length);
    double *vector;

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
double **wam(double **matrix, size_t numberOfVectors, size_t length){
    size_t i,j;
    double wij;
    double **weightedMatrix = (double **) createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
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
double sumOfRow(double **matrix, size_t length, size_t row){
    size_t i;
    double sum = 0;
    for (i = 0; i < length; i++){
        sum += matrix[row][i];
    }
    return sum;
}


/* 
    Paramters:
        matrix - matrix of vectors to calculate its diagonal degree matrix
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        Diagonal Degree Matrix of matrix
        
*/
double ** ddg(double **matrix, size_t numberOfVectors, size_t length){
    size_t i, j;
    double **diag_matrix = (double **) createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    for (i = 0; i < numberOfVectors; i++){
        for (j = 0; j < numberOfVectors; j++){
            if (j != i) diag_matrix[i][j] = 0;
            else diag_matrix[i][i] = sumOfRow(matrix, length, i);
        }
    }
    return diag_matrix;
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
void funcOnMatrix(double **dst_matrix , size_t numberOfRows, size_t numberOfColumns, double (*f)(double)){
    size_t i,j;
    for (i = 0 ; i < numberOfRows; i++){
        for (j = 0; j < numberOfColumns; j++){
            dst_matrix[i][j] = (*f)(dst_matrix[i][j]);
        }
    }
}

/*
    Returns the eye matrix of (size X size) dimensions 
    Paramters:
        size - dimension of matrix
    Return:
*/
double **eye(size_t size){
    size_t i,j;
    double **matrix = (double**) createBlockMatrix(sizeof(double), size, size);
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
void dot(double **dst, double **A, size_t rowsA, size_t colsA, double **B, size_t rowsB, size_t colsB){
    size_t i,j, k, iterNum, lowBound, left;
    double totalSum, sum1, sum2, sum3, sum4;
    rowsB = rowsB;
    for (i = 0; i < rowsA; i++){
        for (j = 0; j < colsB; j++){
            totalSum = 0;
            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
            sum4 = 0;
            iterNum = (size_t) colsA / 4;
            for (k = 0; k < iterNum; k++){
                sum1 += A[i][k] * B[k][j];
                sum2 += A[i][k+1] * B[k+1][j];
                sum3 += A[i][k+2] * B[k+2][j];
                sum4 += A[i][k+3] * B[k+3][j];
            }
            left = colsA % 4;
            lowBound = colsA - left;
            for (k = 0; k < left; k++){
             totalSum += A[i][lowBound + k] * B[lowBound + k][j];   
            }
            totalSum += sum1 + sum2 + sum3 + sum4;
            dst[i][j] = totalSum;
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
void sub(double **dst, double **A, double **B, size_t rows, size_t columns){
    size_t i,j;
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
void transpose(double **A, size_t rows, size_t cols){
    size_t i,j;
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
void copyMatrix(double **destination, double **origin, size_t rows, size_t cols){
    size_t i, j;
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
void InitRotationMatrix(double **rotationMatrix, size_t size, size_t row, size_t col, double c, double s){
    size_t i, j, value;
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
double off(double **matrix, size_t dim){
    size_t i,j;
    double sum;
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
double ** jacobi(double **vectors, size_t dim){
    double espilon = 1, convergence;
    size_t iterNum = 1, i, j, r;
    double maxValue;
    size_t maxIndexRow = 0, maxIndexCol = 1;
    double currentValue, c, s;
    double **matrix = createBlockMatrix(sizeof(double), dim, dim);
    double **matrixPrime = createBlockMatrix(sizeof(double), dim, dim);
    double **finalEigenvectors = createBlockMatrix(sizeof(double), dim,  dim);
    double **rotationMatrix = createBlockMatrix(sizeof(double), dim, dim);
    double **tempEigenvectors = createBlockMatrix(sizeof(double), dim, dim);
    double **result;
    double offA, offA_prime, **temp;

    /* Copying vectors to new matrix. Easier to maintain memory this way.*/
    copyMatrix(matrix, vectors, dim, dim);
    maxValue = fabs(matrix[0][1]);
    /*char suffix;*/
    convergence = pow(10, -5);
    while (espilon > convergence && iterNum <= 100){
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
        espilon = fabs(offA - offA_prime);
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
    result = createBlockMatrix(sizeof(double), dim + 1, dim + 1);
    for (i = 0; i < dim; i++){
        result[0][i] = matrix[i][i];
    }
    copyMatrix((result + 1), finalEigenvectors, dim, dim);
    freeBlock(matrixPrime);
    freeBlock(finalEigenvectors);
    freeBlock(rotationMatrix);
    freeBlock(tempEigenvectors);
    freeBlock(matrix);
    return result;
}

double **lnorm(double ** vectors, size_t numberOfVectors, size_t length){
    double **diag, **weighted, **multLeft, **multRight, **eyeMatrix, **lnorm;
    diag = ddg(vectors, numberOfVectors, length);
    funcOnMatrix(diag, numberOfVectors, numberOfVectors, &sqrt);
    weighted = wam(vectors, numberOfVectors, length);
    multLeft = createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    dot(multLeft, diag, numberOfVectors, numberOfVectors, weighted, numberOfVectors, numberOfVectors);
    multRight = createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    dot(multRight, multLeft, numberOfVectors, numberOfVectors, diag, numberOfVectors, numberOfVectors);
    eyeMatrix = eye(numberOfVectors);
    lnorm = createBlockMatrix(sizeof(double), numberOfVectors, numberOfVectors);
    sub(lnorm, eyeMatrix, multRight, numberOfVectors, numberOfVectors);
    printMatrix(lnorm, numberOfVectors, numberOfVectors);
    freeBlock(diag);
    freeBlock(weighted);
    freeBlock(multLeft);
    freeBlock(multRight);
    freeBlock(eyeMatrix);
    return lnorm;
}




int main(int argc, char* argv[]){
    /*
    Gets goal and input file.
    Does not need kmeans as kmeans only used in python.
    TODO: * Implement wam 
          * Implement ddg 
          * Implement lnorm
            Implement jacboi
    */
   char *input, *operation, c;
   double **vectors, **result;
   size_t length = 1, numberOfVectors=0;
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
    rewind(input_file);
    while ((c = fgetc(input_file))){
        if (feof(input_file)) break;
        if (c == '\n') numberOfVectors++;
    }
    if (numberOfVectors == 0){
        errorMsg(0);
        return 1;
    }
    rewind(input_file);

    /* Parsing vectors from input file */
    vectors = parseMatrix(input_file, numberOfVectors, length);

    /* Closing file */
    fclose(input_file);

    /* Jacobi */
    if (!strcmp(operation, "jacobi")){ 
        result = jacobi(vectors, numberOfVectors);
        printMatrix(result, numberOfVectors + 1, numberOfVectors);
    }

    /* Wam */
    if (!strcmp(operation, "wam")){
        result = wam(vectors, numberOfVectors, length);
        printMatrix(result, numberOfVectors, numberOfVectors);
    }
    /* Ddg */
    else if (!strcmp(operation, "ddg")){
        result = ddg(vectors, numberOfVectors, length);
        printMatrix(result, numberOfVectors, numberOfVectors);
    }

    /* Lnorm */
    else if (!strcmp(operation, "lnorm")){
        result = lnorm(vectors, numberOfVectors, length);
        printMatrix(result, numberOfVectors, numberOfVectors);
    }
    freeBlock(result);
    freeBlock(vectors);
    return 0;
}
