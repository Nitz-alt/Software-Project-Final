#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void **MEMORY_LIST;
int MEMORY_LIST_INDEX = 0;


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

void printMatrix(double **array, size_t numberOfRows, size_t numberOfColumns){
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

/*
    Frees all memory allocated throughout the program
    Paramters:
        None
    Return:
        void
*/
void freeAllMemory(){
    int i;
    for(i = 0 ; i < MEMORY_LIST_INDEX; i++){
        if (MEMORY_LIST[i] != NULL){
            free(MEMORY_LIST[i]);
        }
    }
    if (MEMORY_LIST != NULL){
        free(MEMORY_LIST);
    }
}

void errorMsg(int code){
    /* Free memory */
    freeAllMemory();
    if (code == 0){
        printf("Invalid input");
    }
    if (code == 1)
        printf("An Error Has Occurred");
}

/*
    Paramters:
        size - size of memory to be allocated
    Return:
        Pointer to the new memory allocated
*/
void * allocMemory(size_t size){
    void *memoToAssign = malloc(size);
    MEMORY_LIST[MEMORY_LIST_INDEX++] = memoToAssign;
    return memoToAssign;
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
    Paramters:
        input - File to parse vectors from
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        A matrix of the vectors parsed from the file
*/
double ** parseMatrix(FILE *input, size_t numberOfVectors, size_t length){
    size_t i,j;
    char *suffix;
    double *block = allocMemory(sizeof(double) * numberOfVectors * length);
    double ** matrix = allocMemory(sizeof(double *) * numberOfVectors);
    double *vector;

    for (i = 0; i < numberOfVectors; i++){
        matrix[i] = block + i * length;
    }

    for (i = 0; i < numberOfVectors; i++){
        vector = matrix[i];
        for (j = 0; j < length; j++){
            if (j != length - 1){
                fscanf(input, "%lf,", (vector + j));
            }
            else{
                fscanf(input, "%lf\n", (vector + j));
            }
        }
    }
    return matrix;
}

/*
    Prints the weighted matrix of the input vectors
    Paramters:
        Matrix - matrix of vectors to calculated the weighted matrix on
        numberOfVectors - number of vecotrs
        length - length of each vector in the matrix
    Return:
        
*/

void wam(double **matrix, size_t numberOfVectors, size_t length){
    size_t i,j;
    double wij;
    double *weightedBlock = (double *) malloc(sizeof(double) * numberOfVectors * numberOfVectors);
    double **weightedMatrix = (double **) malloc(sizeof(double *) * numberOfVectors);
    for (i = 0; i < numberOfVectors; i++){
        weightedMatrix[i] = weightedBlock + i * numberOfVectors;
    }
    for (i = 0; i < numberOfVectors; i++){
        weightedMatrix[i][i] = 0;
        for (j = i + 1; j < numberOfVectors; j++){
            wij = exp(-1 * sqrt(eucleadDist(matrix[i], matrix[j], length)) * 0.5);
            weightedMatrix[i][j] = wij;
            weightedMatrix[j][i] = wij;
        }
    }
    printMatrix(weightedMatrix, numberOfVectors, numberOfVectors);
    free(weightedBlock);
    free(weightedMatrix);
}

void ddg(){

}

void lnorm(){

}




int main(int argc, char* argv[]){
    /*
    Gets goal and input file.
    Does not need kmeans as kmeans only used in python.
    TODO: * Implement wam *
            Implement ddg
            Implement lnorm
            Implement jacboi
    */
   char *input, *operation, c;
   double **vectors, result;
   size_t length = 1, numberOfVectors=0;
   FILE *input_file;
   
   
    if (argc != 3) return 1;

    operation = argv[1];
    input = argv[2];

    if (!checkTextFormat(input)) return 1; /* File is not a txt or csv file */

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
    MEMORY_LIST = malloc(sizeof(void *) * 2);

    if (!strcmp(operation, "jacobi")){ 
        return 0;
    }

    /* Parsing vectors from input file */
    vectors = parseMatrix(input_file, numberOfVectors, length);

    /* WAM */
    if (!strcmp(operation, "wam")){
        // wam(vectors, numberOfVectors, length);
        wam(vectors, numberOfVectors, length);
    }
    /* DDG */
    if (!strcmp(operation, "ddg")){
        ddg(vectors);
    }

    /* LNORM */
    if (!strcmp(operation, "lnorm")){
        lnorm(vectors);
    }
    freeAllMemory();
    


}
