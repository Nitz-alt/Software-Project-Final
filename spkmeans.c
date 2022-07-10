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


int main(){
    /*
    Gets goal and input file.
    Does not need kmeans as kmeans only used in python.
    TODO:   Implement wam
            Implement ddg
            Implement lnorm
            Implement jacboi
    */
   return 0;

}
