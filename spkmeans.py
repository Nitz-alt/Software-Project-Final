from tabnanny import check
import numpy as np
import pandas as pd
import sys
import spkmeans


def init_centeroids(K : int, vectors : list):
    np.random.seed(0)
    centeroids = []
    indexes = [_ for _ in range(K)]
    index = np.random.choice(len(vectors));
    lastIndex = 0
    indexes[lastIndex] = index
    lastIndex += 1
    centeroids.append(vectors[index])
    distances = [_ for _ in range(len(vectors))]
    probabilities = [_ for _ in range(len(vectors))]
    for _ in  range(K-1):
        for index, vector in enumerate(vectors):
            min_Distance = sys.float_info.max
            for centeroid in centeroids:
                v = (centeroid - vector) ** 2
                min_Distance = min(min_Distance, np.sum(v))
            distances[index] = min_Distance
        distances_sum = sum(distances)
        for i in range(len(probabilities)):
            probabilities[i] = distances[i] / distances_sum
        index = np.random.choice(len(vectors), p=probabilities);
        indexes[lastIndex] = index
        lastIndex += 1
        centeroids.append(vectors[index])
    # print(','.join([str(i) for i in indexes]))
    return ([cent.tolist() for cent  in centeroids], indexes)



def checkTxtFormat(fileName: str):
    end = fileName.split('.')[-1]
    return end == "txt" or end == "csv"

def invalid_input():
    print("Invalid input!")
    sys.exit(1)

def checkInput(cond : bool):
    if (cond):
        invalid_input()


def extract_vectors(file_name1 : str):
    # Does the input has indexes or not ? >> Not clear from instructions (Maybe from 0 to N-1 (?))
    df1 = pd.read_csv(file_name1, sep = ',', header=None) #.set_index(0)
    return df1 #.sort_values(0)


def vectorsEngineering(i):
    vectors = extract_vectors(sys.argv[i])
    ind = vectors.index.astype('int32').to_numpy()
    vectors = vectors.to_numpy()
    return (ind, vectors)

def print_lst(lst):
    lst = [str(ind) for ind in lst]
    lst_to_print = ",".join(lst)
    print(lst_to_print)

def print_mat(mat):
    for vec in mat:
        print_vec(vec)

def print_vec(vec):
    first_coor = float(vec[0])
    coordinate = "{:.4f}".format(first_coor)
    print(coordinate, end='')
    for i in range(1, len(vec)):
        coor = float(vec[i])
        coor = ",{:.4f}".format(coor)
        print(coor, end='')
    print()


if __name__ == "__main__":
    operations = ["spk", "wam", "ddg", "lnorm", "jacobi"]
    max_iter = 300

    if (len(sys.argv)) != 4:
        invalid_input()
    try:
        K = int(sys.argv[1])
    except:
        invalid_input()
    checkInput(K<0)
    # Why in Adi's code it also raises en error when K = 1 ?
    #if(K == 0):
    #    print("Need to use the heuristic 1.3 in C")
        # Need to use the heuristic 1.3 in C, I think it's automatically like that in Nitzan's code
    operation = sys.argv[2]
    if operation not in operations:
        invalid_input()
    checkInput(not checkTxtFormat(sys.argv[3]))
    input_file = sys.argv[3]
    ind, vectors = vectorsEngineering(3)
    checkInput(len(vectors) < K)
    numOfVecs = len(vectors)
    lengthOfVec = len(vectors[0])
    vecsLst = vectors.tolist()

    res_mat = None
    if operation == 'spk':

        mat_T,K = spkmeans.normalSpectralClustering(vecsLst,len(vectors),len(vectors[0]),K)
        if K <= 0:
            invalid_input()
        mat_T_numpy = np.array(mat_T)
        centeroids, indices = init_centeroids(K, mat_T_numpy)
        print_lst(indices)
        res_mat = spkmeans.kmeans(centeroids,mat_T,len(mat_T),len(mat_T[0]),K)
    
    elif operation == 'wam':
        res_mat = spkmeans.wam(vecsLst,numOfVecs,lengthOfVec)
    
    elif operation == 'ddg':
        res_mat = spkmeans.ddg(vecsLst,numOfVecs,lengthOfVec)
    
    elif operation == 'lnorm':
        res_mat = spkmeans.lnorm(vecsLst,numOfVecs,lengthOfVec)
    
    elif operation == 'jacobi':
        res_mat = spkmeans.jacobi(vecsLst,numOfVecs)

    
    else:
        invalid_input()
    if res_mat is None:
        invalid_input()

    if operation in ['wam','ddg','lnorm','spk']:
        print_mat(res_mat)

    if operation == 'jacobi':
        eigVals = res_mat[0]
        print_vec(eigVals)
        res_mat.pop(0)
        print_mat(res_mat)


    

        


    

    ## What should the epsilon and max iter be? MAX_ITER == 300

