from tabnanny import check
import numpy as np
import pandas as pd
import sys


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
    return [cent.tolist() for cent  in centeroids]



def checkTxtFormat(fileName: str):
    end = fileName.split('.')[-1]
    return end == "txt" or end == "csv"

def invalid_input():
    print("Invalid input!")
    sys.exit(1)

def checkInput(cond : bool):
    if (cond):
        invalid_input()


def extract_vectors(file_name1 : str, file_name2 : str):
    # Does the input has indexes or not ? >> Not clear from instructions (Maybe from 0 to N-1 (?))
    df1 = pd.read_csv(file_name1, sep = ',', header=None).set_index(0)
    return df1.sort_values(0)


def vectorsEngineering(i, j):
    vectors = extract_vectors(sys.argv[i])
    ind = vectors.index.astype('int32').to_numpy()
    vectors = vectors.to_numpy()
    return (ind, vectors)

if __name__ == "__main__":
    if (len(sys.argv)) < 4:
        invalid_input()
    try:
        K = int(sys.argv[1])
    except:
        invalid_input()
    checkInput(K<0)

    ## What should the epsilon and max iter be? MAX_ITER == 300

