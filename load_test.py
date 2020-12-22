import os
import numpy as np
from utils import *
from sklearn.metrics.pairwise import euclidean_distances

zero_indexed_slns = {("qaplib", "tai40a")}

def simple_loader(test):
    collection, test_name = test
    def process_file(in_or_out, ext):
        path = f"data/{collection}/{in_or_out}/{test_name}.{ext}"
        file_lines = open(path, 'r').readlines()
        file_contents = " ".join(file_lines).replace("\n"," ").split()
        return [int(elt) for elt in file_contents]

    def generate_in():
        elts = process_file("input", "dat")
        n = elts.pop(0)
        assert len(elts) == 2*n**2
        return np.array(elts[:n**2]).reshape((n,n)), np.array(elts[n**2:]).reshape((n,n))

    def generate_out():
        elts = process_file("output", "sln")
        n = elts.pop(0)
        f = elts.pop(0)
        assert len(elts) == n
        if test not in zero_indexed_slns:
            elts = [x-1 for x in elts]
        best_P = perm2mat(elts)
        return(f, best_P)

    W, D = generate_in()
    best_f, best_P = generate_out()
    if (best_f!=f_(best_P, W, D)):
        assert(best_f == f_(best_P.T, W, D))
        best_P = best_P.T
    return (W, D, best_f, best_P)

def load_tsp(test_name):
    path = f"data/tsplib/input/{test_name}.tsp"
    file_lines = open(path, 'r').readlines()
    reading_attributes = True
    attributes = {}
    vertices = []
    for line in file_lines:
        if line[0].isdigit():
            reading_attributes = False
            _, x, y = line.strip().split()
            vertices.append((float(x), float(y)))

        if reading_attributes:
            contents = line.strip().split(' : ')
            key = contents[0]
            if len(contents) == 1:
                val = ""
            else:
                val = contents[1]
            attributes[key] = val

    def dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1] -p2[1])**2)**0.5

    n = len(vertices)
    D = euclidean_distances(np.array(vertices))
    W = np.zeros((n,n))
    for i in range(n):
        W[i][(i+1)%n] = 0.5
        W[i][(i-1)%n] = 0.5
        # W[i][(i+1)%n] = 1


    best_sols = open("data/tsplib/best_sols", "r").readlines()
    best_f = int(dict([x.strip().split(' : ') for x in best_sols])[test_name])

    # print(np.linalg.svd(D))
    # print(np.linalg.svd(D**2))

    return(W, D, best_f)

def load(test):
    collection, test_name = test
    if collection == "neos-guide" or collection == "qaplib":
        return(simple_loader(test))
    elif collection == "tsplib":
        W, D, best_f = load_tsp(test_name)
        return (W, D, best_f, None)
    else:
        assert False

# print(load_tsp("pcb1173"))
