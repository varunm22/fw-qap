import sys
import numpy as np
import sfw

def test(num):
    def process_file(suffix):
        file_lines = open("test-files/" + str(num) + "-" + suffix, 'r').readlines()
        return [[int(elt) for elt in line.strip().split(' ')] for line in file_lines]

    def generate_in():
        elts = process_file("in")
        n = len(elts[0])
        return(np.array(elts[:n]), np.array(elts[n:]))

    def generate_out():
        elts = process_file("out")
        return(np.array(elts[0]), elts[1][0])

    A, B = generate_in()
    f_real, P_real, _, _ = sfw.sfw(A, B, i_max = 10) #ignored args are final x and num iters
    P, f = generate_out()
    print("EXPECTED RESULTS")
    print("P: ", P)
    print("f: ", f)
    print("OBTAINED RESULTS")
    print("P: ", P_real)
    print("f: ", f_real)
    if np.array_equal(P, P_real) and f == f_real:
        print("PASS")
    else:
        print("FAIL")

test(sys.argv[1])
