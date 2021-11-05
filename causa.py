import scipy
from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import connected_components

def get_minimum_connections(matrix):
    graph = csr_matrix(matrix)
    cc,other = scipy.sparse.csgraph.connected_components(csgraph=graph,directed=False)
    return cc-1

matrix = \
    [
        [False, True, False, False, True],
        [True, False, False, False, False],
        [False, False, False, True, False],
        [False, False, True, False, False],
        [True, False, False, False, False]
    ]
matrix2 = \
    [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True]
    ]
matrix3 = \
    [
        [False, True, False, False, True],
        [True, False, False, True, False],
        [False, False, False, True, False],
        [False, False, True, False, False],
        [True, False, False, False, False]
    ]
print(get_minimum_connections(matrix)) # should print 1
print(get_minimum_connections(matrix2)) # should print 1
print(get_minimum_connections(matrix3)) # should print 1