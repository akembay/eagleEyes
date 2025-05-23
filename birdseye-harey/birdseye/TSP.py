"""TSP.py - solve the Traveling Salesperson Problem with _elegance_.
"""
from itertools import product
# from sys import stdout as out
from mip import Model, xsum, minimize, BINARY


TIMEOUT = 200


def tsp(places, dists):
    # number of nodes and list of vertices
    n, V = len(dists), set(range(len(dists)))

    # distances matrix
    c = [[0 if i == j
          else dists[i][j-i-1] if j > i
          else dists[j][i-j-1]
          for j in V] for i in V]

    model = Model()

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

    # continuous variable to prevent subtours: each city will have a
    # different sequential id in the planned route except the first one
    y = [model.add_var() for i in V]

    # objective function: minimize the distance
    model.objective = minimize(xsum(c[i][j]*x[i][j] for i in V for j in V))

    # constraint : leave each city only once
    for i in V:
        model += xsum(x[i][j] for j in V - {i}) == 1

    # constraint : enter each city only once
    for i in V:
        model += xsum(x[j][i] for j in V - {i}) == 1

    # subtour elimination
    for (i, j) in product(V - {0}, V - {0}):
        if i != j:
            model += y[i] - (n+1)*x[i][j] >= y[j]-n

    # optimizing
    model.optimize(max_seconds_same_incumbent=TIMEOUT)

    # checking if a solution was found
    if model.num_solutions:
        out = []
        print(f'route with total distance {model.objective_value} found')
        out.append(places[0])
        nc = 0
        while True:
            nc = [i for i in V if x[nc][i].x >= 0.99][0]
            out.append(places[nc])
            if nc == 0:
                return out

if __name__ == '__main__':
    # names of places to visit
    places = ['Antwerp', 'Bruges', 'C-Mine', 'Dinant', 'Ghent',
              'Grand-Place de Bruxelles', 'Hasselt', 'Leuven',
              'Mechelen', 'Mons', 'Montagne de Bueren', 'Namur',
              'Remouchamps', 'Waterloo']

    # distances in an upper triangular matrix
    dists = [[83, 81, 113, 52, 42, 73, 44, 23, 91, 105, 90, 124, 57],
             [161, 160, 39, 89, 151, 110, 90, 99, 177, 143, 193, 100],
             [90, 125, 82, 13, 57, 71, 123, 38, 72, 59, 82],
             [123, 77, 81, 71, 91, 72, 64, 24, 62, 63],
             [51, 114, 72, 54, 69, 139, 105, 155, 62],
             [70, 25, 22, 52, 90, 56, 105, 16],
             [45, 61, 111, 36, 61, 57, 70],
             [23, 71, 67, 48, 85, 29],
             [74, 89, 69, 107, 36],
             [117, 65, 125, 43],
             [54, 22, 84],
             [60, 44],
             [97],
             []]

    out = tsp(places, dists)
    print(f'    starting at {out[0]}, ')
    for i in out[1:-1]:
        print(f'    move to {i}, then')
    print(f'    end at {out[-1]}')
