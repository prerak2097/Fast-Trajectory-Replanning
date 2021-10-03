import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
import random as rand
from matplotlib import colors

ROW=101
COL=101

stack_gridworlds = []                           #stack that will contain 50 grids
grid1=np.zeros((ROW,COL),dtype=np.int64)        #first grid we will test on
visited = np.zeros((ROW,COL),dtype=bool)        #boolean array of visited or not
neighbors = [(0,-1), (1,0), (0,1), (-1,0)]      #neighbors to visit during DFS

def is_valid(row,col):                          #is row/col within the grid 
    global ROW
    global COL
    global visited
    #if cell is out of bounds
    if (row < 0 or col < 0 or row>= ROW or col >= COL):
        return False
    #if the cell is visited return false
    if visited[row][col]:
        return False
    #return true otherwise
    return True


def DFS(row,col,grid):                          #DFS on grid to mark blocked/unblocked cells
    global visited
    global neighbors
    stack_pairs = []
    stack_pairs.append([row,col])

    while len(stack_pairs) > 0:
        current = stack_pairs[-1]
        stack_pairs.pop()
        row = current[0]
        col = current[1]

        if (is_valid(row,col) == False):
            continue

        visited[row][col] = True

        #do the random make the cell blocked or unblocked here
        grid[row][col] = 1 if rand.random() < 0.7 else -1

        #push all adjacent cells randomly
        neighbors = np.random.permutation(neighbors)
        for i in range(4):
            adj_hor = row + neighbors[i][0]
            adj_vert = col + neighbors[i][1]
            stack_pairs.append([adj_hor,adj_vert])
    return grid

grid1 = DFS(0,0,grid1)

pyplot.figure(figsize=(30,30))
colormap = colors.ListedColormap(["darkblue","lightblue"])
pyplot.imshow(grid1,colormap)
pyplot.show()
