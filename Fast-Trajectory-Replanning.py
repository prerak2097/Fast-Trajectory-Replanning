from matplotlib import animation
import numpy as np
from numpy.lib.type_check import asfarray
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
import random as rand
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import math
import heapq as heap
ROW=101
COL=101 
BLOCKED = -1
OPEN = 0
class node:
    def __init__(self,index,hval=None,gval=None,fval=None,nextnode=None) -> None:
        self.index= index
        self.hval = hval
        self.gval = gval
        self.fval = fval
        self.nextnode = nextnode
    def compute_f(self):
        self.fval= self.gval+self.hval
    def compute_h(self,goal_idx):
        self.hval = abs(goal_idx[0]-self.index[0]) + abs(goal_idx.index[1]-self.index[1])
visited = np.zeros((ROW,COL),dtype=bool)        #boolean array of visited or not
neighbors = [(0,-1), (1,0), (0,1), (-1,0)]      #neighbors to visit during DFS
def is_valid(row,col):                          #DFS Helper method 
    global ROW
    global COL
    global visited
    #if cell is out of bounds
    if (row < 0 or col < 0 or row>= ROW or col >= COL):
        return False
    if visited[row][col]:
        return False
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
        grid[row][col] = 0 if rand.random() < 0.7 else -1

        #push all adjacent cells randomly
        neighbors = np.random.permutation(neighbors)
        for i in range(4):
            adj_hor = row + neighbors[i][0]
            adj_vert = col + neighbors[i][1]
            stack_pairs.append([adj_hor,adj_vert])
    return grid
def set_grid (grid):
    rand_row=rand.randint(0,100)
    rand_col=rand.randint(0,100)
    grid=DFS(rand_row,rand_col,grid)
    return grid

def compute_path(grid, start_node, goal_node):
    closedlist=set()
    path=[]
    openlist=[]
    heap.push(openlist, start_node)
    #FINISH LATER

#MAIN
grid1 = np.array([[0,-1,0,0,0],
        [0,-1,-1,0,0],
        [0,0,0,0,0],
        [0,-1,0,0,0],
        [0,-1,0,0,0]])
counter=0
state_grid = np.zeros((5,5),dtype=np.int64)
start_idx = node((0,0))
goal = node((4,4))
while start_idx!=goal:
    counter +=1
    start_idx.gval = 0
    state_grid[start_idx.index[0]][start_idx.index[1]]= counter
    goal.gval=math.inf
    state_grid[goal.index[0]][goal.index[1]] = counter
    openlist =[]
    ### MUST COMPUTE THINGS HERE
    heap.heappush(openlist,start_idx)





















#Visualizations
#fig =pyplot.figure(figsize=(30,30))
#colormap = colors.ListedColormap(["darkblue","lightgreen","Red","yellow"])
#im=pyplot.imshow(grid1,colormap, animated=True)
#red_patch = mpatches.Patch(color='red', label='Agent')
#yellow_patch = mpatches.Patch(color='yellow', label='Target')
#green_patch = mpatches.Patch(color='lightgreen', label= "Open and Unblocked")
#blue_patch = mpatches.Patch(color= "darkblue", label = "blocked")
#pyplot.legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), handles=[red_patch,green_patch,blue_patch,yellow_patch])

# create movie array which captures all of the changes in the agent, and shwo the movie from the code in pygameimp.py
#pyplot.show()



