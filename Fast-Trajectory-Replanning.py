
from typing import List
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
import node as n

# print('index : {idx} | g {g} | h {h} | f {f}'.format(idx = x.index, g = x.gval, h = x.hval, f = x.gval + x.hval))
ROW = 101
COL = 101
BLOCKED = -1
OPEN = 0



visited = np.zeros((ROW, COL), dtype=bool)  # boolean array of visited or not
neighbors = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # neighbors to visit during DFS


def is_valid(row, col):  # DFS Helper method
    global ROW
    global COL
    global visited
    # if cell is out of bounds
    if (row < 0 or col < 0 or row >= ROW or col >= COL):
        return False
    if visited[row][col]:
        return False
    return True






def DFS(row, col, grid):  # DFS on grid to mark blocked/unblocked cells
    global visited
    global neighbors
    stack_pairs = []
    stack_pairs.append([row, col])

    while len(stack_pairs) > 0:
        current = stack_pairs[-1]
        stack_pairs.pop()
        row = current[0]
        col = current[1]

        if (is_valid(row, col) == False):
            continue

        visited[row][col] = True

        # do the random make the cell blocked or unblocked here
        grid[row][col] = 0 if rand.random() < 0.7 else -1

        # push all adjacent cells randomly
        neighbors = np.random.permutation(neighbors)
        for i in range(4):
            adj_hor = row + neighbors[i][0]
            adj_vert = col + neighbors[i][1]
            stack_pairs.append([adj_hor, adj_vert])
    return grid



def set_grid(grid):
    rand_row = rand.randint(0, 100)
    rand_col = rand.randint(0, 100)
    grid = DFS(rand_row, rand_col, grid)
    return grid


def print_node_list(list):
    for n in list:
        print_node(n)

def get_path(start, goal)->List:
    print('get path')
    path = []
    curr = goal
    print_node(curr)
    print_node(curr.parent)
    print(len(path))
    while curr is not None and curr.index != start.index:
        path.insert(0, curr.index)
        curr = curr.parent
    return path

def print_state(state, counter,comment = ""):
    print("%s\n"%comment)
    print("Search : %d\n"%counter)
    for i in range(6):
        for j in range(6):
            if j < 5:
                print(" %2d"%state[i][j], end="")
            else:
                print(" %2d"%state[i][j])

def observe(curr, grid, maze):
    for move in neighbors:
        row = curr[0] + move[0]
        col = curr[1] + move[1]
        if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[row]):
            continue
        else:
            grid[row][col] = -1 if maze[row][col] == -1 else 0

def move_to(start, goal, path, state, maze):
    row = start[0] + path[0][0]
    col = start[1] + path[1][1]

    path.pop()
    curr = start
    # print('start',end="")
    # print(curr)
    # print_state(state, 1)


    for move in path:
        row = move[0]
        col = move[1]
        
        if state[row][col] == -1:
            return (curr[0],curr[1])
        else:
            curr = (row, col)
            observe(curr, state, maze)

    # print(curr)
    return curr



def get_next(pos, grid, goal_idx, counter):
    next_list = []
    print('curr pos {}'.format(pos))

    for move in neighbors:
        row = pos[0] + move[0]
        col = pos[1] + move[1]
        print('({x},{y})'.format(x=row,y=col))
        if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[row])\
          or grid[row][col] == -1 :
            print('out bound')
            continue

        if grid[row][col] == counter + 1:
            print('visit')
            continue
        next_list.append(n.node((row, col), goal_idx, gval=math.inf))

    for p in next_list:
        print('({x},{y})'.format(x=p.index[0],y=p.index[1]),end="")
    
    return next_list




def print_node(node):
    print('index : {idx} | g {g} | h {h} | f {f}'.format(idx = node.index, g = node.gval, h = node.hval, f = node.gval + node.hval))

def compute_path(known_grid, start_node, goal_node, openlist, counter):
    # print_node(openlist[0])
    # print(len(openlist))
    
    goal = goal_node
    k = 0


    while len(openlist) > 0 and goal.gval > openlist[0].get_f():
        k += 1
        if k > 20:
            break
        print('\ncurr top    ',end=":")
        curr = heap.heappop(openlist)
        print_node(curr)
        print('curr goal     ',end=":")
        print_node(goal)


            # print_state(known_grid,0)
            
        # print('min curr',end="")
        # print_node(curr)
        # print('open list size {}'.format(len(openlist)))
        actions = get_next(curr.index, known_grid, goal_node.index, counter)
        # print_state(known_grid,0)
        # print('valid action {}'.format(len(actions)))

        for next_node in actions:
            row = next_node.index[0]
            col = next_node.index[1]
            
            if next_node.index == goal_node.index:
                goal = next_node
                print('Goal node update',end="")
                print_node(goal)
                
            if known_grid[row][col] < counter:
                next_node.gval = math.inf
                known_grid[row][col] = counter

            
            if (next_node.gval) > (curr.gval + 1):
                next_node.gval = curr.gval + 1
                next_node.parent = curr

                if next_node in openlist:
                    print('remove',end="")
                    # print_node(next_node)
                    openlist.remove(next_node)
                # print("push  ",end = "")
                # print_node(next_node)
                heap.heappush(openlist, next_node)
                # print("top    ",end = "")
                # print

    if len(openlist) == 0:
        return None
    # print("top    ",end = "")
    # print_node(openlist[0])
    # goal.parent = curr
    # print_node(goal)
    path = get_path(start_node, curr)
    
    for p in path:
            print('({x},{y})->'.format(x=p[0],y=p[1]),end="")
    
    return path

    # get_path()
        
        # print(len(openlist))
    print('currnode',end="")
    print_node(curr)
    print('opentopnode',end="")
    print_node(openlist[0])

    


    


                
    

    # for node in openlist:
    #     print(node.index)
    # FINISH LATER





# MAIN
maze = np.array([[0, 0, 0, 0, 0, 0],
                 [0, -1, -1, -1, 0, 0],
                 [0, 0, -1, 0, -1, 0],
                 [-1, 0, 0, 0, -1, 0],
                 [-1, -1, 0, -1, -1, 0],
                 [0, 0, 0, -1, 0, 0]])
counter = 0
state_grid = np.zeros((6, 6), dtype=np.int64)
print_state(maze, 0, "Maze : ")
start = (1, 0)
print("observe")
observe(start, state_grid, maze)

print_state(state_grid, counter)
goal_idx = (5, 4)
goal = n.node((5, 4), goal_idx, gval=math.inf)
curr = n.node(start, goal_idx, gval=0)
i = 0
while i < 15 and curr.index != goal.index:
    i +=  1
    print_state(state_grid, counter)
    counter += 1
    curr.gval = 0
    state_grid[curr.index[0]][curr.index[1]] = counter
    goal.gval = math.inf
    state_grid[goal_idx[0]][goal_idx[1]] = counter
    openlist = []
    # MUST COMPUTE THINGS HERE
    heap.heappush(openlist, curr)
    path = compute_path(state_grid, curr, goal, openlist, counter)
    if path is None:
        print("I cannot reach the target")
        break
    else:
        # for p in path:
        #     print('({x},{y})->'.format(x=p[0],y=p[1]),end="")
        curr.index = move_to(curr.index, goal.index, path, state_grid, maze)
        print('agent current position:({x},{y})'.format(x=curr.index[0], y=curr.index[1]))

    



openlist = []
heap.heappush(openlist, curr)
path = compute_path(maze, curr, goal, openlist, 1)

if path is not None:
    for p in path:
        print(p)
    curr.index  = move_to(curr.index, goal.index, path, state_grid, maze)

print_node(curr)


# print_node(curr)
# print_node(goal)





# Visualizations
#fig =pyplot.figure(figsize=(30,30))
#colormap = colors.ListedColormap(["darkblue","lightgreen","Red","yellow"])
#im=pyplot.imshow(grid1,colormap, animated=True)
#red_patch = mpatches.Patch(color='red', label='Agent')
#yellow_patch = mpatches.Patch(color='yellow', label='Target')
#green_patch = mpatches.Patch(color='lightgreen', label= "Open and Unblocked")
#blue_patch = mpatches.Patch(color= "darkblue", label = "blocked")
#pyplot.legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), handles=[red_patch,green_patch,blue_patch,yellow_patch])

# create movie array which captures all of the changes in the agent, and shwo the movie from the code in pygameimp.py
# pyplot.show()
