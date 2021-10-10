from typing import List
from matplotlib import animation
import numpy as np
from numpy.lib.type_check import asfarray
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
import random
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import math
import heapq as heap
import node as n

NEIGHBORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class Node:
    def __init__(self, index, goal_index, hval=0, gval=0, parent=None) -> None:
        self.index = index
        self.goal_index = goal_index
        self.hval = abs(goal_index[0]-self.index[0]) + \
            abs(goal_index[1]-self.index[1])
        self.gval = gval
        self.parent = parent

    def get_f(self):
        return self.gval + self.hval
    def __lt__(self, other):
        return self.gval+self.hval < other.gval + other.hval
    def toStr(self):
        return ('index : {idx} | g {g} | h {h} | f {f}'.format(idx = self.index, h = self.hval, f = self.gval + self.hval ))



class Maze:
    def __init__(self, maze) -> None:
        self.maze = maze
        self.len = len(maze)
        self.width = len(maze[0])
        pass

    def __get_rand_pos(self):
        pos =  (random.randint(0,len(self.maze)-1),random.randint(0,len(self.maze)-1))
    
        while not self.__is_valid(pos):
            pos = (random.randint(0,len(self.maze)-1),random.randint(0,len(self.maze)-1))
        return pos

    def soving_given_pos(self, start, end):
        state_grid = np.zeros((self.len,self.width), dtype=np.int64)
        # print_state(maze, "Maze: ")
        # start = (4, 4)
        # print("observe")
        # observe(start, state_grid, maze)
        # print_state(state_grid, counter)

        goal = Node(end, end, gval=math.inf)
        curr = Node(start, end, gval=0)
        arrive = True
        counter = 0

        while curr.index != goal.index:
            # print_state(state_grid, counter)
            counter += 1
            curr.gval = 0
            state_grid[curr.index[0]][curr.index[1]] = counter
            goal.gval = math.inf
            state_grid[goal.index[0]][goal.index[1]] = counter
            openlist = []
            # MUST COMPUTE THINGS HERE
            heap.heappush(openlist, curr)
            path = self.__compute_path(state_grid, curr, goal, openlist, counter)
            if path is None:
                print('Agent stopped at',curr.index)
                arrive = False
                break
            else:
                curr.index = self.__move_to(curr.index, goal.index, path, state_grid, self.maze)
                print('agent current position:', curr.index)

        if arrive:
            print('Agent arrived!!')
        else:
            print('I cannot reach the target',curr.index)

    def solve_rand_pos(self):
        start = self.__get_rand_pos()
        end = self.__get_rand_pos()

        print('Agent is at',start)
        print('Target is at',end)
        state_grid = np.zeros((self.len,self.width), dtype=np.int64)
        # print_state(maze, "Maze: ")
        # start = (4, 4)
        # print("observe")
        # observe(start, state_grid, maze)
        # print_state(state_grid, counter)

        goal = Node(end, end, gval=math.inf)
        curr = Node(start, end, gval=0)
        arrive = True
        counter = 0

        while curr.index != goal.index:
            # print_state(state_grid, counter)
            counter += 1
            curr.gval = 0
            state_grid[curr.index[0]][curr.index[1]] = counter
            goal.gval = math.inf
            state_grid[goal.index[0]][goal.index[1]] = counter
            openlist = []
            # MUST COMPUTE THINGS HERE
            heap.heappush(openlist, curr)
            path = self.__compute_path(state_grid, curr, goal, openlist, counter)
            if path is None:
                print('Agent stopped at',curr.index)
                arrive = False
                break
            else:
                curr.index = self.__move_to(curr.index, goal.index, path, state_grid, self.maze)
                print('agent current position:', curr.index)

        if arrive:
            print('Agent arrived!!')
        else:
            print('I cannot reach the target',curr.index)

    def str(self):
        s = ''
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                s += '{:>3}'.format(self.maze[i][j], end="") 
            s += '\n'
        return s



    def __is_valid(self, pos):
        maze = self.maze
        if maze[pos[0]][pos[1]] == -1:
            return False
        else:
            return True

    def __get_path(self, start, goal) -> List:
        path = []
        curr = goal
        while curr is not None and curr.index != start.index:
            path.insert(0, curr.index)
            curr = curr.parent

        return path

    def __print_state(self, state, comment=""):
        print('State:')
        for i in range(len(state)):
            for j in range(len(state)):
                if j < (len(state)-1):
                    print(" %2d" % state[i][j], end="")
                else:
                    print(" %2d" % state[i][j])

    def __observe(self, curr, grid, maze):
        for move in NEIGHBORS:
            row = curr[0] + move[0]
            col = curr[1] + move[1]
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[row]):
                continue
            else:
                grid[row][col] = -1 if maze[row][col] == -1 else grid[row][col]
    

    def __move_to(self, start, goal, path, state, maze):
        print('\nBefore move: Agent is at',start)
        curr = start
        # print('start',end="")
        # print(curr)
        # print_state(state, 1)

        for move in path:
            row = move[0]
            col = move[1]

            if state[row][col] == -1:
                return (curr[0], curr[1])
            else:
                curr = (row, col)
                self.__observe(curr, state, maze)

        print('Agent move to',curr)
        return curr


    def __compute_path(self, known_grid, start_node, goal_node, openlist, counter):

        while len(openlist) > 0 and goal_node.gval > openlist[0].get_f():

            curr = heap.heappop(openlist)
            actions = self.__get_next(curr.index, known_grid, goal_node.index, counter)
            # print_state(known_grid,0)
            # print('valid action {}'.format(len(actions)))

            for next_node in actions:
                row = next_node.index[0]
                col = next_node.index[1]

                if next_node.index == goal_node.index:
                    goal_node = next_node
                    # print('Goal node update',end="")
                    # print_node(goal_node)

                if known_grid[row][col] < counter:
                    next_node.gval = math.inf
                    known_grid[row][col] = counter

                if (next_node.gval) > (curr.gval + 1):
                    next_node.gval = curr.gval + 1
                    next_node.parent = curr

                    if next_node in openlist:
                        openlist.remove(next_node)
                    # print("push  ",end = "")
                    # print_node(next_node)
                    heap.heappush(openlist, next_node)
                    # print("top    ",end = "")
                # print

        if len(openlist) == 0:
            return None
        # print("after while loop    ", end="")
        # print_node(openlist[0])
        goal_node.parent = curr
        curr = goal_node
        path = self.__get_path(start_node, curr)

        for p in path:
            print('->({x},{y})'.format(x=p[0], y=p[1]), end="")
        print()
        return path

    def __get_next(self, pos, grid, goal_idx, counter):
        next_list = []

        for move in NEIGHBORS:
            row = pos[0] + move[0]
            col = pos[1] + move[1]
            # print('({x},{y})'.format(x=row,y=col))
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[row])\
                    or grid[row][col] == -1:
                # print('out bound')
                continue

            if grid[row][col] == counter and (row, col) != goal_idx:
                # print('visit')
                continue
            new_neihbor = Node((row, col), goal_idx,gval=math.inf)
            next_list.append(new_neihbor)

        # for p in next_list:
        #     print('({x},{y})'.format(x=p.index[0],y=p.index[1]),end="")

        return next_list

    def __print_node(node):
        print('index : {idx} | g {g} | h {h} | f {f}'.format(
        idx=node.index, g=node.gval, h=node.hval, f=node.gval + node.hval))





# MAIN
# maze = np.array([[0, 0, 0, 0, 0, 0],
#                  [0, -1, -1, -1, 0, 0],
#                  [0, 0, -1, 0, -1, 0],
#                  [-1, 0, 0, 0, -1, 0],
#                  [-1, -1, 0, -1, -1, 0],
#                  [0, 0, 0, -1, 0, 0]])


#                 0  1  2  3  4  5  6  7  8  9  10
maze1 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
                 [ 0, 0, 0, 0, 0,-1,-1,-1,-1, 0, 0], #1
                 [ 0, 0,-1, 0,-1,-1, 0,-1, 0,-1, 0], #2
                 [ 0, 0,-1, 0,-1,-1, 0, 0, 0,-1, 0], #3
                 [-1,-1, 0, 0, 0, 0, 0, 0,-1,-1, 0], #4
                 [ 0, 0, 0,-1,-1,-1,-1, 0,-1,-1, 0], #5
                 [ 0,-1,-1,-1, 0, 0,-1, 0,-1,-1, 0], #6
                 [ 0, 0,-1,-1, 0, 0, 0, 0,-1,-1, 0], #7
                 [-1, 0,-1, 0, 0,-1,-1, 0,-1,-1,-1], #8
                 [-1, 0,-1, 0, 0, 0,-1,-1, 0,-1, 0], #9
                 [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0]]) #10

m1 = Maze(maze1)
print(m1.str())

# m1.soving_given_pos((4,4),(10,10))
m1.solve_rand_pos()




                                                   

