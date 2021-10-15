import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import random
import pyxel




from typing import List
from matplotlib import animation
import numpy as np
from numpy.lib.type_check import asfarray
import pandas as pd
import random
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import math
import heapq as heap
import pyxel

NEIGHBORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

state = []
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

class State:
    def __init__(self, counter, prev_pos) -> None:
        self.counter = counter
        self.prev = prev_pos
        self.assume_path = []
        self.actual_move = []
        self.known_world_update = []
        self.after = []




class Maze:
    def __init__(self, maze) -> None:
        self.maze = maze
        self.len = len(maze)
        self.width = len(maze[0])
        self.state = []
        self.start = []
        self.end = []
        # self.soving_given_pos((4,4),(10,10))
        

    def __get_rand_pos(self):
        pos =  (random.randint(0,len(self.maze)-1),random.randint(0,len(self.maze)-1))
    
        while not self.__is_valid(pos):
            pos = (random.randint(0,len(self.maze)-1),random.randint(0,len(self.maze)-1))
        return pos

    def soving_given_pos(self, start, end):
        self.start = start
        self.end = end
        state_grid = np.zeros((self.len,self.width), dtype=np.int64)
        goal = Node(end, end, gval=math.inf)
        curr = Node(start, end, gval=0)
        arrive = True
        counter = 0
        state = State(counter, curr.index)
        self.__observe(start, state_grid, self.maze, state)
        self.state.append(state)

        while curr.index != goal.index:
            # print_state(state_grid, counter)
            
            counter += 1
            state = State(counter, curr.index)
            self.state.append(state)
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
                state.assume_path = path
                curr.index = self.__move_to(curr.index, goal.index, path, state_grid, self.maze, state)
                print('agent current position:', curr.index)

        if arrive:
            print('Agent arrived!!')
            return self.state
        else:
            print('I cannot reach the target',curr.index)

        return self.state

    def solve_rand_pos(self):
        start = self.__get_rand_pos()
        end = self.__get_rand_pos()
        self.start = start
        self.end = end

        # print('Agent is at',start)
        # print('Target is at',end)
        state_grid = np.zeros((self.len,self.width), dtype=np.int64)
        # print_state(maze, "Maze: ")
        # start = (4, 4)
        # print("observe")
        # observe(start, state_grid, maze)
        # print_state(state_grid, counter)
        state_grid = np.zeros((self.len,self.width), dtype=np.int64)
        goal = Node(end, end, gval=math.inf)
        curr = Node(start, end, gval=0)
        arrive = True
        counter = 0
        state = State(counter, curr.index)
        self.__observe(start, state_grid, self.maze, state)
        self.state.append(state)

        while curr.index != goal.index:
            # print_state(state_grid, counter)
            
            counter += 1
            state = State(counter, curr.index)
            self.state.append(state)
            curr.gval = 0
            state_grid[curr.index[0]][curr.index[1]] = counter
            goal.gval = math.inf
            state_grid[goal.index[0]][goal.index[1]] = counter
            openlist = []
            # MUST COMPUTE THINGS HERE
            heap.heappush(openlist, curr)
            path = self.__compute_path(state_grid, curr, goal, openlist, counter)
            if path is None:
                # print('Agent stopped at',curr.index)
                arrive = False
                break
            else:
                state.assume_path = path
                curr.index = self.__move_to(curr.index, goal.index, path, state_grid, self.maze, state)
                # print('agent current position:', curr.index)

        if arrive:
            print('Agent arrived!!')
            return self.state
        else:
            print('I cannot reach the target',curr.index)

        return self.state


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

    def print_state(state, comment=""):
        for i in range(len(state)):
            for j in range(len(state)):
                if j < (len(state)-1):
                    print(" %2d" % state[i][j], end="")
                else:
                    print(" %2d" % state[i][j])

    def __observe(self, curr, grid, maze, state_record):
        for move in NEIGHBORS:
            row = curr[0] + move[0]
            col = curr[1] + move[1]
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[row]):
                continue
            if maze[row][col] == -1:
                grid[row][col] = -1
                state_record.known_world_update.append((row,col))  
    

    def __move_to(self, start, goal, path, state, maze, state_record):
        print('\nBefore move: Agent is at',start)
        curr = start
        # print('start',end="")
        # print(curr)
        # print_state(state, 1)

        for move in path:
            row = move[0]
            col = move[1]
            self.__observe(curr, state, maze, state_record)

            if state[row][col] == -1:
                state_record.after = (curr[0], curr[1])
                return (curr[0], curr[1])
            else:
                curr = (row, col)
                state_record.actual_move.append(curr)
        
        state_record.after = curr

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



maze1 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
                 [ 0, 0, 0, 0, 0,-1,-1,-1,-1,-1, 0], #1
                 [ 0, 0,-1, 0,-1,-1, 0,-1, 0,-1, 0], #2
                 [ 0, 0,-1, 0,-1,-1, 0, 0, 0,-1, 0], #3
                 [-1,-1, 0, 0, 0, 0, 0, 0,-1,-1, 0], #4
                 [ 0, 0, 0,-1,-1,-1,-1, 0,-1,-1, 0], #5
                 [ 0,-1,-1,-1, 0, 0,-1, 0,-1,-1, 0], #6
                 [ 0, 0,-1,-1, 0, 0, 0, 0,-1,-1, 0], #7
                 [-1, 0,-1, 0, 0,-1,-1, 0,-1,-1, 0], #8
                 [-1, 0,-1, 0, 0, 0,-1,-1, 0,-1, 0], #9
                 [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0]]) #10
m1 = Maze(maze1)
m1.soving_given_pos((4,4),(10,10))


def grid_str(grid):
    for i in range(len(grid)):
        for j in range(len(grid)):
            if j < (len(grid)-1):
                print(" %2d" % grid[i][j], end="")
            else:
                print(" %2d" % grid[i][j])

    


known_world = np.zeros((11,11), dtype=np.int64)
count = 0
for st in m1.state:
    for pos in st.known_world_update:
        known_world[pos[0]][pos[1]] = -1
    count+=1
    print('Count:{a}\nAgent: {x}\nAssume:{y}\nActual:{k}\nAfter move:{z}'.format(a=count,x=st.prev, y=st.assume_path, z=st.after, k=st.actual_move))

    grid_str(known_world)
    print()

# -------------------
pixel = 5
width, height = 11, 11
road_color, wall_color = 7, 13
start_point_color, end_point_color, = 11, 11
head_color, route_color, backtrack_color = 9, 11, 8

class App:
    def __init__(self):
        #pyxel.init(width * pixel, height * pixel, caption='maze', border_width=10, border_color=0xFFFFFF)
        pyxel.init(width * pixel, height * pixel)
        self.maze = maze1
        self.maze_obj = Maze(maze1)
        self.vis_state = self.maze_obj.soving_given_pos((4,4),(10,10))
        self.beg_end = [self.maze_obj.start,self.maze_obj.end]
        self.death = True
        self.start = False
        self.show_maze = True
        self.death = False
        self.curr_assume_path = []
        self.drawing_path = []
        self.agent_pos = self.maze_obj.start
        self.tar_pos = self.maze_obj.end
        self.known_world = np.zeros((height,width), dtype=np.int64)
        self.step = 2
        self.next_search = 0
        self.index = 0
        self.route = []
        self.color = start_point_color
        self.current_graph = []
        # self.bfs_route = my_maze.bfs_route()
        # self.dfs_route = my_maze.dfs_route()
        self.dfs_model = True
        pyxel.run(self.update, self.draw)

    def update(self):
        if pyxel.btn(pyxel.KEY_Q):
            pyxel.quit()

        if pyxel.btn(pyxel.KEY_S):
            self.start = True
            self.show_maze = False

        if pyxel.btn(pyxel.KEY_M):
            self.show_maze = True

        if pyxel.btn(pyxel.KEY_N) and self.next_search < len(self.vis_state)-1:
            self.next_search += 1
            self.curr_assume_path = self.vis_state[self.next_search].assume_path
            self.drawing_path = []
            self.index = 0

        if self.start and not self.death and self.next_search > 0:
            self.check_death()
            self.update_agent()
            self.update_route()
            self.update_world()
            # print(self.curr_assume_path)
            # for i in range(len(self.known_world)):
            #     for j in range(len(self.known_world)):
            #         if j < (len(self.known_world)-1):
            #             print(" %2d" % self.known_world[i][j], end="")
            #         else:
            #             print(" %2d" % self.known_world[i][j])


        
        # print(self.next_search)
        # print(self.curr_assume_path)
        # print(self.drawing_path)

    def check_death(self):
        if self.next_search == len(self.vis_state):
            self.death = True  

    def draw(self):



        # draw maze
        if self.show_maze:
            maze = self.maze
            if not self.start:
                for x in range(height):
                    for y in range(width):
                        color = road_color if maze[x][y] == 0 else wall_color
                        pyxel.rect(y * pixel, x * pixel, pixel, pixel, color)



        if self.start and self.next_search == 0:
            for x in range(height):
                    for y in range(width):
                        color = road_color if self.known_world[x][y] == 0 else wall_color
                        pyxel.rect(y * pixel, x * pixel, pixel, pixel, color)
        

        
        if self.next_search > 0 and self.next_search < len(self.vis_state):
            for x in range(height):
                    for y in range(width):
                        color = road_color if self.known_world[x][y] == 0 else wall_color
                        pyxel.rect(y * pixel, x * pixel, pixel, pixel, color)

            # pyxel.rect(agent_pos[1] * pixel, agent_pos[0] * pixel, pixel, pixel, start_point_color)
            # pyxel.rect(target_pos[1] * pixel, target_pos[0] * pixel, pixel, pixel, start_point_color)

            # self.draw_path()
            if self.index > 0:
                offset = pixel / 2
                for i in range(len(self.drawing_path) - 1):
                    curr = self.drawing_path[i]
                    next = self.drawing_path[i + 1]
                    pyxel.line(curr[1] + offset, (curr[0] + offset), next[1] + offset, next[0] + offset, 6)
                    # q
                pyxel.circ(self.drawing_path[-1][1] + 2, self.drawing_path[-1][0] + 2, 1, head_color)
        

        pyxel.rect(self.agent_pos[1] * pixel, self.agent_pos[0] * pixel, pixel, pixel, start_point_color)
        pyxel.rect(self.tar_pos[1] * pixel, self.tar_pos[0] * pixel, pixel, pixel, start_point_color)
     
    def update_agent(self):
        if self.next_search < len(self.vis_state):
         self.agent_pos = self.vis_state[self.next_search].prev

    def update_route(self):
        index = int(self.index / self.step)
        print('{x}  drawing_path{y}   assumpath {z}'.format(x=index,y=len(self.drawing_path),z=len(self.curr_assume_path)))
        print(self.next_search,end="")
        # print(self.curr_assume_path)
        self.index += 1
        if index == len(self.drawing_path) and index < len(self.curr_assume_path) and len(self.curr_assume_path) > 0:  # moves
            # print(self.drawing_path)
            self.drawing_path.append([pixel * self.curr_assume_path[index][0], pixel * self.curr_assume_path[index][1]])
    
    def update_world(self):
        for pos in self.vis_state[self.next_search-1].known_world_update:
            self.known_world[pos[0]][pos[1]] = -1
        
        # for i in range(len(self.known_world)):
        #     for j in range(len(self.known_world)):
        #         if j < (len(self.known_world)-1):q
        #             print(" %2d" % self.known_world[i][j], end="")
        #         else:
        #             print(" %2d" % self.known_world[i][j])
        # print()



App()
