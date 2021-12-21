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
import random
import time
from typing import List
import math
import heapq as heap
import pyxel

NEIGHBORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


maze1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                 [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],  # 1
                 [0, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0],  # 2
                 [0, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0],  # 3
                 [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0],  # 4
                 [0, 0, 0, -1, -1, -1, -1, 0, -1, -1, 0],  # 5
                 [0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0],  # 6
                 [0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0],  # 7
                 [-1, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0],  # 8
                 [-1, 0, -1, 0, 0, 0, -1, -1, 0, -1, 0],  # 9
                 [0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0]])  # 10

# maze1 = np.zeros((ROW,COL),dtype=np.int64)
state = []


class Node:
    def __init__(self, index, goal_index, hval, gval=0, parent=None, contant_c=101 * 101) -> None:
        self.index = index
        self.goal_index = goal_index
        self.hval = hval
        self.gval = gval
        self.parent = parent
        self.c = contant_c

    def get_f(self):
        return self.gval + self.hval

    def __lt__(self, other):
        f_self = self.gval + self.hval
        f_other = other.gval + other.hval
        if f_self == f_other:
            return (self.c * f_self - self.gval) < (self.c * f_other - other.gval)
        return f_self < f_other

    def toStr(self):
        return ('index : {idx} | g {g} | h {h} | f {f}'.format(idx=self.index, h=self.hval, f=self.gval + self.hval))


class State:
    def __init__(self, counter, prev_pos, expand=0) -> None:
        self.counter = counter
        self.prev = prev_pos
        self.assume_path = []
        self.actual_move = []
        self.known_world_update = []
        self.after = []
        self.dead = False
        self.expand_count = expand


FORWARD = "FORWARD"
BACKWARD = "BACKWARD"
ADATIVE = "ADATIVE"


class Maze:
    def __init__(self, maze=None, rows=101, cols=101) -> None:
        self.maze = maze if maze is not None else Maze.generate_maze(
            rows, cols)
        self.rows = len(maze) if maze is not None else rows
        self.cols = len(maze[0]) if maze is not None else cols
        self.record = []
        self.start = []
        self.end = []
        self.man_dist = lambda x, y: abs(x[0]-y[0])+abs(x[1]-y[1])
        self.method = FORWARD

    def generate_maze(rows=101, cols=101):
        rand_maze = np.zeros((rows, cols), dtype=np.int64)
        row, col = 0, 0
        # boolean array of visited or not
        visited = np.zeros((rows, cols), dtype=np.int64)
        stack_pairs = []
        stack_pairs.append([row, col])

        while len(stack_pairs) > 0:
            current = stack_pairs[-1]
            stack_pairs.pop()
            row = current[0]
            col = current[1]

            if not Maze.is_valid(current, visited, rows, cols):
                continue

            visited[row][col] = -1

            # do the random make the cell blocked or unblocked here
            rand_maze[row][col] = 0 if random.randint(
                0, 100) / 100.0 < 0.7 else -1

            # push all adjacent cells randomly
            neighbors = np.random.permutation(NEIGHBORS)
            for i in range(4):
                adj_hor = row + neighbors[i][0]
                adj_vert = col + neighbors[i][1]
                stack_pairs.append([adj_hor, adj_vert])

        return rand_maze

    def is_valid(pos, grid, rows, cols):
        row = pos[0]
        col = pos[1]
        # if cell is out of bounds
        if (row < 0 or col < 0 or row >= rows or col >= cols):
            return False

        if grid[row][col] == -1:
            return False
        # return true otherwise.
        return True

    def get_rand_pos(self):
        pos = random.randint(
            0, self.rows - 1), random.randint(0, self.cols - 1)

        while not Maze.is_valid(pos, self.maze, self.rows, self.cols):
            pos = random.randint(
                0, self.rows - 1), random.randint(0, self.cols - 1)
        return pos

    def a_star_fw(self, start=None, end=None):
        self.method = FORWARD
        self.start = start if start is not None else self.get_rand_pos()
        self.end = end if end is not None else self.get_rand_pos()

        state_grid = np.zeros((self.rows, self.cols), dtype=np.int64)
        goal = Node(self.end, self.end, hval=self.man_dist(
            self.end, self.end), gval=math.inf)
        curr = Node(self.start, self.end, hval=self.man_dist(
            self.start, self.end), gval=0)
        arrive = True
        counter = 0
        state = State(counter, curr.index)
        self.__observe(self.start, state_grid, self.maze, state)
        self.record.append(state)

        while curr.index != goal.index:
            counter += 1
            state = State(counter, curr.index,
                          expand=self.record[counter-1].expand_count)
            self.record.append(state)
            curr.gval = 0
            state_grid[curr.index[0]][curr.index[1]] = counter
            goal.gval = math.inf
            state_grid[goal.index[0]][goal.index[1]] = counter
            openlist = []
            heap.heappush(openlist, curr)
            path = self.__compute_path(
                state_grid, curr, goal, openlist, counter, state)
            if path is None:
                arrive = False
                break
            else:
                state.assume_path = path
                curr.index = self.__move_to(
                    curr.index, goal.index, path, state_grid, self.maze, state)


        if not arrive:
            state.dead = True

        return self.record

    def a_star_adative(self, start=None, end=None):
        self.method = ADATIVE
        self.start = start if start is not None else self.get_rand_pos()
        self.end = end if end is not None else self.get_rand_pos()
        state_grid = np.zeros((self.rows, self.cols), dtype=np.int64)
        history = np.zeros((self.rows, self.cols), dtype=np.int64)
        goal = Node(self.end, end, hval=self.man_dist(
            self.end, self.end), gval=math.inf)
        curr = Node(self.start, end, hval=self.man_dist(
            self.start, self.end), gval=0)
        arrive = True
        counter = 0
        state = State(counter, curr.index)
        self.__observe(self.start, state_grid, self.maze, state)
        self.record.append(state)

        while curr.index != goal.index:
            counter += 1
            state = State(counter, curr.index,
                          expand=self.record[counter-1].expand_count)
            self.record.append(state)
            curr.gval = 0
            if counter > 1:
                curr.hval = history[curr.index[0]][curr.index[1]]
            state_grid[curr.index[0]][curr.index[1]] = counter
            goal.gval = math.inf
            state_grid[goal.index[0]][goal.index[1]] = counter
            openlist = []
            heap.heappush(openlist, curr)
            path = self.__compute_path2(
                state_grid, curr, goal, openlist, counter, state, history)

            if path is None:
                arrive = False
                break
            else:
                state.assume_path = path
                curr.index = self.__move_to(
                    curr.index, goal.index, path, state_grid, self.maze, state)

        if not arrive:
            state.dead = True

        return self.record

    def a_star_bw(self, start=None, end=None):
        self.method = BACKWARD
        self.start = start if start is not None else self.get_rand_pos()
        self.end = end if end is not None else self.get_rand_pos()

        state_grid = np.zeros((self.rows, self.cols), dtype=np.int64)
        goal = Node(self.end, self.end, hval=self.man_dist(
            self.end, self.end), gval=math.inf)
        curr = Node(self.start, self.end, hval=self.man_dist(
            self.start, self.end), gval=0)
        arrive = True
        counter = 0
        state = State(counter, curr.index)
        self.__observe(self.start, state_grid, self.maze, state)
        self.record.append(state)

        while curr.index != goal.index:
            counter += 1
            state = State(counter, curr.index,
                          expand=self.record[counter-1].expand_count)
            self.record.append(state)
            curr.gval = math.inf
            state_grid[curr.index[0]][curr.index[1]] = counter
            goal.gval = 0
            state_grid[goal.index[0]][goal.index[1]] = counter
            openlist = []
            heap.heappush(openlist, goal)
            path = self.__compute_path(
                state_grid, goal, curr, openlist, counter, state)
            if path is None:
                arrive = False
                break
            else:
                path.reverse()
                state.assume_path = path
                state.assume_path.insert(0, curr.index)
                curr.index = self.__move_to(
                    curr.index, goal.index, path, state_grid, self.maze, state)

        if not arrive:
            state.dead = True
        return self.record

    def __get_path(self, start, goal) -> List:
        path = []
        curr = goal
        while curr is not None and curr.index != start.index:
            path.insert(0, curr.index)
            curr = curr.parent

        path.insert(0, start.index)

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
                state_record.known_world_update.append((row, col))

    def __move_to(self, start, goal, path, state, maze, state_record):
        curr = start
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
        return curr

    def __compute_path(self, known_grid, start_node, goal_node, openlist, counter, state):
        while len(openlist) > 0 and goal_node.gval > openlist[0].get_f():
            curr = heap.heappop(openlist)
            state.expand_count += 1
            actions = self.__get_next(
                curr.index, known_grid, goal_node.index, counter)

            for next_node in actions:
                row = next_node.index[0]
                col = next_node.index[1]

                if next_node.index == goal_node.index:
                    goal_node = next_node

                if known_grid[row][col] < counter:
                    next_node.gval = math.inf
                    known_grid[row][col] = counter

                if (next_node.gval) > (curr.gval + 1):
                    next_node.gval = curr.gval + 1
                    next_node.parent = curr

                    if next_node in openlist:
                        openlist.remove(next_node)

                    heap.heappush(openlist, next_node)
        if len(openlist) == 0:
            return None

        goal_node.parent = curr
        curr = goal_node
        path = self.__get_path(start_node, curr)
        return path

    def __compute_path2(self, known_grid, start_node, goal_node, openlist, counter, state, history=[]):
        count = 0
        list = []
        while len(openlist) > 0 and goal_node.gval > openlist[0].get_f():
            count += 1
            curr = heap.heappop(openlist)
            list.append(curr)
            state.expand_count += 1
            actions = self.__get_next(
                curr.index, known_grid, goal_node.index, counter, history)

            for next_node in actions:
                row = next_node.index[0]
                col = next_node.index[1]

                if next_node.index == goal_node.index:
                    goal_node = next_node

                if known_grid[row][col] < counter:
                    next_node.gval = math.inf
                    known_grid[row][col] = counter

                if (next_node.gval) > (curr.gval + 1):
                    next_node.gval = curr.gval + 1
                    next_node.parent = curr

                    if next_node in openlist:
                        openlist.remove(next_node)

                    heap.heappush(openlist, next_node)

        if len(openlist) == 0:
            return None

        goal_node.parent = curr
        curr = goal_node
        path = self.__get_path(start_node, curr)
        g_goal = len(path) - 1

        for node in list:
            row = node.index[0]
            col = node.index[1]
            history[row][col] = g_goal - node.gval

        return path

    def __get_next(self, pos, grid, goal_idx, counter, history=[]):
        next_list = []
        for move in NEIGHBORS:
            row = pos[0] + move[0]
            col = pos[1] + move[1]
            indx = (row, col)
            # print('({x},{y})'.format(x=row,y=col))
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[row])\
                    or grid[row][col] == -1:
                # print('out bound')
                continue

            if grid[row][col] == counter and (row, col) != goal_idx:
                # print('visit')
                continue
            if self.method == ADATIVE:
                hval = history[row][col]
                if hval == 0:
                    new_neihbor = Node(indx, goal_idx, hval=self.man_dist(
                        indx, goal_idx), gval=math.inf)
                else:
                    new_neihbor = Node(
                        indx, goal_idx, hval=hval, gval=math.inf)
            else:
                new_neihbor = Node(indx, goal_idx, hval=self.man_dist(
                    indx, goal_idx), gval=math.inf)
                # self.__print_node(new_neihbor)
            next_list.append(new_neihbor)

        # for p in next_list:
        #     print('({x},{y})'.format(x=p.index[0],y=p.index[1]),end="")

        return next_list

    # def __print_node(self, node):
    #     print('index : {idx} | g {g} | h {h} | f {f}'.format(
    #         idx=node.index, g=node.gval, h=node.hval, f=node.gval + node.hval))

    def print_maze(self):
        grid = self.maze
        for i in range(len(grid)):
            for j in range(len(grid)):
                if j < (len(grid)-1):
                    print(" %2d" % grid[i][j], end="")
                else:
                    print(" %2d" % grid[i][j])


def grid_str(grid):
    for i in range(len(grid)):
        for j in range(len(grid)):
            if j < (len(grid)-1):
                print(" %2d" % grid[i][j], end="")
            else:
                print(" %2d" % grid[i][j])



# -------------------
PIXEL = 3
COLS = 20
ROWS = 20
ROAD_COLOR, WALL_COLOR = 7, 13
ASSUME_COLOR, MOVES_COLOR = 6, 12
AGENT_COLOR, TARGET_COLOR= 9, 8



class App:
    def __init__(self):
        pyxel.init(COLS * PIXEL, ROWS * PIXEL, caption='maze')
        self.new_game()
        pyxel.run(self.update, self.draw)

    def restart(self):
        self.start = False
        self.show_maze = True
        self.death = False
        self.agent_start_move = False
        self.curr_assume_path = []
        self.drawing_path = []
        self.actual_moves = []
        self.drawing_moves = []
        self.total_moves = []
        self.agent_pos = self.start_pos
        self.tar_pos = self.goal_pos
        self.known_world = np.zeros((ROWS, COLS), dtype=np.int64)
        self.step = 2
        self.next_search = 0
        self.index = 0
        self.index2 = 0
        self.dfs_model = True

    def new_game(self):
        self.maze = Maze.generate_maze(20,20)
        self.maze_obj = Maze(self.maze)
        self.start_pos = self.maze_obj.get_rand_pos()
        self.goal_pos = self.maze_obj.get_rand_pos()
        self.vis_state = self.maze_obj.a_star_adative(self.start_pos, self.goal_pos)
        self.beg_end = [self.maze_obj.start, self.maze_obj.end]
        self.restart()

    def update(self):

        if pyxel.btn(pyxel.KEY_A):
            self.vis_state = self.maze_obj.a_star_adative(self.start_pos, self.goal_pos)
            self.restart()
            time.sleep(0.2)

        if pyxel.btn(pyxel.KEY_B):
            self.vis_state = self.maze_obj.a_star_bw(self.start_pos, self.goal_pos)
            self.restart()
            time.sleep(0.2)

        if pyxel.btn(pyxel.KEY_F):
            self.vis_state = self.maze_obj.a_star_fw(self.start_pos, self.goal_pos)
            self.restart()
            time.sleep(0.2)

        if pyxel.btn(pyxel.KEY_N):
            self.new_game()
            time.sleep(0.2)

        if pyxel.btn(pyxel.KEY_R):
            self.restart()

        if pyxel.btn(pyxel.KEY_Q):
            pyxel.quit()

        if pyxel.btn(pyxel.KEY_M):
            self.show_maze = not self.show_maze
            time.sleep(0.1)

        if pyxel.btn(pyxel.KEY_DOWN) and self.next_search < len(self.vis_state) - 1:
            self.show_maze = False
            self.agent_start_move = False
            self.next_search += 1
            self.curr_assume_path = self.vis_state[self.next_search].assume_path
            self.actual_moves = self.vis_state[self.next_search].actual_move
            self.drawing_path = []
            self.drawing_moves = []
            self.index = 0
            self.index2 = 0
            time.sleep(0.3)

        if self.next_search > 0:
            self.check_death()
            self.update_route()
            self.update_world()

    def check_death(self):
        if self.vis_state[self.next_search].dead:
            self.death = True

    def draw(self):
        # draw maze
        offset = PIXEL / 2
        if self.show_maze:
            maze = self.maze
            for x in range(ROWS):
                for y in range(COLS):
                    color = ROAD_COLOR if maze[x][y] == 0 else WALL_COLOR
                    pyxel.rect(y * PIXEL, x * PIXEL, PIXEL, PIXEL, color)





        # draw assuming path
        if not self.show_maze and self.next_search > 0 and self.next_search < len(self.vis_state):

            for x in range(ROWS):
                for y in range(COLS):
                    color = ROAD_COLOR if self.known_world[x][y] == 0 else WALL_COLOR
                    pyxel.rect(y * PIXEL, x * PIXEL, PIXEL, PIXEL, color)

            if self.index > 0:
                for i in range(len(self.drawing_path) - 1):
                    curr = self.drawing_path[i]
                    next = self.drawing_path[i + 1]
                    pyxel.line(curr[1] + offset, (curr[0] + offset),
                               next[1] + offset, next[0] + offset, ASSUME_COLOR)

            if self.index2 > 0 and self.agent_start_move:
                for i in range(len(self.drawing_moves) - 1):
                    curr = self.drawing_moves[i]
                    next = self.drawing_moves[i + 1]
                    pyxel.line(curr[1] + offset, (curr[0] + offset),
                               next[1] + offset, next[0] + offset, MOVES_COLOR)




        for pos in self.total_moves:
            pyxel.rect(pos[1] * PIXEL, pos[0]
                   * PIXEL, PIXEL, PIXEL, 12)



        pyxel.rect(self.tar_pos[1] * PIXEL, self.tar_pos[0]
                   * PIXEL, PIXEL, PIXEL, TARGET_COLOR)
        pyxel.rect(self.agent_pos[1] * PIXEL, self.agent_pos[0]
                   * PIXEL, PIXEL, PIXEL, AGENT_COLOR)




    def update_agent(self):
        if self.next_search < len(self.vis_state):
            self.agent_pos = self.vis_state[self.next_search].prev

    def update_route(self):
        index = int(self.index / self.step)

        self.index += 1
        if index == len(self.drawing_path) and index < len(self.curr_assume_path) and len(self.curr_assume_path) > 0:  # moves
            self.drawing_path.append(
                [PIXEL * self.curr_assume_path[index][0], PIXEL * self.curr_assume_path[index][1]])

        if index == len(self.curr_assume_path):
            self.agent_start_move = True
            self.index2 = 0

        index2 = int(self.index2 / self.step)
        if self.agent_start_move:
            self.index2 += 1
            if index2 == len(self.drawing_moves) and index2 < len(self.actual_moves) and len(self.actual_moves) > 0:  # moves
                self.drawing_moves.append(
                    [PIXEL * self.actual_moves[index2][0], PIXEL * self.actual_moves[index2][1]])

        if index2 == len(self.actual_moves):
            time.sleep(0.1)
            self.total_moves.extend(self.actual_moves)
            self.agent_pos = self.vis_state[self.next_search].after if not self.death else self.maze_obj.start
            for pos in self.vis_state[self.next_search].known_world_update:
                self.known_world[pos[0]][pos[1]] = -1




    def update_world(self):
        for pos in self.vis_state[self.next_search-1].known_world_update:
            self.known_world[pos[0]][pos[1]] = -1


def compare_method():
    count = 50
    fw_count, bw_count, ad_count = 0, 0, 0
    maze_list = []
    # generate 50 mazes
    while count > 0:
        rand_m = Maze.generate_maze()
        m = Maze(rand_m)
        print(f"Maze Generating Count Down: {count}")

        start = m.get_rand_pos()
        end = m.get_rand_pos()
        maze_list.append((m, start, end))
        count -= 1

    # get Foward time and count
    start_time = time.time()
    for maze in maze_list:
        m = maze[0]
        start = maze[1]
        end = maze[2]
        fw_count += m.a_star_fw(start,end).pop().expand_count
    fw_time = time.time()-start_time

    print(f"Repeated Forward A*:\n   - Number of expanded cells:{fw_count}\n   - Average expanded cells:{fw_count/50.0}\n   - Runtime:{fw_time}\n")


    start_time = time.time()
    for maze in maze_list:
        m = maze[0]
        start = maze[1]
        end = maze[2]
        bw_count += m.a_star_bw(start,end).pop().expand_count
    bw_time = time.time()-start_time
    print(f"Repeated Backward A*:\n   - Number of expanded cells:{bw_count}\n   - Average expanded cells:{bw_count/50.0}\n   - Runtime:{bw_time}\n")

    start_time = time.time()
    for maze in maze_list:
        m = maze[0]
        start = maze[1]
        end = maze[2]
        ad_count += m.a_star_adative(start,end).pop().expand_count
    ad_time = time.time()-start_time
    print(f"Adative A*:\n   - Number of expanded cells:{ad_count}\n   - Average expanded cells:{ad_count/50.0}\n   - Runtime:{ad_time}\n")

def get_report():
    rand_m = Maze.generate_maze(ROWS,COLS)
    m = Maze(rand_m)
    m.print_maze()

    start = m.get_rand_pos()
    end = m.get_rand_pos()
    print_report(m.a_star_adative(start,end),"Adative")
    print_report(m.a_star_fw(start,end),"Forward")
    print_report(m.a_star_bw(start,end),"Backward")

def print_report(report, method):
    print(f'==-=-=-=-=--=-=-=-=-={method}-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    total_moves = []
    known_world = np.zeros((ROWS, COLS), dtype=np.int64)
    for st in report:
        for pos in st.known_world_update:
            known_world[pos[0]][pos[1]] = -1
            total_moves.extend(st.actual_move)
    st = report.pop()
    print(f'{method}: Expanded Cell count', st.expand_count)
    # print(f'total moves{total_moves}')
    grid_str(known_world)

# App()


compare_method()
