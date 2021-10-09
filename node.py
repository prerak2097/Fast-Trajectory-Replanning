class node:
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