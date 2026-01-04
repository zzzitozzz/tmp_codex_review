import os
import copy

import mesa
import numpy as np
import pandas as pd
import math
import solara


class Human(mesa.Agent):
    def __init__(self, unique_id, model,
                 pos, space, pos_array=[],
                 in_goal=False, elapsed_time=1.,  # 経過時間
                 ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.space = space
        self.tmp_pos = np.array((0., 0.))
        self.in_goal = in_goal
        self.pos_array = pos_array
        self.elapsed_time = 1

    def step(self):  # 次の位置を特定するための計算式を書く
        if self.elapsed_time == len(self.pos_array):
            self.in_goal = True
            return None
        self.tmp_pos = self.pos_array[self.elapsed_time]
        return None

    def advance(self):
        self.pos = copy.deepcopy(self.tmp_pos)
        self.elapsed_time += 1
        if (self.in_goal):  # goalした場合
            self.model.space.remove_agent(self)
            self.model.schedule.remove(self)
            return None
        else:
            self.model.space.move_agent(self, self.pos)  # goalしていない場合
        return None


class ForcefulHuman(Human):
    def __init__(self, unique_id, model,
                 pos, space, pos_array=[],
                 in_goal=False, elapsed_time=1.,  # 経過時間
                 ):
        super().__init__(unique_id, model, pos, space, pos_array, in_goal, elapsed_time)

class Obstacle(mesa.Agent):
    def __init__(self, unique_id, model, pos, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dir = dir

    def step(self):
        return None


class Wall(mesa.Agent):
    def __init__(self, unique_id, model, pos, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dir = dir

    def step(self):
        return None
    
class Line(mesa.Agent):
    def __init__(self, unique_id, model, x1, y1, x2, y2):
        super().__init__(unique_id, model)
        self.start_pos = np.array((x1, y1))
        self.end_pos = np.array((x2, y2))

    def step(self):
        return None


class Goal(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)

    def step(self):
        return None
