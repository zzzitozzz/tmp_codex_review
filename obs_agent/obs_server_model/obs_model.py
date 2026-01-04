import mesa
import glob
import re
import copy
import os
import sys

import numpy as np
import pandas as pd
import pathlib

from .obs_agent import Human, ForcefulHuman, Obstacle, Goal, Wall, Line


class MoveAgent(mesa.Model):

    def __init__(self, nol_pop, for_pop, seed, wall_arr, width, height, add_file_name):
        mesa.Model.__init__(self)
        self.nol_pop = nol_pop
        self.for_pop = for_pop
        self.seed = seed
        self.wall_arr = wall_arr
        self.width = width
        self.height = height
        self.add_file_name = add_file_name
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.make_agents()
        self.running = True

    def make_agents(self):
        self.generate_human()
        self.generate_wall()

    def generate_human(self):
        self.generate_nol_humans()
        self.generate_for_humans()
        return None

    def generate_nol_humans(self):
        id = self.schedule.get_agent_count() + 1
        nol_file_name_array = glob.glob(f"{self.add_file_name}id*_normal.csv")
        for nol_file_name in nol_file_name_array:
            pos_array = pd.read_csv(nol_file_name, header = None).values
            pos = pos_array[0]
            human = Human(id, self, pos, self.space, pos_array)
            self.space.place_agent(human, pos)
            self.schedule.add(human)
            id += 1
        return None

    def generate_for_humans(self):
        id = self.schedule.get_agent_count() + 1
        for_file_name_array = glob.glob(
            f"{self.add_file_name}id*_forceful.csv")
        for for_file_name in for_file_name_array:
            pos_array = pd.read_csv(for_file_name, header = None).values
            pos = pos_array[0]
            for_human = ForcefulHuman(
                id, self, pos, self.space, pos_array)
            self.space.place_agent(for_human, pos)
            self.schedule.add(for_human)
            id += 1
        return None

    def generate_wall(self):
        id = self.schedule.get_agent_count() + 1
        i = 0
        wall_a = self.wall_arr[:, 0]           # 各壁の始点 (N_wall, 2)
        wall_b = self.wall_arr[:, 1]           # 各壁の終点 (N_wall, 2)
        while 1:
            start_point = wall_a[i]
            end_point = wall_b[i]
            wall = Line(id, self, start_point[0], start_point[1], end_point[0], end_point[1])
            self.space.place_agent(wall, (wall.start_pos+wall.end_pos)/2.)
            self.schedule.add(wall)
            i += 1
            id += 1
            if i >= len(wall_a):
                break
        return None

    # def generate_wall(self):  # 壁を作る
    #     i = 0
    #     while 1:
    #         if i >= len(self.wall_arr) - 1:
    #             # print("i: " + str(i) + " len_self.wall_arr:" +
    #             #   str(len(self.wall_arr)))
    #             break
    #         tmp_wall_1 = self.wall_arr[i]
    #         tmp_wall_2 = self.wall_arr[i + 1]
    #         print(f"{tmp_wall_1=}")
    #         print(f"{tmp_wall_2=}")
    #         tmp_x, tmp_y = tmp_wall_1[0], tmp_wall_1[1]
    #         # print("wall_1: " + str(tmp_wall_1) + " wall_2: " + str(tmp_wall_2))
    #         not_skip = 1
    #         if tmp_wall_1[2] == 0 or tmp_wall_1[2] == 2:
    #             while 1:
    #                 if tmp_y >= tmp_wall_2[1]:
    #                     break
    #                 if i != 0:
    #                     neighbors = self.space.get_neighbors(
    #                         np.array((27., 20.)), 30, False)
    #                     for neighbor in neighbors:
    #                         if type(neighbor) is Wall:
    #                             if tmp_x == neighbor.pos[0] and tmp_y == neighbor.pos[1]:
    #                                 not_skip = 0
    #                                 break
    #                 if not_skip:
    #                     self.generate_block(
    #                         tmp_x, tmp_y, self.next_id(), tmp_wall_1[2])
    #                     tmp_y += self.height * 0.001
    #                 else:
    #                     not_skip = 1
    #                     tmp_y += self.height * 0.001
    #         elif tmp_wall_1[2] == 1 or tmp_wall_1[2] == 3:
    #             while 1:
    #                 if tmp_x >= tmp_wall_2[0]:
    #                     break
    #                 if i != 0:
    #                     neighbors = self.space.get_neighbors(
    #                         np.array((27., 20.)), 30, False)
    #                     for neighbor in neighbors:
    #                         if type(neighbor) is Wall:
    #                             if tmp_x == neighbor.pos[0] and tmp_y == neighbor.pos[1]:
    #                                 not_skip = 0
    #                                 break
    #                 if not_skip:
    #                     self.generate_block(
    #                         tmp_x, tmp_y, self.next_id(), tmp_wall_1[2])
    #                     tmp_x += self.width * 0.001
    #                 else:
    #                     not_skip = 1
    #                     tmp_x += self.width * 0.001
    #         else:
    #             print("generate_wall_error")
    #             print(f"{tmp_wall_1=}")
    #             print(f"{tmp_wall_2=}")
    #         i += 2
    #     return None

    # def generate_block(self, x, y, id, cnt):  # 1つブロックを生成する
    #     pos = np.array((x, y))
    #     dir = cnt  # 1:左 2:上　3:右　4:下　の方向に避難者に力を与える
    #     wall = Wall(
    #         id,
    #         self,
    #         pos,
    #         dir,
    #     )
    #     self.space.place_agent(wall, pos)
    #     self.schedule.add(wall)

    def step(self):
        self.schedule.step()
