import sys
sys.path.append('/usr/local/lib/python3.11/site-packages')
import mesa
import numpy as np

from .obs_model import Human, ForcefulHuman, Obstacle, Goal, Wall, Line
from .obs_model import MoveAgent
from .obs_SimpleContinousModule import SimpleCanvas





def __draw(agent):
    if type(agent) is Human:
        # return {"Shape": "ellipse", "r": 0.5, "Filled": True, "Color": "Red"}
        return {"Shape": "ellipse", "r": 0.0025, "Filled": True, "Color": "Red"}
    if type(agent) is ForcefulHuman:
        # return {"Shape": "ellipse", "r": 0.5, "Filled": True, "Color": "blue"}
        return {"Shape": "ellipse", "r": 0.0025, "Filled":True, "Color": "blue"}
    if type(agent) is Obstacle:
        # 見かけの大きさに対する相対的な高さと幅
        return {"Shape": "rect", "w": 0.001, "h": 0.001, "Filled": True, "Color": "blue"}
    if type(agent) is Goal:
        # return {"Shape": "rect", "w":0.1, "h":0.1, "Filled":False, "Color":"Green"}
        return {"Shape": "circle", "r": 25, "Filled": False, "Color": "Green"}
    if type(agent) is Wall:
        return {"Shape": "rect", "w": 0.001, "h": 0.001, "Filled": True, "Color": "Gray"}
    if type(agent) is Line:
        return {"Shape": "line", "Color": "Black", "LineWidth": 1}
    return None

def convex_map():
    canvas_element = SimpleCanvas(__draw, 1500, 1500)  # キャンパスの大きさ height width
    width = 200
    height = 200
    wall_arr = np.array([[[4., 40., 1], [54., 40., 1]], #三叉路マップ
                [[4., 26., 3], [16., 26., 3]],
                [[22., 26., 3], [54., 26., 3]],
                [[16., 4., 2], [16., 26., 2]],
                [[22., 4., 0], [22., 26., 0]]])
    return canvas_element, width, height, wall_arr

def intersection_map():
    canvas_element = SimpleCanvas(__draw, 1500, 1500)  # キャンパスの大きさ height width
    width = 200
    height = 200
    wall_arr = np.array([[[2.0, 2.0], [156.0, 2.0]], [[156.0, 2.0], [156.0, 100.0]],
                        [[156.0, 100.0], [2.0, 100.0]], [[2.0, 100.0], [2.0, 2.0]], 
                        [[8.0, 8.0], [48.0, 8.0]], [[48.0, 8.0], [48.0, 38.0]], 
                        [[48.0, 38.0], [8.0, 38.0]], [[8.0, 38.0], [8.0, 8.0]], 
                        [[54.0, 8.0], [84.0, 8.0]], [[84.0, 8.0], [84.0, 38.0]], 
                        [[84.0, 38.0], [54.0, 38.0]], [[54.0, 38.0], [54.0, 8.0]], 
                        [[90.0, 8.0], [150.0, 8.0]], [[150.0, 8.0], [150.0, 38.0]], 
                        [[150.0, 38.0], [90.0, 38.0]], [[90.0, 38.0], [90.0, 8.0]], 
                        [[8.0, 44.0], [48.0, 44.0]], [[48.0, 44.0], [48.0, 94.0]], 
                        [[48.0, 94.0], [8.0, 94.0]], [[8.0, 94.0], [8.0, 44.0]], 
                        [[54.0, 44.0], [84.0, 44.0]], [[84.0, 44.0], [84.0, 94.0]], 
                        [[84.0, 94.0], [54.0, 94.0]], [[54.0, 94.0], [54.0, 44.0]], 
                        [[90.0, 44.0], [150.0, 44.0]], [[150.0, 44.0], [150.0, 94.0]], 
                        [[150.0, 94.0], [90.0, 94.0]], [[90.0, 94.0], [90.0, 44.0]],
                        [[118.0, 38.0], [118.0, 44.0]], [[122.0, 38.0], [122.0, 44.0]]
                        ]) 
    return canvas_element, width, height, wall_arr




# canvas_element = SimpleCanvas(__draw, 750, 750)  # キャンパスの大きさ
# canvas_element = SimpleCanvas(__draw, 1500, 1500)  # キャンパスの大きさ height width
# wall_arr = np.array([[[4., 40., 1], [54., 40., 1]], #三叉路マップ
#             [[4., 26., 3], [16., 26., 3]],
#             [[22., 26., 3], [54., 26., 3]],
#             [[16., 4., 2], [16., 26., 2]],
#             [[22., 4., 0], [22., 26., 0]]])

# canvas_element, width, height, wall_arr = convex_map()
canvas_element, width, height, wall_arr = intersection_map()

nol_num = 10  # 通常の人数
for_num = 0  # 強引な人の人数
len = 1.5
seed = 10 # 乱数生成用  #old: tmp 201 tau 0.2 seed103(timestep220?ぐらい気になる)seed104 seed105(おしくらまんじゅうっぽい異常ではない) 
f_tau = 0.1 

add_file_name = f"/home/user/projects/sfm/tmp_mesa/tmp_codex_review/server_model/tmp_data/tau_20/nol_pop_100/seed_100/csv"

model_params = {
    "nol_pop": nol_num,  # 人数
    "for_pop": for_num,
    "seed": seed,
    "wall_arr": wall_arr,
    # "width": 60,  # 見かけの大きさ(マップ) #三叉路マップ
    # "height": 60,  # 見かけの大きさ(マップ)
    "width": 200,  # 見かけの大きさ(マップ)
    "height": 200,  # 見かけの大きさ(マップ)
    # フォルダ名
    # "add_file_name": f"../data/ex4_convex_for_1/hendel_ex4_convex_for_1_csv/{pop_num}_force_m_{f_m}/seed_{seed}/csv/",
    "add_file_name": add_file_name,
}

server = mesa.visualization.ModularServer(
    MoveAgent,
    [canvas_element],
    "SFM evacuation",  # 題名(左上に表示)
    model_params,
)
