import numpy as np


def load_agent_csv(path):
    """
    描画用にCSVを読み込み、位置と状態配列を返す。
    2列(旧仕様)でも3列(新仕様)でも動作する。
    """
    data = np.atleast_2d(np.loadtxt(path, delimiter=","))
    pos_array = data[:, :2]
    state_array = data[:, 2].astype(int) if data.shape[1] >= 3 else None
    return pos_array, state_array


class Human:
    def __init__(self, unique_id, pos_array=None, state_array=None, dt=0.1):
        self.unique_id = unique_id
        self.pos_array = pos_array if pos_array is not None else np.zeros((0, 2))
        self.state_array = state_array
        self.dt = dt
        self.elapsed_step = 0
        self.pos = np.array(self.pos_array[0]) if len(self.pos_array) else np.zeros(2)
        self.block_info_state = 0
        self.tmp_pos = np.copy(self.pos)

    def step(self):
        if self.elapsed_step < len(self.pos_array):
            self.tmp_pos = self.pos_array[self.elapsed_step]

    def advance(self):
        if self.elapsed_step < len(self.pos_array):
            self.pos = self.tmp_pos
            if self.state_array is not None:
                self.block_info_state = int(self.state_array[self.elapsed_step])
            self.elapsed_step += 1


class ForcefulHuman(Human):
    pass


def portrayal_method(obj):
    color = "blue" if getattr(obj, "block_info_state", 0) >= 1 else "red"
    return {"Shape": "ellipse", "Color": color, "r": 0.5, "Filled": True}
