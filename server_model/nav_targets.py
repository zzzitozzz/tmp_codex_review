import numpy as np

DEFAULT_ROAD_WIDTH = 6.0


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _axis_dir(from_pos, to_pos):
    vec = np.array(to_pos) - np.array(from_pos)
    if abs(vec[0]) >= abs(vec[1]):
        return np.array([_sign(vec[0]), 0], dtype=int)
    return np.array([0, _sign(vec[1])], dtype=int)


def _project_to_segment(pos, a, b):
    v = b - a
    denom = np.dot(v, v)
    if denom <= 0.0:
        return np.array(a, dtype=float)
    t = np.dot(pos - a, v) / denom
    t = np.clip(t, 0.0, 1.0)
    return a + t * v


def _perp_dir(dir_in):
    if abs(dir_in[0]) > 0:
        return np.array([0.0, 1.0])
    return np.array([1.0, 0.0])


def _straight_target(agent_pos, cur_pos, dir_in, road_width):
    if np.all(dir_in == 0):
        return np.array(cur_pos, dtype=float)
    half_width = road_width / 2.0
    dir_perp = _perp_dir(dir_in)
    a = np.array(cur_pos, dtype=float) - dir_perp * half_width
    b = np.array(cur_pos, dtype=float) + dir_perp * half_width
    return _project_to_segment(np.array(agent_pos, dtype=float), a, b)


def _turn_corner(cur_pos, dir_in, dir_out, half_width):
    corner_x = cur_pos[0] + half_width * (dir_out[0] if dir_out[0] != 0 else dir_in[0])
    corner_y = cur_pos[1] + half_width * (dir_out[1] if dir_out[1] != 0 else dir_in[1])
    return np.array([corner_x, corner_y])


def _corner_target(cur_pos, corner_pos, distance):
    sign_x = -1 if corner_pos[0] > cur_pos[0] else 1
    sign_y = -1 if corner_pos[1] > cur_pos[1] else 1
    return np.array([
        corner_pos[0] + sign_x * distance,
        corner_pos[1] + sign_y * distance,
    ])


def _turn_distance(agent_pos, corner_pos, dir_in, road_width):
    if dir_in[0] == 0:
        distance = abs(corner_pos[0] - agent_pos[0])
    else:
        distance = abs(corner_pos[1] - agent_pos[1])
    return float(np.clip(distance, 0.0, road_width))


def compute_target_pos(agent_pos, prev_pos, cur_pos, next_pos,
                       in_dest_d, road_width=DEFAULT_ROAD_WIDTH):
    """Return a target position for SFM guidance.

    Straight paths use a centerline segment projection through the intersection.
    Turns use a corner-based target that depends on the agent's wall distance.
    """
    if next_pos is None:
        return np.array(cur_pos)

    prev_anchor = agent_pos if prev_pos is None else prev_pos
    dir_in = _axis_dir(prev_anchor, cur_pos)
    dir_out = _axis_dir(cur_pos, next_pos)
    if np.all(dir_in == dir_out) or np.dot(dir_in, dir_out) != 0:
        return _straight_target(agent_pos, cur_pos, dir_in, road_width)

    half_width = road_width / 2.0
    corner_pos = _turn_corner(cur_pos, dir_in, dir_out, half_width)
    distance = _turn_distance(agent_pos, corner_pos, dir_in, road_width)
    return _corner_target(cur_pos, corner_pos, distance)


def debug_turn_targets():
    cur = np.array([5.0, 38.0])
    prev = np.array([5.0, 100.0])
    next_pos = np.array([51.0, 38.0])
    agent_pos = np.array([3.0, 90.0])
    target = compute_target_pos(agent_pos, prev, cur, next_pos, in_dest_d=3.0)
    expected = np.array([3.0, 40.0])
    assert np.allclose(target, expected), f"expected {expected}, got {target}"

    agent_pos = np.array([7.0, 90.0])
    target = compute_target_pos(agent_pos, prev, cur, next_pos, in_dest_d=3.0)
    expected = np.array([7.0, 36.0])
    assert np.allclose(target, expected), f"expected {expected}, got {target}"

    dir_in = _axis_dir(prev, cur)
    dir_out = _axis_dir(cur, next_pos)
    corner_check = _turn_corner(cur, dir_in, dir_out, DEFAULT_ROAD_WIDTH / 2.0)
    assert np.allclose(corner_check, corner), f"expected {corner}, got {corner_check}"

    return True


if __name__ == "__main__":
    debug_turn_targets()
