import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from scipy.ndimage import binary_dilation

def checkLimits(envmap, state):
    x, y = state
    return 1 <= x <= envmap.shape[1] and 1 <= y <= envmap.shape[0] and envmap[y-1, x-1] != 1

def astar(envmap, start, goal, epsilon=1.0, inflate_factor=0):
    """
    A* 路径规划算法。
    :param envmap: 环境地图，0 表示自由空间，1 表示障碍物
    :param start: 起点坐标，格式为 [x, y]
    :param goal: 终点坐标，格式为 [x, y]
    :param epsilon: 启发式因子，默认为 1.0
    :param inflate_factor: 障碍物膨胀因子，默认为 0
    :return: 规划的路径 (fpath), 路径成本 (cost), 显示地图 (displaymap)
    """
    if len(start) != 2 or len(goal) != 2:
        raise ValueError("Start and goal must be 2D coordinates.")
    if epsilon < 1.0:
        raise ValueError("Epsilon must be >= 1.0.")
    if not isinstance(inflate_factor, int) or inflate_factor < 0:
        raise ValueError("inflate_factor must be a non-negative integer.")
    
    print(f"Using epsilon = {epsilon}, inflate_factor = {inflate_factor}")
    
    # 定义结构元素
    structure = np.ones((3,3), dtype=int)
    
    # 膨胀障碍物
    inflate_map = binary_dilation(envmap == 1, structure=structure, iterations=inflate_factor)
    
    # 更新环境地图
    envmap_inflated = envmap.copy()
    envmap_inflated[inflate_map] = 1
    
    # 确保起点和终点不被设置为障碍物
    envmap_inflated[goal[1]-1, goal[0]-1] = 0
    envmap_inflated[start[1]-1, start[0]-1] = 0
    
    # 初始化显示地图
    displaymap = envmap_inflated.copy()
    
    # 定义动作和成本
    actions = [(-1, 0, 1), (-1, 1, np.sqrt(2)), (0, 1, 1), (1, 1, np.sqrt(2)),
               (1, 0, 1), (1, -1, np.sqrt(2)), (0, -1, 1), (-1, -1, np.sqrt(2))]
    
    # 初始化开放列表和关闭列表
    open_list = PriorityQueue()
    open_list.put((0, start[0], start[1], 0, 0, 0))  # (f_cost, x, y, g_cost, parent_x, parent_y)
    closed_list = set()
    parents = {}
    
    while not open_list.empty():
        _, x, y, g_cost, parent_x, parent_y = open_list.get()
        
        if (x, y) == tuple(goal):
            break
        
        if (x, y) in closed_list:
            continue
        
        closed_list.add((x, y))
        displaymap[y-1, x-1] = 3  # 标记为已访问
        
        for dx, dy, cost in actions:
            nx, ny = x + dx, y + dy
            if not checkLimits(envmap_inflated, (nx, ny)):
                continue
            
            new_g_cost = g_cost + cost
            new_f_cost = new_g_cost + epsilon * np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
            
            if (nx, ny) not in closed_list:
                open_list.put((new_f_cost, nx, ny, new_g_cost, x, y))
                displaymap[ny-1, nx-1] = 2  # 标记为待访问
                parents[(nx, ny)] = (x, y)
        
    # 回溯找到路径
    fpath = []
    cost = 0
    current = tuple(goal)
    
    while current != tuple(start):
        fpath.append(current)
        next_node = parents[current]
        cost += np.sqrt((current[0] - next_node[0])**2 + (current[1] - next_node[1])**2)
        current = next_node
    
    fpath.append(tuple(start))
    fpath.reverse()
    
    print(f"Number of closed states: {np.sum(displaymap == 3)}")
    
    return np.array(fpath), cost, displaymap

# 示例使用：
# envmap = np.zeros((10, 10))  # 示例地图
# start = [1, 1]
# goal = [8, 8]
# epsilon = 1.0
# inflate_factor = 1
# fpath, cost, displaymap = astar(envmap, start, goal, epsilon, inflate_factor)
