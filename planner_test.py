import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from maze import Maze

MAX_ROLLOUT_DEPTH = 20
EXPLORATION_CONSTANT = 1.4
SIMULATIONS = 1000

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_fully_expanded(self, maze):
        return len(self.children) == len(maze.get_legal_actions(self.state))

    def best_child(self, c=EXPLORATION_CONSTANT):
        return max(
            self.children,
            key=lambda node: node.reward / node.visits + c * math.sqrt(math.log(self.visits) / node.visits)
        )

def rollout(state, maze, goal):
    prev_state = None
    for _ in range(MAX_ROLLOUT_DEPTH):
        actions = maze.get_legal_actions(state)
        if prev_state and prev_state in actions:
            actions.remove(prev_state)
        if not actions:
            break
        prev_state = state
        state = random.choice(actions)
        if state == goal:
            return 1
    return 0

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.reward += result
        node = node.parent

def expand(node, maze):
    tried = [child.state for child in node.children]
    legal = maze.get_legal_actions(node.state)
    for move in legal:
        if move not in tried and (node.parent is None or move != node.parent.state):
            new_node = Node(state=move, parent=node)
            node.children.append(new_node)
            return new_node
    return node

def mcts(root, maze, goal):
    for _ in range(SIMULATIONS):
        node = root
        # Selection: Traverse the tree by picking the best child until a node is not fully expanded
        while node.is_fully_expanded(maze) and node.children:
            node = node.best_child()
        # Expansion: If the goal hasn't been reached, expand the current node with a new child
        if node.state != goal:
            node = expand(node, maze)
        # Simulation: Perform a random rollout from the selected node to estimate the reward    
        result = rollout(node.state, maze, goal)
        # Backpropagation: Update the reward and visit counts back up the tree
        backpropagate(node, result)

    if not root.children:
        print("⚠️ No valid children found. Possibly stuck or fire blocked all paths.")
        return root.state 
    
    # After all simulations, choose the best child (exploitation only) as the next move
    return root.best_child(c=0).state

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

def solve_maze(maze):
    current_state = maze.start
    root = Node(current_state)
    path = [current_state]

    while current_state != maze.goal:
        next_state = mcts(root, maze, maze.goal)

        # Dynamically block (8,8) with fire (logic only)
        if current_state == (6, 6):
            print("\n⚡ Blocking (8,8) dynamically with fire!")
            maze.fire_block = (2, 6)

        new_root = Node(next_state, parent=root)
        root = new_root
        current_state = next_state
        path.append(current_state)

    return path

def plot_path(maze, path, delay=0.3):
    grid = np.array(maze.grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='binary')

    # Plot start and goal
    sx, sy = maze.start
    gx, gy = maze.goal
    ax.plot(sy, sx, "go")  
    goal_img = mpimg.imread("assets/goal.png")
    goal_icon = OffsetImage(goal_img, zoom=0.05)
    goal_box = AnnotationBbox(goal_icon, (gy, gx), frameon=False)
    ax.add_artist(goal_box)

    ax.set_xticks(np.arange(len(grid[0])))
    ax.set_yticks(np.arange(len(grid)))
    ax.grid(True)
    ax.legend()
    plt.gca().invert_yaxis()
    plt.title("Maze Path using MCTS")

    if maze.fire_block:
            fire_img = mpimg.imread("assets/fire.png")
            fx, fy = maze.fire_block
            fire_box = OffsetImage(fire_img, zoom=0.01)
            fire_ab = AnnotationBbox(fire_box, (fy, fx), frameon=False)
            ax.add_artist(fire_ab)
       
    try:
        person_img = mpimg.imread("assets/person.png")
        imagebox = OffsetImage(person_img, zoom=0.05)
        ab = None
        for (x, y) in path:
            if ab:
                ab.remove()
            ab = AnnotationBbox(imagebox, (y, x), frameon=False)
            ax.add_artist(ab)
            plt.pause(delay)
    except FileNotFoundError:
        for (x, y) in path:
            ax.plot(y, x, "bs")
            plt.pause(delay)

    plt.show()

if __name__ == "__main__":
    grid = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (10, 1)
    maze = Maze(grid=grid, start=start, goal=goal)

    start_time = time.time()
    path = solve_maze(maze)
    print("Found path:", path)
    print("Steps:", len(path))
    print("Time taken: %.2f seconds" % (time.time() - start_time))

    plot_path(maze, path, delay=0.1)
