import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
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
    for _ in range(MAX_ROLLOUT_DEPTH):
        actions = maze.get_legal_actions(state)
        if not actions:
            break
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
        if move not in tried:
            new_node = Node(state=move, parent=node)
            node.children.append(new_node)
            return new_node
    return node

def mcts(root, maze, goal):
    for _ in range(SIMULATIONS):
        node = root
        while node.is_fully_expanded(maze) and node.children:
            node = node.best_child()
        if node.state != goal:
            node = expand(node, maze)
        result = rollout(node.state, maze, goal)
        backpropagate(node, result)
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
    ax.plot(sy, sx, "go", label="Start")  # Green dot
    ax.plot(gy, gx, "ro", label="Goal")   # Red dot

    # Static setup
    ax.set_xticks(np.arange(len(grid[0])))
    ax.set_yticks(np.arange(len(grid)))
    ax.grid(True)
    ax.legend()
    plt.gca().invert_yaxis()
    plt.title("Maze Path using MCTS")

    # Animate step-by-step
    for (x, y) in path:
        ax.plot(y, x, "bs")  # Blue square for path
        plt.pause(delay)

    plt.show()

if __name__ == "__main__":
    grid = [
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    start = (0, 0)
    goal = (4, 4)

    maze = Maze(grid=grid, start=start, goal=goal)

    start_time = time.time()
    path = solve_maze(maze)
    print("Found path:", path)
    print("Steps:", len(path))
    print("Time taken: %.2f seconds" % (time.time() - start_time))

    plot_path(maze, path, delay=0.3)
