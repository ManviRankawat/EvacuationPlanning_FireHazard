import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import deque

from maze import Maze

# In planner_test.py, after "from maze import Maze"
class DynamicFireMaze(Maze):
    """Extends the Maze class to support dynamic fire appearance"""
    def __init__(self, grid, start, goal):
        super().__init__(grid, start, goal)
        self.fire_blocks = set()  # Use a set instead of a single fire_block
        self.potential_fires = {
            (6, 6): 2,  # Format: fire_position: proximity_threshold
            (0, 9): 3
        }
        
    def update_fires(self, agent_position):
        """Check if agent is near potential fires and activate them if so"""
        new_fires = False
        # Check if agent is close to any potential fire locations
        for fire_pos, threshold in list(self.potential_fires.items()):
            distance = abs(agent_position[0] - fire_pos[0]) + abs(agent_position[1] - fire_pos[1])
            if distance <= threshold:
                print(f"\n🔥 Fire appears at {fire_pos} as the robot approaches!")
                self.fire_blocks.add(fire_pos)  # Add to fire blocks set
                del self.potential_fires[fire_pos]  # Remove from potential fires
                new_fires = True
        return new_fires
        
    def get_legal_actions(self, state):
        """Override to check against all fire blocks"""
        actions = []
        x, y = state
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                if self.grid[nx][ny] == 0 and (nx, ny) not in self.fire_blocks:
                    actions.append((nx, ny))
        return actions

MAX_ROLLOUT_DEPTH = 30
EXPLORATION_CONSTANT = 1.4
SIMULATIONS = 1000
PATH_LENGTH_PENALTY = 0.05

class Node:
    def __init__(self, state, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.depth = depth

    def is_fully_expanded(self, maze):
        return len(self.children) == len(maze.get_legal_actions(self.state))

    def best_child(self, c=EXPLORATION_CONSTANT):
        return max(
            self.children,
            key=lambda node: (node.reward / max(node.visits, 1)) - 
                            (PATH_LENGTH_PENALTY * node.depth) + 
                            c * math.sqrt(math.log(max(self.visits, 1)) / max(node.visits, 1))
        )
    
def rollout(state, maze, goal, max_depth=MAX_ROLLOUT_DEPTH):
    path_length = 0
    visited = set([state])  # Track visited states to avoid cycles
    
    for _ in range(max_depth):
        actions = maze.get_legal_actions(state)
        
        # Filter out actions that lead to already visited states
        unvisited_actions = [a for a in actions if a not in visited]
        
        if not unvisited_actions:
            if not actions:  # No actions at all
                return 0
            # If all actions lead to visited states, pick randomly from all actions
            next_state = random.choice(actions)
        else:
            # Calculate Manhattan distance to goal for each unvisited action
            distances = [(abs(a[0] - goal[0]) + abs(a[1] - goal[1]), a) for a in unvisited_actions]
            
            # 70% of the time, pick from the actions that lead closer to the goal
            if random.random() < 0.7:
                distances.sort()  # Sort by distance (closest first)
                next_state = distances[0][1]  # Pick the closest to goal
            else:
                next_state = random.choice(unvisited_actions)
        
        state = next_state
        visited.add(state)
        path_length += 1
        
        if state == goal:
            # Higher reward for shorter paths
            return 1.0 * (1 + (max_depth - path_length) / max_depth)
            
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
            new_node = Node(state=move, parent=node, depth=node.depth + 1)
            node.children.append(new_node)
            return new_node
    return node

def mcts(root, maze, goal, recent_states=None):
    for _ in range(SIMULATIONS):
        node = root
        # Selection: Traverse the tree by picking the best child until a node is not fully expanded
        while node.is_fully_expanded(maze) and node.children:
            node = node.best_child()
            if node is None:
                break
        # Expansion: If the goal hasn't been reached, expand the current node with a new child
        if node and node.state != goal:
            expanded_node = expand(node, maze)
            if expanded_node == node:
                pass
            else:
                node = expanded_node
        # Simulation: Perform a random rollout from the selected node to estimate the reward    
        if node:
            result = rollout(node.state, maze, goal)
        # Backpropagation: Update the reward and visit counts back up the tree
        backpropagate(node, result)

    if not root.children:
        print("⚠️ No valid children found. Possibly stuck or fire blocked all paths.")
        return root.state 
    
    # When selecting the best move, avoid states that have been visited recently
    if recent_states:
        # First try to find a child that hasn't been visited recently
        unvisited_children = [child for child in root.children if child.state not in recent_states]
        if unvisited_children:
            # Find the best among unvisited
            best = max(unvisited_children, 
                      key=lambda node: node.reward / max(node.visits, 1))
            return best.state
    
    # If all children have been visited recently or recent_states is None, pick the best
    best = root.best_child(c=0)
    if best:
        return best.state
    else:
        return root.state
    
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

    recent_states = deque(maxlen=10)  # Remember last 10 states
    recent_states.append(current_state)
    
    # Dictionary to track state visitation count
    state_visits = {current_state: 1}

    fire_events = []

    while current_state != maze.goal:

        if hasattr(maze, 'update_fires') and maze.update_fires(current_state):
            # If new fires appeared, record the event
            fire_events.append((len(path), current_state, list(maze.fire_blocks)))
        next_state = mcts(root, maze, maze.goal, recent_states=set(recent_states))

        if next_state in state_visits:
            state_visits[next_state] += 1
            # If we've visited this state many times, try to force a different path
            if state_visits[next_state] > 3:
                print(f"Visited state {next_state} multiple times. Forcing exploration.")
                legal_actions = maze.get_legal_actions(current_state)
                # Remove heavily visited states
                unexplored = [state for state in legal_actions 
                             if state not in state_visits or state_visits[state] < 2]
                if unexplored:
                    next_state = random.choice(unexplored)
        else:
            state_visits[next_state] = 1

        if hasattr(maze, 'fire_blocks') and next_state in maze.fire_blocks:
            print("Avoiding fire block!")
            legal_actions = maze.get_legal_actions(current_state)
            if legal_actions:
                next_state = random.choice(legal_actions)
            else:
                print("No safe moves available!")
                break

        # Create new root with correct depth information
        next_depth = root.depth + 1 if hasattr(root, 'depth') else 1
        new_root = Node(next_state, parent=None, depth=next_depth)
        root = new_root
        current_state = next_state
        path.append(current_state)
        recent_states.append(current_state)

    return path, fire_events

def plot_path(maze, path, fire_events=None, delay=0.3):
    grid = np.array(maze.grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='binary')

    # Plot start and goal
    sx, sy = maze.start
    gx, gy = maze.goal
    ax.plot(sy, sx, "go")  
    try:
        goal_img = mpimg.imread("assets/goal.png")
        goal_icon = OffsetImage(goal_img, zoom=0.05)
        goal_box = AnnotationBbox(goal_icon, (gy, gx), frameon=False)
        ax.add_artist(goal_box)
    except FileNotFoundError:
        ax.plot(gy, gx, "r*", markersize=12)

    ax.set_xticks(np.arange(len(grid[0])))
    ax.set_yticks(np.arange(len(grid)))
    ax.grid(True)
    ax.legend()
    plt.gca().invert_yaxis()
    plt.title("Maze Path using MCTS")

    fire_boxes = {}
    active_fires = set()

    fire_event_idx = 0

    previous_positions = []
    
    try:
        person_img = mpimg.imread("assets/person.png")
        fire_img = mpimg.imread("assets/fire.png")
        imagebox = OffsetImage(person_img, zoom=0.05)
        ab = None

        for i, (x, y) in enumerate(path):
            for prev_x, prev_y in previous_positions:
                ax.plot(prev_y, prev_x, 'b.', alpha=0.7, markersize=8)
            if fire_events and fire_event_idx < len(fire_events) and i == fire_events[fire_event_idx][0]:
                # Get new fires from the event
                _, _, fires = fire_events[fire_event_idx]
                new_fires = set(fires) - active_fires
                
                # Animate fire appearance
                for fx, fy in new_fires:
                    # Flashing animation
                    for _ in range(3):
                        fire_icon = OffsetImage(fire_img, zoom=0.01)
                        fire_box = AnnotationBbox(fire_icon, (fy, fx), frameon=False)
                        ax.add_artist(fire_box)
                        plt.title("FIRE APPEARS!")
                        plt.pause(0.2)
                        fire_box.remove()
                        plt.pause(0.1)
                    
                    # Add permanent fire
                    fire_icon = OffsetImage(fire_img, zoom=0.02)
                    fire_box = AnnotationBbox(fire_icon, (fy, fx), frameon=False)
                    ax.add_artist(fire_box)
                    fire_boxes[(fx, fy)] = fire_box
                
                active_fires.update(new_fires)
                fire_event_idx += 1
                plt.pause(1.0)  

            if ab:
                ab.remove()
                if i > 0:
                    previous_positions.append(path[i-1])
            ab = AnnotationBbox(imagebox, (y, x), frameon=False)
            ax.add_artist(ab)
            plt.title(f"Maze Path using MCTS - Step {i+1}/{len(path)}")
            plt.pause(delay)
    except FileNotFoundError:
        for i, (x, y) in enumerate(path):
            # Add blue dots at previous positions
            for prev_x, prev_y in previous_positions:
                ax.plot(prev_y, prev_x, 'b.', alpha=0.7, markersize=8)
            
            if fire_events and fire_event_idx < len(fire_events) and i == fire_events[fire_event_idx][0]:
                _, _, fires = fire_events[fire_event_idx]
                new_fires = set(fires) - active_fires
                
                for fx, fy in new_fires:
                    ax.plot(fy, fx, "rx", markersize=10)
                    ax.text(fy, fx, "FIRE", color='red', ha='center', va='center')
                
                active_fires.update(new_fires)
                fire_event_idx += 1
                plt.pause(1.0)
           
            if i > 0:
                previous_positions.append(path[i-1])
            
            ax.plot(y, x, "bs")
            
            plt.title(f"Maze Path using MCTS - Step {i+1}/{len(path)}")
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
    maze = DynamicFireMaze(grid=grid, start=start, goal=goal)

    start_time = time.time()
    path, fire_events = solve_maze(maze)
    print("Found path:", path)
    print("Steps:", len(path))
    print("Fire events:", fire_events)
    print("Time taken: %.2f seconds" % (time.time() - start_time))

    plot_path(maze, path, fire_events, delay=0.1)