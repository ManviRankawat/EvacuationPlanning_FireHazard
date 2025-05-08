# Evacuation Planning during Fire Hazard
## Manvi Rankawat - G24324576

---

### Problem Statement:
An intelligent evacuation planning system using Monte Carlo Tree Search to guide a rescue agent toward the nearest exit while dynamically adapting to spreading fires and uncertain environments.

**Uncertainty involved**: The simulation introduces new obstacles randomly during execution.  
We are restricting the model to a 2D floor plan (grid-based environment) to ensure the project remains feasible within the course timeline.

---

### Why the Problem is Non-Trivial?
This problem is non-trivial because the environment is **dynamic** and **partially observable**. Hazards do not follow predictable patterns and can appear at any time, making precomputed paths invalid.

Traditional pathfinding methods like **BFS** or **A\*** assume a static grid and cannot adapt efficiently to frequent changes. Replanning from scratch with every change is computationally expensive, especially when hazards must be predicted ahead of time.

---

### Existing Solution Methods
Existing approaches to dynamic path planning include:

- **Simultaneous Localization and Mapping (SLAM)**: Used in robotics to map unknown environments but is often excessive for 2D grid scenarios and suffers from high computational cost.

- **Rapidly-Exploring Random Trees (RRTs)**: Effective in continuous spaces but require frequent replanning in dynamic environments, making them inefficient in real-time grid-based planning.

- **Probabilistic Roadmaps (PRM)**: Work in known static environments but require resampling when obstacles move, reducing their reliability in highly dynamic settings.

---

## Modeling and Solving the Problem

### State Space:
Represents the rescue agent’s position on a 2D grid (0,0), along with dynamic obstacles and escape target(10,1
).

### Action Space:
Consists of 4 discrete movement directions: Up, Down, Left, Right.

### Transitions:
  #### Agent Movement:
   - The agent navigates the maze using MCTS- Selection, Expansion, Simulation and Backpropagation phases guide each move.
   - Implemented in solve_maze() when current_state is updated to next_state
   - Path history is tracked and visualized with blue dots
  #### Environment Appearance:
   - Fire hazards appear dynamically in the environment
     
### Observations:
The rescue agent **directly observes** its surroundings (e.g., adjacent grid cells) at each step. These include fire and blocked paths. The agent perceives the environment based on **line of sight**, not sensors.

---

### Methodology

**Monte Carlo Simulation for Pathfinding Under Uncertainty**
Monte Carlo methods are used to evaluate and select optimal evacuation paths by simulating numerous possible scenarios under uncertain and dynamic conditions. By repeatedly sampling possible hazard spreads, blocked routes, and agent decisions, the agent can identify paths with the highest expected safety and success rate. This allows for informed, probabilistic decision-making even in partially observable environments.

---
![image](https://github.com/user-attachments/assets/3fc34011-8700-49f9-9f3a-d9a7dd320ad7)

