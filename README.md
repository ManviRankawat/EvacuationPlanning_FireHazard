# Evacuation Planning during Fire Hazard
## Manvi Rankawat - G24324576

---

### Problem Statement:
Goal is to create an intelligent evacuation plan to help guide a rescue agent toward the nearest exit.  
This is done using an algorithm that enables the rescuer to adapt in real time—rerouting when paths are blocked, fire spreads.

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
Represents the rescue agent’s position on a 2D grid, along with static obstacles, predicted hazard regions, and estimated positions of civilians and exits.

### Action Space:
Consists of 8 discrete movement directions: Up, Down, Left, Right, and the four diagonals (Up-Right, Up-Left, Down-Right, Down-Left).

### Observations:
The rescue agent **directly observes** its surroundings (e.g., adjacent grid cells) at each step. These include visible civilians, fire, smoke, debris, and blocked paths. The agent perceives the environment based on **line of sight**, not sensors.

---

### Methodology

**Monte Carlo Simulation for Pathfinding Under Uncertainty**
Monte Carlo methods are used to evaluate and select optimal evacuation paths by simulating numerous possible scenarios under uncertain and dynamic conditions. By repeatedly sampling possible hazard spreads, blocked routes, and agent decisions, the agent can identify paths with the highest expected safety and success rate. This allows for informed, probabilistic decision-making even in partially observable environments.

---
![image](https://github.com/user-attachments/assets/3fc34011-8700-49f9-9f3a-d9a7dd320ad7)

