class Maze:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def is_free(self, position):
        x, y = position
        return (
            0 <= x < self.rows and
            0 <= y < self.cols and
            self.grid[x][y] == 0
        )

    def get_legal_actions(self, state):
        x, y = state
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        actions = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_free((nx, ny)):
                actions.append((nx, ny))
        return actions
