class Maze:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.fire_block = None
        
    def is_free(self, position):
        x, y = position
        if self.fire_block and position == self.fire_block:
            return False
        return (
            0 <= x < self.rows and
            0 <= y < self.cols and
            self.grid[x][y] == 0
        )

    def get_legal_actions(self, state):
        actions = []
        x, y = state
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                if self.grid[nx][ny] == 0 and (nx, ny) != self.fire_block:
                    actions.append((nx, ny))
        return actions

