import numpy as np
from collections import namedtuple
import heapq
import matplotlib.pyplot as plt

class GenerateMaze:
    def __init__(self, width, height, obstacle_probability=0.3):
        self.width = width
        self.height = height
        self.obstacle_probability = obstacle_probability
        self.maze = None
        self.solution = None
        self.start = None
        self.end = None
        self._generate_maze()

    def _generate_maze(self):
        maze_found = False
        while not maze_found:
            self.maze = np.zeros((self.width, self.height), dtype=int)
            num_obstacles = int(self.obstacle_probability * self.width * self.height)
            random_obstacle_x_pos = np.random.randint(0, self.width, num_obstacles, dtype=int)
            random_obstacle_y_pos = np.random.randint(0, self.height, num_obstacles, dtype=int)
            self.maze[random_obstacle_x_pos, random_obstacle_y_pos] = 1
            start = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            end = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            while self.maze[*start]:
                start = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            while start == end or self.maze[end] == 1:
                end = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            self.a_star_algorithm(start, end)
            self.start = start
            self.end = end
            if self.solution is not None:
                maze_found = True

    def heuristic_fn(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    def cost_fn(self, loc1, loc2):
        return 1
    
    def _get_next_moves(self, maze, loc):
        moves = []
        if loc[0] > 0 and maze[loc[0] - 1, loc[1]] == 0:
            moves.append((loc[0] - 1, loc[1]))
        if loc[0] < self.width - 1 and maze[loc[0] + 1, loc[1]] == 0:
            moves.append((loc[0] + 1, loc[1]))
        if loc[1] > 0 and maze[loc[0], loc[1] - 1] == 0:
            moves.append((loc[0], loc[1] - 1))
        if loc[1] < self.height - 1 and maze[loc[0], loc[1] + 1] == 0:
            moves.append((loc[0], loc[1] + 1))
        return moves
    
    def a_star_algorithm(self, start, end):
        Node = namedtuple('Node', ['f_n', 'h_n', 'g_n', 'location', 'prev_node'])
        num_movement_spaces = int(self.width * self.height * (1 - self.obstacle_probability))
        fringe = []
        heapq.heappush(fringe, Node(0, 0, 0, start, None))
        while fringe:
            current = heapq.heappop(fringe)
            if current.g_n > num_movement_spaces:
                break
            if current.location == end:
                path = []
                while current is not None:
                    path.append(current.location)
                    current = current.prev_node
                self.solution = path[::-1]
                break
            for move in self._get_next_moves(self.maze, current.location):
                g_n = current.g_n + self.cost_fn(current.location, move)
                h_n = self.heuristic_fn(move, end)
                f_n = g_n + h_n
                for node in fringe:
                    if node.location == move and node.g_n >= g_n:
                        fringe.remove(node)
                        heapq.heapify(fringe)
                heapq.heappush(fringe, Node(f_n, h_n, g_n, move, current))

def display_maze(maze, start, end, solution=None, title="Random Solvable Maze", show_solution=False):
    width = maze.shape[0]
    height = maze.shape[1]
    maze_image = np.ones((width, height, 3), dtype=int) * 255
    maze_image[*start, :] = (255, 0, 0)
    maze_image[*end, :] = (0, 128, 0)
    for obstacle in np.argwhere(maze == 1):
        maze_image[obstacle[0], obstacle[1], :] = (0, 0, 0)
    if show_solution:
        for move in solution[1:-1]:
            maze_image[move[0], move[1], :] = (0, 0, 255)
    maze_image = np.transpose(maze_image, (1, 0, 2))
    fig, ax = plt.subplots()
    ax.imshow(maze_image)
    ax.axis('off')
    ax.set_title(title, color="white", fontsize=20)
    fig.set_facecolor("black")
    fig.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    maze = GenerateMaze(15, 15, 0.3)
    print(maze.maze, maze.start, maze.end, maze.solution)
    display_maze(maze.maze, maze.start, maze.end, maze.solution, show_solution=True)