import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Maze:
    def __init__(self, maze, start, end, solution=None, render_maze=False):
        self.maze = maze
        self.height = maze.shape[0]
        self.width = maze.shape[1]
        self.start = start
        self.end = end
        self.current = start
        self.solution = solution
        self.render_maze = render_maze
        self.fitness_value = len(solution)
    
    def get_actions(self):
        loc = self.current
        moves = [0, 0, 0, 0]
        # left (loc[0] - 1, loc[1])
        if loc[0] > 0 and self.maze[loc[0] - 1, loc[1]] == 0:
            moves[0] = 1
        # up (loc[0], loc[1] - 1)
        if loc[1] > 0 and self.maze[loc[0], loc[1] - 1] == 0:
            moves[1] = 1
        # right (loc[0] + 1, loc[1])
        if loc[0] < self.width - 1 and self.maze[loc[0] + 1, loc[1]] == 0:
            moves[2] = 1
        # down (loc[0], loc[1] + 1)
        if loc[1] < self.height - 1 and self.maze[loc[0], loc[1] + 1] == 0:
            moves[3] = 1
        return moves
    
    def random_action(self):
        moves = self.get_actions()
        return np.random.choice([i for i in range(4) if moves[i] == 1])
    
    def move(self, action):
        loc = self.current
        self.fitness_value -= 1
        if action == 0:
            self.current = (loc[0] - 1, loc[1])
        elif action == 1:
            self.current = (loc[0], loc[1] - 1)
        elif action == 2:
            self.current = (loc[0] + 1, loc[1])
        elif action == 3:
            self.current = (loc[0], loc[1] + 1)
        else:
            raise ValueError("Invalid action")

    def _render_one_move(self, title="Maze", show_solution=False):
        if not self.render_maze:
            raise ValueError("Cannot render without plotting")
        maze_image = np.ones((self.width, self.height, 3), dtype=int) * 255
        maze_image[*self.start, :] = (255, 0, 0)
        maze_image[*self.end, :] = (0, 255, 0)
        for obstacle in np.argwhere(self.maze == 1):
            maze_image[obstacle[0], obstacle[1], :] = (0, 0, 0)
        if show_solution:
            for move in self.solution[1:-1]:
                maze_image[move[0], move[1], :] = (128, 128, 128)
        maze_image[*self.current, :] = (0, 0, 255)
        maze_image = np.transpose(maze_image, (1, 0, 2))
        self.ax.clear()
        self.ax.imshow(maze_image)
        self.ax.axis('off')
        self.ax.set_title(title, color="white", fontsize=20)
        self.fig.set_facecolor("black")
        self.fig.tight_layout()

    def render(self, action_sequence, title="Maze", wait_time=0.3, show_solution=False):
        if not self.render_maze:
            raise ValueError("Cannot render without plotting")
        new_maze = Maze(self.maze, self.start, self.end, self.solution, render_maze=True)
        new_maze.fig, new_maze.ax = plt.subplots()
        new_maze.exit_plot = False
        new_maze._render_one_move(title, show_solution)
        plt.draw()
        plt.pause(wait_time)
        for action in action_sequence:
            new_maze.move(action)
            new_maze._render_one_move(title, show_solution)
            plt.draw()
            plt.pause(wait_time)
            if new_maze.exit_plot:
                plt.show()
                break
        plt.show()

    def get_current_maze(self):
        maze = deepcopy(self.maze)
        maze[self.current[0], self.current[1]] = 2
        maze[self.end[0], self.end[1]] = 3
        return maze
    

if __name__ == "__main__":
    from maze_generator import GenerateMaze
    maze_gen = GenerateMaze(10, 10, 0.5)
    maze = Maze(maze_gen.maze, maze_gen.start, maze_gen.end, maze_gen.solution, render_maze=True)
    action_sequence = [maze.random_action()]
    for i in range(10):
        maze.move(action_sequence[-1])
        action_sequence.append(maze.random_action())
    maze.move(action_sequence[-1])
    maze.render(action_sequence, show_solution=True)
    print(maze.start, maze.end, maze.current, maze.solution)
    print(maze.maze)
    print(maze.get_current_maze())