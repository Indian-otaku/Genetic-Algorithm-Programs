import numpy as np
import math

class NNChromosome:
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, output_size, initialize_weights=True): 
        self.reward = 0       
        self.input_size = input_size if isinstance(input_size, int) else math.prod(input_size[1:])
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        if initialize_weights:
            self.weights = []
            self.biases = []
            for i in range(num_hidden_layers + 1):
                if i == 0:
                    self.weights.append(np.random.randn(self.input_size, self.hidden_layer_size)*2-1)
                    self.biases.append(np.random.randn(self.hidden_layer_size)*2-1)
                elif i == self.num_hidden_layers:
                    self.weights.append(np.random.randn(self.hidden_layer_size, self.output_size)*2-1)
                    self.biases.append(np.random.randn(self.output_size)*2-1)
                else:
                    self.weights.append(np.random.randn(self.hidden_layer_size, self.hidden_layer_size)*2-1)
                    self.biases.append(np.random.randn(self.hidden_layer_size)*2-1)

    def activation(self, x):
        return np.where(x>=1, 1, np.where(x<=0, 0, x))
    
    def output_activation(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, input):
        output = input.reshape(-1, self.input_size)
        for i in range(self.num_hidden_layers):
            output = np.matmul(output, self.weights[i]) + self.biases[i]
            output = self.activation(output)
        output = np.matmul(output, self.weights[self.num_hidden_layers]) + self.biases[self.num_hidden_layers]
        output = self.output_activation(output)
        return output
    
    def mutate(self, mutation_rate=0.01):
        new_nnchromo = NNChromosome(self.input_size, self.num_hidden_layers, self.hidden_layer_size, self.output_size, initialize_weights=False)
        new_nnchromo.weights = self.weights
        new_nnchromo.biases = self.biases
        for i, weight in enumerate(new_nnchromo.weights):
            weight_mutate_prob = np.random.rand(*weight.shape)
            new_nnchromo.weights[i] = np.where(weight_mutate_prob <= mutation_rate, weight + (np.random.randn(*weight.shape)-0.5)/2, weight)
        for i, bias in enumerate(new_nnchromo.biases):
            bias_mutate_prob = np.random.rand(*bias.shape)
            new_nnchromo.biases[i] = np.where(bias_mutate_prob <= mutation_rate, bias + (np.random.randn(*bias.shape)-0.5)/2, bias)
        return new_nnchromo
    


if __name__ == "__main__":
    from maze_object import Maze
    from maze_generator import GenerateMaze

    x = []
    for i in range(32):
        maze_gen = GenerateMaze(10, 10, 0.5)
        maze = Maze(maze_gen.maze, maze_gen.start, maze_gen.end, maze_gen.solution, render_maze=True)
        x.append(maze.maze)
    x = np.array(x)
    nn = NNChromosome(x.shape, 3, 20, 2)
    print(nn.forward(x))
    print(x.shape, nn.forward(x).shape)
    nn = nn.mutate(0.5)
    print(nn.forward(x))