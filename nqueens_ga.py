import numpy as np
import matplotlib.pyplot as plt
import random
from collections.abc import Sequence


class NQueens:
    def __init__(self, n_queens=8, mutation_rate=0.05, population_size=50, max_generations=50) -> None:
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations = max_generations
        self.n_queens = n_queens

    def _create_initial_generation(self) -> Sequence[np.ndarray]:
        return [np.random.randint(0, self.n_queens, (self.n_queens,)) for _ in range(self.population_size)]

    def _cross_over(self, parent1: np.ndarray, parent2: np.ndarray) -> Sequence[np.ndarray]:
        random_index = np.random.randint(1, self.n_queens-1)
        return [np.concatenate([parent1[:random_index], parent2[random_index:]]), np.concatenate([parent2[:random_index], parent1[random_index:]])]
    
    def _mutate(self, childrens: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        for child in childrens:
            if np.random.random() <= self.mutation_rate:
                random_index = np.random.randint(0, self.n_queens)
                child[random_index] = np.random.randint(0, self.n_queens)
        return childrens
    
    def _different_col_or_diag(self, row1: int, col1: int, row2: int, col2: int) -> int:
        if col1 == col2:
            return 0
        elif abs(row1 - row2) == abs(col1 - col2):
            return 0
        return 1
    
    def _fitness_fn(self, nqueen_seq: np.ndarray) -> int:
        fitness_val = 0
        for row1 in range(self.n_queens):
            col1 = nqueen_seq[row1]
            for row2 in range(row1+1, self.n_queens):
                col2 = nqueen_seq[row2]
                fitness_val += self._different_col_or_diag(row1, col1, row2, col2)
        return fitness_val
    
    def _calculate_population_fitness(self, population: Sequence[np.ndarray]) -> Sequence[int]:
        fitness_values = []
        for nqueen_seq in population:
            fitness_values.append(self._fitness_fn(nqueen_seq))
        return fitness_values
    
    
    

    def __call__(self) -> Sequence[int]:
        population = self._create_initial_generation()

def _fitness_fn(nqueen_seq: np.ndarray) -> int:
    fitness_val = 0
    for row1 in range(8):
        col1 = nqueen_seq[row1]
        for row2 in range(row1+1, 8):
            col2 = nqueen_seq[row2]
            fitness_val += _different_col_or_diagonal(row1, col1, row2, col2)
    return fitness_val

def _different_col_or_diagonal(row1: int, col1: int, row2: int, col2: int) -> int:
        if col1 == col2:
            return 0
        elif abs(row1 - row2) == abs(col1 - col2):
            return 0
        return 1

if __name__ == "__main__":
    print(_fitness_fn(np.array([0, 1, 3, 5, 6, 6, 7, 0])))
    print(_fitness_fn(np.array([4, 6, 0, 3, 1, 7, 5, 2])))
    print(np.random.multinomial(1, [0.25, 0.25, 0.25, 0.25], size=(2,)))