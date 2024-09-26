import math
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence


class NQueens:
    def __init__(self, n_queens=8, mutation_rate=0.05, population_size=50, max_generations=50) -> None:
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations = max_generations
        self.n_queens = n_queens
        self.fig, self.ax = plt.subplots()

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
    
    def _get_parents(self, population: Sequence[np.ndarray], fitness_values: Sequence[int]) -> Sequence[Sequence[np.ndarray]]:
        all_parents_indices = []
        all_parents = []
        fitness_values_percentage = np.array(fitness_values) / np.sum(fitness_values)
        for _ in range(self.population_size):
            parents_indices = tuple(np.random.choice(self.population_size, size=2, p=fitness_values_percentage, replace=False))
            if parents_indices not in all_parents_indices:
                all_parents_indices.append(parents_indices)
        for parents_indices in all_parents_indices:
            all_parents.append([population[parents_indices[0]], population[parents_indices[1]]])
        return all_parents
    
    def _get_next_population(self, current_population: Sequence[np.ndarray], fitness_values: Sequence[int]) -> Sequence[Sequence[np.ndarray]]:
        best_from_population = np.argpartition(fitness_values, -self.population_size)[-self.population_size:]
        return [current_population[i] for i in best_from_population], [fitness_values[i] for i in best_from_population]

    def _plot_solution(self, solution: np.ndarray, fitness_val: int = -1, iteration: int = -1) -> None:
        self.ax.clear()
        self.ax.set_title(f"{self.n_queens}-Queens Problem Solution with Fitness = {fitness_val} at iteration {iteration+1}")
        for i in range(self.n_queens):
            for j in range(self.n_queens):
                if (i + j) % 2 != 0:
                    self.ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=True, color='white'))
                else:
                    self.ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=True, color='gray'))
        for i in range(self.n_queens):
            j = solution[i]
            self.ax.text(j+0.5, i+0.5, f'♕', ha='center', va='center', fontsize=int(40*8/self.n_queens), color='red')
        self.ax.set_xlim(0, self.n_queens)
        self.ax.set_ylim(0, self.n_queens)
        self.ax.set_aspect('equal')
        self.ax.axis(False)
        plt.pause(0.2)

    def __call__(self) -> Sequence[int]:
        max_fitness_value = math.comb(self.n_queens, 2)
        population = self._create_initial_generation()
        fitness_values = self._calculate_population_fitness(population)
        for i in range(self.max_generations):
            all_parents = self._get_parents(population, fitness_values)
            childrens = []
            for parent1, parent2 in all_parents:
                childrens.extend(self._cross_over(parent1, parent2))
            next_population = self._mutate(childrens)
            next_population.extend(population)
            next_population = np.unique(next_population, axis=0)
            fitness_values = self._calculate_population_fitness(next_population)
            population, fitness_values = self._get_next_population(next_population, fitness_values)
            best_solution_index = np.argmax(fitness_values)
            best_solution = population[best_solution_index]
            best_solution_fitness = fitness_values[best_solution_index]
            self._plot_solution(best_solution, best_solution_fitness, i)
            if best_solution_fitness == max_fitness_value:
                break
        plt.show()
        return best_solution.tolist()


if __name__ == "__main__":
    nq = NQueens(n_queens=15, mutation_rate=0.1, population_size=100, max_generations=150)
    print("Solution is", nq())