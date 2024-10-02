import math
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence


class NQueens:
    def __init__(self, n_queens=8, mutation_rate=0.05, population_size=50, max_generations=50) -> None:
        """
        Initialize the NQueens object.

        Parameters
        ----------
        n_queens : int, optional
            The number of queens in the chessboard. Defaults to 8.
        mutation_rate : float, optional
            The probability of mutation. Defaults to 0.05.
        population_size : int, optional
            The size of the population. Defaults to 50.
        max_generations : int, optional
            The maximum number of generations. Defaults to 50.
        """

        if n_queens < 4:
            raise ValueError("The number of queens must be at least 4.")
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations = max_generations
        self.n_queens = n_queens
        self.fig, self.ax = plt.subplots()

    def _create_initial_generation(self) -> Sequence[np.ndarray]:
        """
        Create an initial generation of chromosomes for the genetic algorithm.

        Each chromosome is a one-dimensional array of length n_queens, where
        each element is a random integer between 0 and n_queens-1. Each element
        in the array represents the column value of a queen in the chessboard.
        The index of each element is the row value of the queen. For example,
        [3, 0, 4, 2, 1] represents a configuration where the first queen is
        at row 0 and column 3, the second queen is at row 1 and column 0, and
        so on. The representation is a list of arrays and not a 2D array.

        Returns
        -------
        list of numpy.ndarray
            A list of one-dimensional numpy arrays, each representing a
            chromosome in the initial generation.
        """
        
        return [np.random.randint(0, self.n_queens, (self.n_queens,)) for _ in range(self.population_size)]

    def _cross_over(self, parent1: np.ndarray, parent2: np.ndarray) -> Sequence[np.ndarray]:
        """
        Perform a crossover operation between two parent chromosomes.

        A crossover operation is where we combine two parent chromosomes to
        create two new offspring chromosomes. The crossover point is chosen
        randomly between 1 and n_queens-1. The portion of the parent
        chromosome before the crossover point is copied to the first offspring
        chromosome, and the portion after the crossover point is copied to the
        second offspring chromosome, and vice versa.

        Parameters
        ----------
        parent1 : numpy.ndarray
            The first parent chromosome.
        parent2 : numpy.ndarray
            The second parent chromosome.

        Returns
        -------
        list of numpy.ndarray
            A list of two offspring chromosomes.
        """
        
        random_index = np.random.randint(1, self.n_queens-1)
        return [np.concatenate([parent1[:random_index], parent2[random_index:]]), np.concatenate([parent2[:random_index], parent1[random_index:]])]
    
    def _mutate(self, childrens: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        Perform a mutation operation on the given sequence of children chromosomes.

        Each chromosome undergoes a mutation with a probability given by the
        mutation_rate attribute of the object. If a chromosome is to be mutated,
        a random index is chosen and the value at that index is replaced with
        a random integer between 0 and n_queens-1.

        Parameters
        ----------
        childrens : Sequence[numpy.ndarray]
            A sequence of one-dimensional numpy arrays, each representing a
            chromosome.

        Returns
        -------
        list of numpy.ndarray
            The same sequence of chromosomes, but with some of them mutated.
        """

        for child in childrens:
            if np.random.random() <= self.mutation_rate:
                random_index = np.random.randint(0, self.n_queens)
                child[random_index] = np.random.randint(0, self.n_queens)
        return childrens
    
    def _different_col_or_diag(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Checks if two positions on a chessboard are on the same column or on the same diagonal.

        Parameters
        ----------
        row1 : int
            The row of the first position.
        col1 : int
            The column of the first position.
        row2 : int
            The row of the second position.
        col2 : int
            The column of the second position.

        Returns
        -------
        int
            0 if the positions are on the same column or on the same diagonal, 1 otherwise.
        """
        
        if col1 == col2:
            return 0
        elif abs(row1 - row2) == abs(col1 - col2):
            return 0
        return 1
    
    def _fitness_fn(self, nqueen_seq: np.ndarray) -> int:
        """
        Calculates the fitness value of a given sequence of queen positions.

        The fitness value is the number of pairs of queens that do not attack each
        other, i.e. are not on the same column or diagonal.

        Parameters
        ----------
        nqueen_seq : numpy.ndarray
            A one-dimensional numpy array of length n_queens, where each element
            is the column index of a queen in the corresponding row.

        Returns
        -------
        int
            The fitness value of the given sequence of queen positions.
        """
        
        fitness_val = 0
        for row1 in range(self.n_queens):
            col1 = nqueen_seq[row1]
            for row2 in range(row1+1, self.n_queens):
                col2 = nqueen_seq[row2]
                fitness_val += self._different_col_or_diag(row1, col1, row2, col2)
        return fitness_val
    
    def _calculate_population_fitness(self, population: Sequence[np.ndarray]) -> Sequence[int]:
        """
        Calculates the fitness values of all sequences in a given population.

        Parameters
        ----------
        population : Sequence[numpy.ndarray]
            A sequence of one-dimensional numpy arrays, each representing a
            sequence of queen positions.

        Returns
        -------
        list of int
            A list of fitness values, each corresponding to a sequence in the
            given population.
        """
        
        fitness_values = []
        for nqueen_seq in population:
            fitness_values.append(self._fitness_fn(nqueen_seq))
        return fitness_values
    
    def _get_parents(self, population: Sequence[np.ndarray], fitness_values: Sequence[int]) -> Sequence[Sequence[np.ndarray]]:
        """
        Gets a list of pairs of parent sequences from a given population,
        the selection of which is based on the given fitness values.

        Parameters
        ----------
        population : Sequence[numpy.ndarray]
            A sequence of one-dimensional numpy arrays, each representing a
            sequence of queen positions.
        fitness_values : Sequence[int]
            A sequence of fitness values, each corresponding to a sequence in
            the given population.

        Returns
        -------
        list of list of numpy.ndarray
            A list of pairs of one-dimensional numpy arrays, each pair
            representing two parent sequences that will be used for
            crossover.
        """

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
        """
        Gets the next population from the current one by selecting the
        population size best individuals from the current population.

        Parameters
        ----------
        current_population : Sequence[numpy.ndarray]
            A sequence of one-dimensional numpy arrays, each representing a
            sequence of queen positions.
        fitness_values : Sequence[int]
            A sequence of fitness values, each corresponding to a sequence in
            the given population.

        Returns
        -------
        list of list of numpy.ndarray
            A list of one-dimensional numpy arrays, each representing a
            sequence of queen positions in the next population.
        """
        
        best_from_population = np.argpartition(fitness_values, -self.population_size)[-self.population_size:]
        return [current_population[i] for i in best_from_population], [fitness_values[i] for i in best_from_population]

    def _plot_solution(self, solution: np.ndarray, fitness_val: int = -1, iteration: int = -1) -> None:
        """
        Plots a solution to the N-Queens problem.

        Parameters
        ----------
        solution : numpy.ndarray
            A one-dimensional numpy array of length n_queens, where each
            element is the column index of a queen in the corresponding row.
        fitness_val : int, optional
            The fitness value of the solution. Defaults to -1.
        iteration : int, optional
            The iteration number at which the solution was found. Defaults to -1.

        Returns
        -------
        None
        """
        
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
        """
        Solves the N-Queens problem using a genetic algorithm.

        Returns
        -------
        list of int
            A sequence of column indices of the queens in the solution, where
            the index of each element is the row index of the queen in the
            corresponding column.
        """

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
    nq = NQueens(n_queens=20, mutation_rate=0.1, population_size=200, max_generations=150)
    print("Solution is", nq())