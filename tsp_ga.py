import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

class TSP_GA:
    def __init__(self, mutation_rate=0.05, population_size=50, max_generations=50, early_stopping_count=None) -> None:
        self.num_cities = 0
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations = max_generations
        self.early_stopping_count = early_stopping_count if early_stopping_count is not None else max_generations

    def distance_fn(self, loc1, loc2):
        return np.sqrt(((loc1 - loc2)**2).sum())
    
    def find_total_distance_fn(self, path):
        distance = 0
        for i in range(self.num_cities - 1):
            distance += self.distance_fn(path[i], path[i + 1])
        return distance
    
    def get_initial_population(self):
        cities = np.arange(self.num_cities)
        return np.array([np.random.permutation(cities) for _ in range(self.population_size)])
    
    def get_new_population(self, current_population, population_distances):
        population_fitness_exp = np.exp(-population_distances) # Fitness is inverse of distance 
        best_parents_prob = population_fitness_exp / population_fitness_exp.sum()

        new_population = current_population
        while(len(new_population) < self.population_size*2):
            parent1, parent2 = self.get_parents(current_population, best_parents_prob)
            child1, child2 = self.get_children(parent1, parent2)
            new_population = np.append(new_population, child1[np.newaxis, :], axis=0)
            new_population = np.append(new_population, child2[np.newaxis, :], axis=0)
        return new_population
    
    def get_parents(self, population, best_parents_prob):
        parent1 = np.random.choice(self.population_size, p=best_parents_prob)
        parent2 = np.random.choice(self.population_size, p=best_parents_prob)

        while parent1 == parent2:
            parent2 = np.random.choice(self.population_size, p=best_parents_prob)

        return population[parent1], population[parent2]
    
    def get_children(self, parent1, parent2):        
        child1 = self.crossover(parent1, parent2)
        child2 = self.crossover(parent2, parent1)
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)
        return child1, child2
    
    def mutate(self, child):
        if np.random.random() < self.mutation_rate:
            if np.random.rand() < 0.5:
                child = self.swap_mutate(child)
            else:
                child = self.rotate_mutate(child)
        return child

    def rotate_mutate(self, child):
        index1, index2 = sorted(self.get_random_indices())
        child[index1:index2] = child[index1:index2][::-1]
        return child

    def swap_mutate(self, child):
        index1, index2 = self.get_random_indices()
        child[index1], child[index2] = child[index2], child[index1]
        return child

    def get_random_indices(self):
        index1 = np.random.randint(0, self.num_cities)
        index2 = np.random.randint(0, self.num_cities)

        while (index2 == index1):
            index2 = np.random.randint(0, self.num_cities)
        
        return index1, index2

    def crossover(self, parent1, parent2):
        random_index = np.random.randint(1, self.num_cities)
        child = parent1[:random_index]
        for city in parent2:
            if city not in child:
                child = np.append(child, city)
        return child
    
    def __call__(self, cities_locations=[]):
        self.num_cities = len(cities_locations)
        self.cities_locations = np.array(cities_locations)

        early_stopper_counter = 0
        last_best_soln = 0

        population = self.get_initial_population()
        plt.xlim(min(self.cities_locations[:, 0]) - 1, max(self.cities_locations[:, 0]) + 1)
        plt.ylim(min(self.cities_locations[:, 1]) - 1, max(self.cities_locations[:, 1]) + 1)

        for i, (x, y) in enumerate(self.cities_locations):
            plt.text(x, y, f"Location {i}")

        self.find_total_distance = lambda individual: self.find_total_distance_fn(self.cities_locations[individual])

        for gen_count in range(self.max_generations):
            population_distances = np.array(list(map(self.find_total_distance, population)))
            population_fitness_exp = np.exp(-population_distances) # Fitness is inverse of distance 
            best_parents_prob = population_fitness_exp / population_fitness_exp.sum()

            best_indices = np.argsort(best_parents_prob, )[-self.population_size:]
            population = population[best_indices]
            population_distances = population_distances[best_indices]
            best_solution = self.cities_locations[population[-1]]

            plt.clf()
            for i, (x, y) in enumerate(self.cities_locations):
                plt.text(x, y, f"Town{i}")
            plt.plot(best_solution[:, 0], best_solution[:, 1])
            plt.title(f"Distance:{population_distances[-1]:.2f}\nGeneration:{gen_count+1}")
            plt.pause(0.2)

            if population_distances[-1] == last_best_soln:
                early_stopper_counter += 1
                if early_stopper_counter == self.early_stopping_count:
                    break
            else:
                early_stopper_counter = 0
            last_best_soln = population_distances[-1]

            population = self.get_new_population(population, population_distances)
    
        plt.show()


tsp = TSP_GA(mutation_rate=0.05, population_size=100, max_generations=150)
# tsp([[2, 1],
#     [7, 2],
#     [7, 8],
#     [1, 3],
#     [4, 8],
#     [1, 2],
#     [3, 7],
#     [5, 5],
#     [6, 6],
#     [6, 9],
#     [5, 7],
#     [9, 6]])    

tsp(np.random.randint(0, 10, (20, 2)))