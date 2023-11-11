import random
import copy

class GeneticAlgorithm:
    def __init__(self, graph, population_size, generations):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations

    def initialize_population(self):
        population = []

        for _ in range(self.population_size):
            order = self.random_order()
            population.append(order)

        return population

    def random_order(self):
        order = copy.deepcopy(self.graph.nodes)
        random.shuffle(order)
        return order

    def calculate_fitness(self, order):
        total_duration = 0
        for i in range(len(order) - 1):
            edge = self.graph.get_edge_by_nodes(order[i], order[i + 1])
            total_duration += edge.weight

        # Add duration for returning to the starting node
        total_duration += self.graph.get_edge_by_nodes(order[-1], order[0]).weight

        return 1 / total_duration  # Invert for maximization

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1) - 1)
        child = parent1[:crossover_point]

        for node in parent2:
            if node not in child:
                child.append(node)

        return child

    def mutate(self, order):
        mutation_point1 = random.randint(0, len(order) - 1)
        mutation_point2 = random.randint(0, len(order) - 1)

        order[mutation_point1], order[mutation_point2] = order[mutation_point2], order[mutation_point1]

    def evolve(self):
        population = self.initialize_population()

        for generation in range(self.generations):
            population = sorted(population, key=lambda x: self.calculate_fitness(x), reverse=True)

            # Select top 50% of the population as parents
            parents = population[:len(population) // 2]

            # Generate offspring through crossover and mutation
            offspring = []

            while len(offspring) < len(population) - len(parents):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)

                if random.random() < 0.5:
                    self.mutate(child)

                offspring.append(child)

            # Combine parents and offspring for the next generation
            population = parents + offspring

        # Return the best order from the final generation
        best_order = max(population, key=lambda x: self.calculate_fitness(x))
        return best_order