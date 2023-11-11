import random
import copy

class GeneticAlgorithm:
    def __init__(self, graph, subset_nodes=None, population_size=100, generations=500):
        self.graph = graph
        self.subset_nodes = subset_nodes
        self.population_size = population_size
        self.generations = generations

    def initialize_population(self):
        population = []

        for _ in range(self.population_size):
            order = self.random_order()
            population.append(order)

        return population

    def random_order(self):
        if self.subset_nodes is not None:
            order = copy.deepcopy(self.subset_nodes)
        else:
            order = [node.id for node in self.graph.nodes]
        random.shuffle(order)
        return order

    def calculate_fitness(self, order):
        total_duration = 0

        for i in range(len(order) - 1):
            edge = self.graph.get_edge_by_nodes(order[i], order[i + 1])

            # Check if the edge exists
            if edge is not None:
                total_duration += edge.weight[0]
            else:
                total_duration += 1000

        # Add duration for returning to the starting node
        return 1 / (total_duration)


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
