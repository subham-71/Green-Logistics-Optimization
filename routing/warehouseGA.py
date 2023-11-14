import numpy as np
import random
import matplotlib.pyplot as plt
from routingGA import RoutingGeneticAlgorithm
import os

class Cluster:
    def __init__(self, nodes):
        self.nodes = nodes
        self.fitness = 0

class Population:
    def __init__(self, clusters):
        self.clusters = clusters
        self.fitness = 0

class WarehouseGeneticAlgorithm:
    def __init__(self, graph, n_clusters=12, population_size=100, generations=1, crossover_prob=0.7, mutation_prob=0.2):
        self.graph = graph
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []
        self.edge_weights = self.calculate_edge_weights()

    
    def calculate_edge_weights(self):
        num_nodes = len(self.graph.nodes)
        edge_weights = np.zeros((num_nodes, num_nodes))

        edges = self.graph.edges

        node_indices = {node: i for i, node in enumerate(self.graph.nodes)}

        for edge in edges:
            i, j = node_indices[edge.node1], node_indices[edge.node2]
            edge_weights[i, j] = edge_weights[j, i] = edge.weight[1]

        return edge_weights


    def initialize_population(self):

        print("Initializing Population...")
        for _ in range(self.population_size):
            clusters = self.initialize_clusters()
            population = Population(clusters)
            self.population.append(population)

    def initialize_clusters(self):
        # Initialize clusters randomly using NumPy

        nodes = np.random.permutation(self.graph.nodes)
        cluster_size = len(nodes) // self.n_clusters

        clusters = [Cluster(nodes[i:i + cluster_size]) for i in range(0, len(nodes), cluster_size)]
        return clusters

    def convert_clusters_to_nodes_list(self, clusters):
        # Convert a list of Cluster instances to a list of node lists
        return [cluster.nodes for cluster in clusters]

    def convert_nodes_list_to_clusters(self, nodes_list):
        # Convert a list of node lists to a list of Cluster instances
        return [Cluster(nodes) for nodes in nodes_list]

    def evaluate_population(self):
        # Evaluate the fitness of each individual in the population
        fitness_values = []
        for population in self.population:
            fitness = self.calculate_fitness(population.clusters)
            fitness_values.append(fitness)

            # Attach fitness value to the population
            population.fitness = fitness

    def get_routing_fitness(self, cluster):
        vehicles = [1,2]
        genetic_algorithm = RoutingGeneticAlgorithm(self.graph, vehicles, [(node.id,random.randint(1,10)) for node in cluster.nodes ] , population_size=100, generations=1)  
        best_order , best_fitness = genetic_algorithm.evolve()
        return best_fitness

    def calculate_fitness(self, clusters):
        total_fitness = 0
        for cluster in clusters:
            if len(cluster.nodes) > 0:
                intra_cluster_distance = self.calculate_intra_cluster_distance(cluster.nodes)
                routing_fitness = self.get_routing_fitness(cluster)
                total_fitness += intra_cluster_distance 
                total_fitness += routing_fitness
        return total_fitness

    def calculate_intra_cluster_distance(self, cluster_nodes):
        # Extract node indices
        indices = [self.graph.nodes.index(node) for node in cluster_nodes]
        indices = np.array(indices, dtype=int)
        intra_cluster_distances = self.edge_weights[indices[:, None], indices].sum()
        return intra_cluster_distances

    def select_parents(self):
        # Tournament selection
        parents = []
        for _ in range(self.population_size):
            tournament_size = random.randint(2, min(self.population_size, len(self.population)))
            tournament = random.sample(self.population, tournament_size)

            # Select the population with the highest fitness in each tournament
            best_population = max(tournament, key=lambda x: x.fitness)
            parents.append(best_population)
        return parents


    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        crossover_point = random.randint(0, len(parent1.clusters) - 1)
        child_clusters = self.convert_nodes_list_to_clusters(
            self.convert_clusters_to_nodes_list(parent1.clusters)[:crossover_point] +
            self.convert_clusters_to_nodes_list(parent2.clusters)[crossover_point:]
        )
        child = Population(child_clusters)
        return child
    
    def generate_blue_shades(self,n_clusters):
        start_color = (0.2, 0.4, 0.6)
        end_color = (0.1, 0.1, 0.6)

        # Calculate the step size for each color component
        step_size = tuple((end - start) / max(1, n_clusters - 1) for start, end in zip(start_color, end_color))

        # Generate blue shades
        shades = [tuple(start + step * i for start, step in zip(start_color, step_size)) for i in range(n_clusters)]

        return shades

    def mutate(self, population):
        # Perform mutation on a set of clusters
        cluster_index = random.randint(0, len(population.clusters) - 1)
        cluster = population.clusters[cluster_index]
        mutation_point1 = random.randint(0, len(cluster.nodes) - 1)
        mutation_point2 = random.randint(0, len(cluster.nodes) - 1)
        cluster.nodes[mutation_point1], cluster.nodes[mutation_point2] = cluster.nodes[mutation_point2], cluster.nodes[mutation_point1]
    
    def plot_clusters(self, clusters_map, iteration):
        # Create a directory to save the plots
        results_directory = "../results/warehouses"
        
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        hardcoded_colors = self.generate_blue_shades(len(clusters_map))
        
        # Get node coordinates
        node_coordinates = {node.id: (node.x, node.y) for node in self.graph.nodes}

        # Plot the graph with clusters and centroids
        for idx, (cluster_id, cluster_nodes) in enumerate(clusters_map.items()):
            cluster_color = hardcoded_colors[idx]
            cluster_x = [node_coordinates[node_id][0] for node_id in cluster_nodes]
            cluster_y = [node_coordinates[node_id][1] for node_id in cluster_nodes]
            plt.scatter(cluster_x, cluster_y, color=cluster_color, s=50, label=cluster_id)
            centroid_x = np.mean(cluster_x)
            centroid_y = np.mean(cluster_y)
            plt.scatter(centroid_x, centroid_y, marker='o', color='black', edgecolor='black', s=200)

        plt.title(f'Iteration {iteration}')
        plt.legend()
        plt.savefig(os.path.join(results_directory, f'iteration_{iteration}.png'))
        plt.clf()

    def run(self):
        self.initialize_population()
        # Create a map of cluster ID to nodes for the best clusters
        best_clusters_map = {}

        print("Warehouse Generations : " , self.generations)

        for generation in range(self.generations):
            print(f'Generation {generation}')
            parents = self.select_parents()
            offspring = []

            for i in range(0, self.population_size, 2):
                if random.random() < self.crossover_prob:
                    child = self.crossover(parents[i], parents[i + 1])
                else:
                    child = parents[i]

                if random.random() < self.mutation_prob:
                    self.mutate(child)

                offspring.append(child)

            self.population = offspring
            self.evaluate_population()

            # Return the best population
            best_population = max(self.population, key=lambda x: x.fitness)
            
            for idx, cluster in enumerate(best_population.clusters):
                cluster_id = f"Cluster_{idx}"
                best_clusters_map[cluster_id] = [node.id for node in cluster.nodes]
            
            # Plot the clusters for each generation
            # if generation % 50 == 0:
            self.plot_clusters(best_clusters_map, int(generation))

        return best_clusters_map

