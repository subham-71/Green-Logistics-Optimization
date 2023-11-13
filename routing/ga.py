import random
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

class GeneticAlgorithm:
    def __init__(self, graph, vehicles, subset_nodes=None, population_size=100, generations=500):
        self.graph = graph
        self.subset_nodes = subset_nodes if subset_nodes is not None else [(node.id, 1) for node in self.graph.nodes]
        self.population_size = population_size
        self.generations = generations
        self.vehicles_df = self.get_vehicle_info('../data/world/vehicles.csv', vehicles)
        self.co2_emissions = self.get_vehicle_emissions(self.vehicles)
        self.capacity = self.get_vehicles_capacity(self.vehicles_df, vehicles)
        
    def get_vehicles_info(self, path, vehicles):
        df = pd.read_csv(path)
        filtered_df = df[df['Vehicle ID'].isin(vehicles)]
        return filtered_df

    def get_vehicles_capacity(self, vehicle_df):
        capacity_map = dict(zip(vehicle_df['Vehicle ID'], vehicle_df['Capacity (cubic feet)']))
        return capacity_map

    def get_emission_prediction(self, sample):
        loaded_model = tf.keras.models.load_model('../ml-modules/models/emission_model')
        return loaded_model.predict(sample).tolist()[0][0]

    def get_vehicle_emmisions(self, vehicles_df):
        features_columns = ['Capacity (cubic feet)', 'Engine Size(L)', 'Cylinders', 'Transmission', 
                         'Fuel Type', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 
                         'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']

        predictions = {row['Vehicle ID']: self.get_emission_prediction(np.array([row[features_columns]])) for _, row in vehicles_df.iterrows()}

        return predictions

    def initialize_population(self):
        population = []

        for _ in range(self.population_size):
            individual = self.random_individual()
            population.append(individual)

        return population

    def random_individual(self):
        individual = {}

        # Assign nodes to vehicles based on capacity constraints
        remaining_nodes = set(self.subset_nodes)

        while remaining_nodes:
            # Randomly select a vehicle
            vehicle_id = random.choice(self.vehicle_data.index)

            # Select nodes that fit in the vehicle's capacity
            vehicle_capacity = self.vehicle_data.loc[vehicle_id]['capacity']
            nodes_for_vehicle = set()

            for node, delivery_capacity in remaining_nodes:
                if self.graph.get_node_weight(node) * delivery_capacity <= vehicle_capacity:
                    nodes_for_vehicle.add(node)
                    vehicle_capacity -= self.graph.get_node_weight(node) * delivery_capacity

            # Add the vehicle assignment to the individual
            individual[vehicle_id] = list(nodes_for_vehicle)

            # Remove assigned nodes from the remaining set
            remaining_nodes -= nodes_for_vehicle

        return individual

    def calculate_fitness(self, individual):
        total_fitness = 0

        for vehicle_id, nodes in individual.items():
            route_duration = self.calculate_route_duration(nodes)
            carbon_emissions = self.calculate_carbon_emissions(nodes)

            # You can define your own formula for combining duration, time, and carbon emissions
            fitness_value = route_duration * carbon_emissions

            total_fitness += fitness_value

        return total_fitness

    def calculate_route_duration(self, nodes):
        total_duration = 0

        for i in range(len(nodes) - 1):
            edge = self.graph.get_edge_by_nodes(nodes[i][0], nodes[i + 1][0])

            # Check if the edge exists
            if edge is not None:
                total_duration += edge.weight[0]
            else:
                total_duration += 1000  # Penalize for non-existent edges

        # Add duration for returning to the starting node
        return total_duration

    def crossover(self, parent1, parent2):
        child = {}

        for vehicle_id in set(parent1.keys()) | set(parent2.keys()):
            # Crossover at the route level
            if random.random() < 0.5:
                child[vehicle_id] = copy.deepcopy(parent1.get(vehicle_id, []))
            else:
                child[vehicle_id] = copy.deepcopy(parent2.get(vehicle_id, []))

        # Ensure the child's routes adhere to capacity constraints
        self.adjust_routes_capacity(child)

        return child

    def mutate(self, individual):
        mutation_point = random.choice(list(individual.keys()))

        # Mutate a random route
        route_to_mutate = individual[mutation_point]

        # Modify the route (add/remove nodes) to adhere to capacity constraints
        individual[mutation_point] = self.adjust_route_capacity(mutation_point, route_to_mutate)

    def adjust_routes_capacity(self, individual):
        for vehicle_id, nodes in individual.items():
            individual[vehicle_id] = self.adjust_route_capacity(vehicle_id, nodes)

    def adjust_route_capacity(self, vehicle_id, nodes):
        # Adjust the route to adhere to capacity constraints
        vehicle_capacity = self.vehicle_data.loc[vehicle_id]['capacity']
        remaining_capacity = vehicle_capacity

        # Filter nodes based on capacity constraints
        valid_nodes = []

        for node, delivery_capacity in nodes:
            if self.graph.get_node_weight(node) * delivery_capacity <= remaining_capacity:
                valid_nodes.append((node, delivery_capacity))
                remaining_capacity -= self.graph.get_node_weight(node) * delivery_capacity

        return valid_nodes

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