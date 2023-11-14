from utils import *
from ga import GeneticAlgorithm
import json
import random

BASE_PATH = '../data/world'
LOCAL_NODES_PATH = '../results/clusters/local/'
REGIONAL_NODES_PATH = '../results/clusters/regional/'
CENTRAL_NODES_PATH = '../results/clusters/central/'
EDGES_PATH = BASE_PATH + '/world_edges.txt'

NUM_LOCAL_WAREHOUSES = 100
local_ordering = []

for i in range(NUM_LOCAL_WAREHOUSES):
    NODES_PATH_i = f'{LOCAL_NODES_PATH}/local_{i}.txt'
    graphBuilder = GraphBuilder(NODES_PATH_i, EDGES_PATH)
    graph = graphBuilder.build_graph()

    print(f'Generating Optimal Route for {i}th local warehouse... ')

    subset_nodes = []
    with open(NODES_PATH_i, 'r') as file:
        for line in file:
            # print(line)
            values = line.split()
            if values[0][0] == 'W':
                continue
            if values:
                first_value = int(values[0])
                subset_nodes.append((first_value, random.randint(1, 20)))

    vehicles = [1,2]

    genetic_algorithm = GeneticAlgorithm(graph, vehicles, subset_nodes , population_size=100, generations=1)
    best_order = genetic_algorithm.evolve()
    local_ordering.append(best_order)
    print(f"Best order for {i}th local warehouse: ", best_order)

output_file_path = '../results/clusters/orders/local_ordering.json'

with open(output_file_path, 'w') as json_file:
    json.dump(local_ordering, json_file)


NUM_REGIONAL_WAREHOUSES = 10
regional_ordering = []

for i in range(NUM_REGIONAL_WAREHOUSES):
    NODES_PATH_i = f'{REGIONAL_NODES_PATH}/regional_{i}.txt'
    graphBuilder = GraphBuilder(NODES_PATH_i, EDGES_PATH)
    graph = graphBuilder.build_graph()

    print(f'Generating Optimal Route for {i}th regional warehouse... ')

    subset_nodes = [] 
    with open(NODES_PATH_i, 'r') as file:
        for line in file:
            # print(line)
            values = line.split()
            if values[0][0] == 'W':
                continue
            if values:
                first_value = int(values[0])
                subset_nodes.append((first_value, random.randint(1, 20)))

    vehicles = [1,2]

    genetic_algorithm = GeneticAlgorithm(graph, vehicles, subset_nodes , population_size=100, generations=1)
    best_order = genetic_algorithm.evolve()
    regional_ordering.append(best_order)
    print(f"Best order for {i}th regional warehouse: ", best_order)

output_file_path = '../results/clusters/orders/regional_ordering.json'

with open(output_file_path, 'w') as json_file:
    json.dump(regional_ordering, json_file)


NUM_CENTRAL_WAREHOUSES = 3
central_ordering = []

for i in range(NUM_CENTRAL_WAREHOUSES):
    NODES_PATH_i = f'{CENTRAL_NODES_PATH}/central_{i}.txt'
    graphBuilder = GraphBuilder(NODES_PATH_i, EDGES_PATH)
    graph = graphBuilder.build_graph()

    print(f'Generating Optimal Route for {i}th central warehouse... ')

    subset_nodes = [] 
    with open(NODES_PATH_i, 'r') as file:
        for line in file:
            # print(line)
            values = line.split()
            if values[0][0] == 'W':
                continue
            if values:
                first_value = int(values[0])
                subset_nodes.append((first_value, random.randint(1, 20)))

    vehicles = [1,2]

    genetic_algorithm = GeneticAlgorithm(graph, vehicles, subset_nodes , population_size=100, generations=1)
    best_order = genetic_algorithm.evolve()
    central_ordering.append(best_order)
    print(f"Best order for {i}th central warehouse: ", best_order)

output_file_path = '../results/clusters/orders/central_ordering.json'

with open(output_file_path, 'w') as json_file:
    json.dump(central_ordering, json_file)

