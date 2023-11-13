from utils import *
from ga import GeneticAlgorithm

BASE_PATH = '../data/world'
NODES_PATH = '../results/clusters/local/local_0.txt'
EDGES_PATH = BASE_PATH + '/world_edges.txt'

GraphBuilder = GraphBuilder(NODES_PATH, EDGES_PATH)
graph = GraphBuilder.build_graph()


print("Generating Optimal Route...")

# Specify a subset of nodes if needed
subset_nodes = [2499 ,3816, 3817,3819, 3820, 3821, 3822, 3824, 3825, 3826, 3827, 3828, 3829, 3830] 

# Create a GeneticAlgorithm instance and run the algorithm
genetic_algorithm = GeneticAlgorithm(graph, subset_nodes , population_size=100, generations=500)
best_order = genetic_algorithm.evolve()
print("Best order:", best_order)