import random
import json

LOCAL_NODES_PATH = '../results/clusters/local/'
REGIONAL_NODES_PATH = '../results/clusters/regional/'
CENTRAL_NODES_PATH = '../results/clusters/central/'

NUM_LOCAL_WAREHOUSES = 100
local_data = {}

for i in range(NUM_LOCAL_WAREHOUSES):
    loc_nodes = []

    NODES_PATH_i = f'{LOCAL_NODES_PATH}/local_{i}.txt'
    with open(NODES_PATH_i, 'r') as file:
        for line in file:
            # print(line)
            values = line.split()
            if values[0][0] == 'W':
                continue
            if values:
                first_value = int(values[0])
                loc_nodes.append([first_value, random.randint(1, 20)])
    
    local_data[f'{i}'] = loc_nodes

json_string = json.dumps(local_data, indent=2)

with open('../results/clusters/capacity/local.json', 'w') as json_file:
    json_file.write(json_string)


NUM_REGIONAL_WAREHOUSES = 10
regional_data = {}

for i in range(NUM_REGIONAL_WAREHOUSES):
    reg_nodes = []

    NODES_PATH_i = f'{REGIONAL_NODES_PATH}/regional_{i}.txt'
    with open(NODES_PATH_i, 'r') as file:
        for line in file:
            # print(line)
            values = line.split()
            if values[0][0] == 'W':
                continue
            if values:
                first_value = int(values[0])
                reg_nodes.append([first_value, random.randint(1, 20)])
    
    regional_data[f'{i}'] = reg_nodes

json_string = json.dumps(regional_data, indent=2)

with open('../results/clusters/capacity/regional.json', 'w') as json_file:
    json_file.write(json_string)


NUM_CENTRAL_WAREHOUSES = 3
central_data = {}

for i in range(NUM_CENTRAL_WAREHOUSES):
    cen_nodes = []

    NODES_PATH_i = f'{CENTRAL_NODES_PATH}/central_{i}.txt'
    with open(NODES_PATH_i, 'r') as file:
        for line in file:
            # print(line)
            values = line.split()
            if values[0][0] == 'W':
                continue
            if values:
                first_value = int(values[0])
                cen_nodes.append([first_value, random.randint(1, 20)])
    
    central_data[f'{i}'] = cen_nodes

json_string = json.dumps(central_data, indent=2)

with open('../results/clusters/capacity/central.json', 'w') as json_file:
    json_file.write(json_string)