import numpy
import random

#GENETIC ALGORITHM OPERATORS

#Mating function
def mating_pool(pop_inputs, objective, num_parents):
    
    objective = numpy.asarray(objective)
    parents = [[None,None,None, None, None, None, None, None, None]]* num_parents
    for parent_num in range(num_parents):
        best_fit_index = numpy.where(objective == numpy.max(objective))
        best_fit_index = best_fit_index[0][0]
        parents[parent_num] = pop_inputs[best_fit_index, :]
        objective[best_fit_index] = -9999999
    return parents

#Crossover function
def crossover(parents, offspring_size):
    
    offspring = [[None,None,None, None, None, None, None, None, None]]* offspring_size[0]
    crossover_loc = numpy.uint32(offspring_size[1]/2)
    parents_list = parents.tolist()
    for k in range(offspring_size[0]):
        # Loc first parent
        parent_1_index = k%parents.shape[0]
        # Loc second parent
        parent_2_index = (k+1)%parents.shape[0]
        # Offspring generation
        offspring[k] = parents_list[parent_1_index][0:crossover_loc] + parents_list[parent_2_index][crossover_loc:]
    return offspring

def mutation(offspring_crossover, sol_per_pop, num_parents_mating, mutation_percent):
    # Convert to list of lists first to handle mixed types
    offspring_mutation = []
    for _ in range(sol_per_pop - len(offspring_crossover)):
        # Create a copy of a random parent
        parent = random.choice(offspring_crossover)
        new_offspring = parent.copy()
        
        # Apply mutations
        if random.random() < mutation_percent/100:
            # Mutate layers
            new_offspring[0] = random.choice([1,2,3,4])
            
            # Mutate neurons based on new layer count
            if new_offspring[0] == 1:
                new_offspring[1] = random.randint(1,20)
            elif new_offspring[0] == 2:
                new_offspring[1] = (random.randint(1,20), random.randint(1,20))
            elif new_offspring[0] == 3:
                new_offspring[1] = (random.randint(1,20), random.randint(1,20), random.randint(1,20))
            else:
                new_offspring[1] = (random.randint(1,20), random.randint(1,20), random.randint(1,20), random.randint(1,20))
            
            # Mutate other parameters
            new_offspring[2] = random.choice([10, 25, 50, 100, 200])  # batch
            new_offspring[3] = random.choice(['Adam', 'Adagrad', 'RMSprop', 'sgd'])  # optimizer
            new_offspring[4] = random.choice(['uniform','normal'])  # kernel initializer
            new_offspring[5] = random.choice([50, 100, 150, 200])  # epochs
            new_offspring[6] = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])  # dropout
            new_offspring[7] = random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])  # training
            new_offspring[8] = random.choice(['relu', 'tanh', 'sigmoid', 'elu'])  # activation
        
        offspring_mutation.append(new_offspring)
    
    # Combine original offspring with mutated ones
    return offspring_crossover + offspring_mutation
