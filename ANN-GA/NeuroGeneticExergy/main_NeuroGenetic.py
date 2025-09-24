import tensorflow as tf
import os
import numpy
import genetic
import ann
import csv
import numpy as np
import random
import data_input
import gc

#Parallel units definition - Kaggle optimized
# Kaggle has 4 CPUs, but we'll use 2 to be safe
NUM_PARALLEL_EXEC_UNITS = 2

# TensorFlow 2.x configuration
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
tf.config.threading.set_inter_op_parallelism_threads(1)

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

# Kaggle environment detection
print("🔍 Environment Detection:")
print(f"  Running on: {'Kaggle' if os.path.exists('/kaggle/input') else 'Local'}")
print(f"  CPUs configured: {NUM_PARALLEL_EXEC_UNITS}")
print(f"  TensorFlow version: {tf.__version__}")
print("="*50)



#Genetic Algorithm Parameters
# Reduced for Kaggle to avoid timeout (you can increase these)
num_generations = 5  # Reduced from 50 for faster execution
sol_per_pop = 20      # Reduced from 40 for faster execution  
num_parents_mating = 4 # Reduced accordingly
mutation_percent = 50

#ANN hyperparameter definition search space
layers_list = [1,2,3,4]
batch_list = [10, 25, 50, 100, 200]
optimisers = ['Adam', 'Adagrad', 'RMSprop', 'sgd']
kernel_initializer = ['uniform','normal']
epochs = [50, 100, 150, 200]
dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
training = [0.05, 0.10,0.15,0.20,0.25,0.30]
activation = ['relu', 'tanh', 'sigmoid', 'elu']

#Creating an empty list to store the initial population
initial_population = []
#Creating an empty list to store the final solutions
final_list=[]

#Create initial population
for curr_sol in numpy.arange(0, sol_per_pop):
    layers= random.choice(layers_list)
    batch = random.choice(batch_list)
    opt = random.choice(optimisers)
    ker = random.choice(kernel_initializer)
    epo = random.choice(epochs)
    drop = random.choice(dropout)
    train = random.choice(training)
    act = random.choice(activation)
     
    if layers == 1:  
        n1 = random.randint(1,20)
        neurons = n1
    elif layers == 2: 
        n1 = random.randint(1,20)
        n2 = random.randint(1,20)
        neurons = n1,n2
    elif layers == 3: 
        n1 = random.randint(1,20)
        n2 = random.randint(1,20)
        n3 = random.randint(1,20)
        neurons = n1,n2,n3
    elif layers == 4: 
        n1 = random.randint(1,20)
        n2 = random.randint(1,20)    
        n3 = random.randint(1,20)
        n4 = random.randint(1,20)
        neurons = n1,n2,n3,n4
    
    initial_population.append([layers, neurons, batch, opt, ker, epo, drop, train, act])

#Initial population - use object dtype to handle mixed types
pop_inputs = np.array(initial_population, dtype=object)
del(initial_population)

#Start GA process
for generation in range(num_generations):    
    pre_list=[]
    list_inputs =[]
    list_fitness=[]
    list_objective=[]
    list_other_metrics = []
    
    print("================================================================")
    print("================================================================")
    print("\nGeneration : ", generation+1)
    print("Inputs : \n",  pop_inputs)

    pop_inputs = pop_inputs      
                             
    # Measuring the fitness of each solution in the population.
    fitness = []
    objective = []
    other_metrics =[]
    
    #ANN model training for sol_population p in generation g
    for index in range(sol_per_pop):
        
        print('\n Generation: ', generation+1, " of ", num_generations, ' Simulation: ', index+1 ,' of ', sol_per_pop)
        X_train, X_test, Y_train, Y_test = data_input.data(pop_inputs[index][7])
        print('\n Test/Training :', ((1-pop_inputs[index][7])*100),'/',pop_inputs[index][7]*100)
        
        #Export ANN metric performance
        RMSE, RMSE_val, mae, val_mae, R2, R2_v = ann.model_ANN(X_train.shape[1], Y_train.shape[1], pop_inputs[index][0], pop_inputs[index][1],\
                                   pop_inputs[index][2], pop_inputs[index][3], pop_inputs[index][4],
                                   pop_inputs[index][5], pop_inputs[index][6], pop_inputs[index][8],
                                   X_train, Y_train, X_test, Y_test)
        
        #OBJECTIVE FUNCTION
        obj = ((1-RMSE)*.5 + (1-RMSE_val)*.5)
        
        #Appending obj 1 and 2
        fitness.append([RMSE, RMSE_val])
        
        #Appending objective list
        objective.append([obj])
        
        print("Fitness")
        print(RMSE, RMSE_val)
        print("Objective")
        print(obj)
        other_metrics.append([mae, val_mae, R2, R2_v])
        del  X_train, Y_train, X_test, Y_test
        gc.collect()
    
    print(fitness)
    print(objective)
    
    list_fitness.append(fitness)
    list_objective.append(objective)
    list_inputs.append(pop_inputs.tolist())
    list_other_metrics.append(other_metrics)
    
    
    # top performance ANN model in the population are selected for mating.
    parents = genetic.mating_pool(pop_inputs, 
                                    objective.copy(), 
                                    num_parents_mating)
    print("Parents")
    print(parents)
    parents = numpy.asarray(parents) 


    # Crossover to generate the next geenration of solutions
    offspring_crossover = genetic.crossover(parents,
                                       offspring_size=(int(num_parents_mating/2), pop_inputs.shape[1]))
    print("Crossover")
    print(offspring_crossover)


    # Mutation for population variation
    offspring_mutation = genetic.mutation(offspring_crossover, sol_per_pop, num_parents_mating, 
                                     mutation_percent=mutation_percent)
    
    print("Mutation")
    print(offspring_mutation)
        
    # New population for generation g+1
    pop_inputs[0:len(offspring_crossover), :] = offspring_crossover
    pop_inputs[len(offspring_crossover):, :] = offspring_mutation
    print('NEW INPUTS :\n', pop_inputs )
    
    
    for x in range(len(list_inputs)):
        for y in range(len(list_inputs[0])):
            pre_list = list_inputs[x][y]
            for m in range(len(list_fitness[x][y])):
                pre_list.append(list_fitness[x][y][m])
            pre_list.append(list_objective[x][y][0])
            for w in range(len(list_other_metrics[x][y])):
                pre_list.append(list_other_metrics[x][y][w])
    
            
            final_list.append(pre_list)      
   
    del(fitness, objective, other_metrics, parents, offspring_mutation, offspring_crossover, list_inputs, list_fitness, list_objective, list_other_metrics, pre_list)
    gc.collect()
    
#Insert headers to final list
final_list.insert(0, ['Layers', 'Neurons', 'batch', 'optimiser', 'keras', 'epochs', 'dropout', 'train %', 'activation', 'RMSE', 'VAL_RMSE', 'Objective', 'mae', 'val_mae', 'R2', 'R2_v' ])
final_list
 
# Kaggle-compatible file saving
import os
kaggle_output_dir = '/kaggle/working/' if os.path.exists('/kaggle/working/') else './'

#Saving all ANN structures, hyperparameters and metrics
results_file = os.path.join(kaggle_output_dir, 'FINAL_RESULTS.csv')
with open(results_file, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerows(final_list)

print(f"\nResults saved to: {results_file}")
print(f"Total solutions evaluated: {len(final_list)-1}")  # -1 for header

# Display best solution
if len(final_list) > 1:
    # Sort by objective (column 11, 0-indexed)
    best_solutions = sorted(final_list[1:], key=lambda x: float(x[11]), reverse=True)
    print("\n" + "="*60)
    print("🏆 TOP 3 BEST SOLUTIONS:")
    print("="*60)
    headers = final_list[0]
    for i, sol in enumerate(best_solutions[:3]):
        print(f"\nRank #{i+1}:")
        for j, (header, value) in enumerate(zip(headers, sol)):
            print(f"  {header}: {value}")
        print(f"  Objective Score: {sol[11]}")
    print("="*60)

