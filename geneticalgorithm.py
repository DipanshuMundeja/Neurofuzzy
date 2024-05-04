import random

def fitness_function(schedule):
    return -sum(schedule) 


def generate_population(pop_size, task_count):
    return [[random.randint(0, 1) for _ in range(task_count)] for _ in range(pop_size)]

def genetic_algorithm(pop_size, generations, task_count):
    population = generate_population(pop_size, task_count)
    for gen in range(generations):
        fitness_scores = [fitness_function(individual) for individual in population]
        selected_parents = [population[i] for i in np.argsort(fitness_scores)[-pop_size // 2:]]
        children = []
        while len(children) < pop_size:
            parent1, parent2 = random.choice(selected_parents), random.choice(selected_parents)
            crossover_point = random.randint(1, task_count - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            children.extend([child1, child2])
        population = children
        best_individual = max(population, key=fitness_function)
        print(f"Generation {gen+1}: Best Fitness = {fitness_function(best_individual)}")
    return max(population, key=fitness_function)


best_solution = genetic_algorithm(pop_size=50, generations=100, task_count=10)
print("Best Solution:", best_solution)
print("Best Fitness:", fitness_function(best_solution))
