import numpy as np
import random

# Define the cities and their coordinates
victims = {}

# Define parameters
population_size = 50
generations = 1000
mutation_rate = 0.01

# Function to calculate the total distance of a route
def calculate_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        victim1, victim2 = route[i], route[i + 1]
        total_distance += np.linalg.norm(np.array(victims[victim1]) - np.array(victims[victim2]))
    return total_distance

# Function to generate an initial population of routes
def generate_population(size):
    population = []
    victims_list = list(victims.keys())
    for _ in range(size):
        route = random.sample(victims_list, len(victims_list))
        population.append(route)
    return population

# Function to select parents using tournament selection
def tournament_selection(population, k=5):
    tournament = random.sample(population, k)
    return min(tournament, key=calculate_distance)

# Function to perform crossover (order crossover)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    remaining = [victim for victim in parent2 if victim not in child]
    child[:start] = remaining[:start]
    child[end:] = remaining[start:]
    return child

# Function to perform mutation (swap mutation)
def mutate(route):
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Main genetic algorithm function
def genetic_algorithm():
    population = generate_population(population_size)

    for generation in range(generations):
        population = sorted(population, key=calculate_distance)[:population_size]

        new_population = []

        for _ in range(population_size // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    best_route = min(population, key=calculate_distance)
    best_distance = calculate_distance(best_route)

    print("Best Route:", best_route)
    print("Best Distance:", best_distance)