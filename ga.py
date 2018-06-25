import pandas as pd
import numpy as np
import random

from collections import OrderedDict

from pylab import imshow, show, get_cmap
from numpy import random

random.seed(1)

def plot_agent(agent):
# 	# 4x3: rgb 
# 	Z = [[[0.1,0.9,0.9],[0.9,0.1,0.9]],
# 	[[0.1,0.9,0.8],[0.2,0.2,0.9]],
# 	[[0.6,0.5,0.5],[0.4,0.4,0.4]],
# 	[[0.1,0.1,0.1],[0.9,0.9,0.9]]]
 	imshow(agent['pixels'], interpolation='nearest')
 	show()


def initialise_agent(number_pixels):
	init_agent = {'pixels': random.random((number_pixels, 3))}
	return init_agent

	
def calculate_fitness(agent_pixels, number_pixels):
	# Will fill in properly later, for now just random
	#fitness = random.random()
	# good = closer to [1,1,1] = low difference from [1,1,1]
	# fitness = 1 - difference
	#fitness = np.absolute(np.full((number_pixels, 3), 1) - agent_pixels).mean()

	fitness = agent_pixels.mean()
	return fitness

# def mutate(agent_pixels, percentage_pixels_mutate, nx, ny):
# 	'''
# 	Randomly mutate a certain percentage of the agent's pixels randomly
# 	'''
# 	random_mutation_pixels = np.random.choice(range(0, nx*ny), round((nx*ny)*percentage_pixels_mutate), replace = False)
# 	new_pixels = agent_pixels
# 	for pixel in random_mutation_pixels:
# 		row_number = pixel//ny
# 		column_number = pixel%ny
# 		replace_with = random.random(3)
# 		new_pixels[row_number][column_number] = replace_with

# 	new_fitness = calculate_fitness(new_pixels, nx, ny)
# 	return new_pixels, new_fitness

def breed(agent1, agent2, prob_mutation, number_pixels):
	'''
	Make a baby between agent1 and agent2
	Update agent1 with the random_half_pixels from agent2
	Calculate it's fitness
	'''

	random_half_pixels = np.random.choice(range(0, number_pixels), 
		round((number_pixels)/2), replace = False)

	baby_agent = {'pixels' : agent1.get('pixels')}
	for pixel in random_half_pixels:
		baby_agent['pixels'][pixel]= agent2['pixels'][pixel]

	print('1')
	print(baby_agent)

	random_pixels = np.random.choice(range(0, number_pixels), 
		round(prob_mutation*number_pixels), replace = False)

	#for pixel in random_pixels:
	#	baby_agent.get('pixels')[pixel] = random.random(3)

	baby_agent['pixels'][random_pixels] = random.random(
		(round(prob_mutation*number_pixels), 3)
	)
	print('2')
	print(baby_agent)

	baby_agent = {'pixels': baby_agent['pixels'], 'fitness': calculate_fitness(
		baby_agent.get('pixels'), number_pixels
	)}
	print('3')
	print(baby_agent)

	return baby_agent

def make_next_gen(skim_percentage, agents, pop_size, number_pixels):

	number_who_survive = round(skim_percentage*len(agents))

	# Always keep at least 2 for breeding
	if number_who_survive < 2:
		number_who_survive = 2

	agents_ordered = sorted(agents, key=lambda d: d['fitness'], reverse=True)

	# Replace the ones with the lowest fitness with random breedings of the highest fitness

	fittest_agents = agents_ordered[:number_who_survive]

	new_agents_pixels = []
	#agents = fittest_agents
	c = 0
	for rep_agent in range(0,(pop_size - number_who_survive)):

		print('3.5')
		print(new_agents_pixels)
		agent1 = random.choice(fittest_agents)
		agent2 = random.choice(fittest_agents)
		print('3.6')
		print(new_agents_pixels)
		#baby_agent = breed(agent1, agent2, prob_mutation, number_pixels)
		print('3.7')
		print(new_agents_pixels)
		print('4')
		#print(baby_agent)
		#new_agents.append(baby_agent)
		#new_agents['pixels'].append(baby_agent.get('pixels'))
		#new_agents['fitness'].append(baby_agent.get('fitness'))
		#new_agents_pixels.append(baby_agent.copy())
		new_agents_pixels.append(breed(agent1, agent2, prob_mutation, number_pixels))
		
		#agents.append({'pixels':baby_agent.get('pixels'), 'fitness':baby_agent.get('fitness')})
		print('5')
		#print(agents)
		print(new_agents_pixels)
		c = c+1

	#for new_agents in new_agents_pixels:
	# 	new_agents.update({'fitness' : calculate_fitness(new_agents.get('pixels'), number_pixels)})


	#print(new_agents)
	#agents = fittest_agents + new_agents
	print('6')
	print(agents)
	return agents



def get_population_mean_fitness(agents):
	fitnesses = []
	for agent in agents:
		fitnesses.append(agent.get('fitness'))
	return np.mean(fitnesses)


if __name__ == '__main__':

	nx = 2
	ny = 2
	number_pixels = nx*ny
	pop_size = 4
	assert pop_size >= 3, "Pick a population size >=3"
	skim_percentage = 0.3 # percentage of agents who survive each year
	#percentage_pixels_mutate = 0.1 # What percentage of all pixels get mutated if at all
	prob_mutation = 0.8
	num_iterations = 3

	# Initialisation:
	agents = []
	for agent in range(0, pop_size):
		agents.append(initialise_agent(number_pixels))

	for agent in agents:
	 	agent.update({'fitness' : calculate_fitness(agent.get('pixels'), number_pixels)})

	print("population before:")
	mean_pixel = get_population_mean_fitness(agents)
	print(mean_pixel)

	print("before")
	print(agents)

	# Iterations: 
	for iteration in range(0,num_iterations):

		agents = make_next_gen(skim_percentage, agents, pop_size, number_pixels)

		print('7')
		print(agents)

		if iteration % 1000 == 0:
			print(iteration)

	print("population after:")
	mean_pixel = get_population_mean_fitness(agents)
	print(mean_pixel)
	print(skim_percentage)
	print(prob_mutation)
	print(num_iterations)

	print("after")
	print(agents)
	#for ag in agents:
	 #	print(ag.get('pixels'))

	 # The pixels are changing, the fitness isn't




