import pandas as pd
import numpy as np
from random import *

from collections import OrderedDict

from pylab import imshow, show, get_cmap
from numpy import random

import matplotlib.pyplot as plt

from tqdm import tqdm

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
		round((number_pixels)/2), replace = True)

	baby_agent = {'pixels' : agent1.get('pixels')}
	for pixel in random_half_pixels:
		baby_agent['pixels'][pixel]= agent2['pixels'][pixel]

	random_pixels = np.random.choice(range(0, number_pixels), 
		round(prob_mutation*number_pixels), replace = True)

	#for pixel in random_pixels:
	#	baby_agent.get('pixels')[pixel] = random.random(3)

	baby_agent['pixels'][random_pixels] = random.random(
		(round(prob_mutation*number_pixels), 3)
	)

	baby_agent = {'pixels': baby_agent['pixels'], 'fitness': calculate_fitness(
		baby_agent.get('pixels'), number_pixels
	)}

	return baby_agent

def make_next_gen(skim_percentage, agents, pop_size, number_pixels):

	number_who_survive = round(skim_percentage*len(agents))

	# Always keep at least 2 for breeding
	if number_who_survive < 2:
		number_who_survive = 2

	agents_ordered = sorted(agents, key=lambda d: d['fitness'], reverse=True)

	# Replace the ones with the lowest fitness with random breedings of the highest fitness

	fittest_agents = agents_ordered[:number_who_survive]

	new_agents = []
	#agents = fittest_agents

	for rep_agent in range(0,(pop_size - number_who_survive)):

		agent1 = random.choice(fittest_agents)
		agent2 = random.choice(fittest_agents)
	
		#print(baby_agent)
		#new_agents.append(baby_agent)
		#new_agents['pixels'].append(baby_agent.get('pixels'))
		#new_agents['fitness'].append(baby_agent.get('fitness'))
		#new_agents_pixels.append(baby_agent.copy())
		new_agents.append(breed(agent1, agent2, prob_mutation, number_pixels))
		
		#agents.append({'pixels':baby_agent.get('pixels'), 'fitness':baby_agent.get('fitness')})


	#for new_agents in new_agents:
	# 	new_agents.update({'fitness' : calculate_fitness(new_agents.get('pixels'), number_pixels)})


	#print(new_agents)
	agents = fittest_agents + new_agents
	return agents



def get_population_mean_fitness(agents):
	fitnesses = []
	for agent in agents:
		fitnesses.append(agent.get('fitness'))
	return np.mean(fitnesses)


if __name__ == '__main__':

	save_details = []

	for rand_params in tqdm(range(0,100)):

		nx = 10
		ny = 10
		number_pixels = nx*ny
		pop_size = 100
		assert pop_size >= 3, "Pick a population size >=3"
		skim_percentage = uniform(0, 1) # percentage of agents who survive each year
		#percentage_pixels_mutate = 0.1 # What percentage of all pixels get mutated if at all
		prob_mutation = uniform(0, 1)
		num_iterations = 1000

		# Initialisation:
		agents = []
		for agent in range(0, pop_size):
			agents.append(initialise_agent(number_pixels))

		for agent in agents:
		 	agent.update({'fitness' : calculate_fitness(agent.get('pixels'), number_pixels)})

		#print("population before:")
		mean_pixel_begin = get_population_mean_fitness(agents)
		#print(mean_pixel_begin)

		# Iterations: 
		for iteration in range(0,num_iterations):

			agents = make_next_gen(skim_percentage, agents, pop_size, number_pixels)
			#if iteration % 1000 == 0:
				#print(iteration)

		#print("population after:")
		mean_pixel = get_population_mean_fitness(agents)
		#print(mean_pixel)
		#print(skim_percentage)
		#print(prob_mutation)
		#print(num_iterations)

		save_details.append([number_pixels,pop_size,skim_percentage,prob_mutation,num_iterations,mean_pixel_begin,mean_pixel])

	save_details = pd.DataFrame(save_details, columns = ['Number of pixels','Population size','skim_percentage','prob_mutation','Number of iterations','Beginning mean pixel','End mean pixel'])
	#print(save_details)
	save_details.to_csv('Looking for parameters3.csv')

	x = save_details['skim_percentage']
	y = save_details['prob_mutation']
	maxs = np.max(save_details['End mean pixel'])
	mins = np.min(save_details['End mean pixel'])
	colors = [[(s-mins)/(maxs-mins),0,0] for s in save_details['End mean pixel']]
	
	plt.scatter(x, y, c=colors)
	plt.xlabel("skim_percentage")
	plt.ylabel("prob_mutation")
	plt.title("highest is "+str(round(maxs,3))+" (red) lowest is "+str(round(mins,3))+" (black)")
	plt.show()

	# blue is end result 0 , magenta is 1
	#plt.scatter(1,1, c=1, alpha=0.5) 
	#plt.show()



