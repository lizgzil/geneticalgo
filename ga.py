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


def initialise_agent(nx, ny):
	''' 
	nx: number of pixels in the x axis
	ny: number of pixels in the y axis
	'''
	init_agent = {'pixels': random.random((nx, ny, 3))}
	return init_agent


	
def calculate_fitness(agent):
	# Will fill in properly later, for now just random
	fitness = random.random()
	return fitness

def breed(agent1, agent2, nx, ny):
	'''
	Make a baby between agent1 and agent2
	Update agent1 with the random_half_pixels from agent2
	Calculate it's fitness
	'''

	random_half_pixels = np.random.choice(range(0, nx*ny), round((nx*ny)/2), replace = False)
		
	baby_agent = agent1
	for pixel in random_half_pixels:
		row_number = pixel//ny
		column_number = pixel%ny
		baby_agent.get('pixels')[row_number][column_number] = agent2.get('pixels')[row_number][column_number]

	baby_agent.update({'fitness' : calculate_fitness(baby_agent)})
	return baby_agent

def make_next_gen(skim_percentage, agents, pop_size, nx, ny):

	print(len(agents))
	number_who_survive = round(skim_percentage*len(agents))

	number_who_died = pop_size - number_who_survive

	agents_order_by_fitness = sorted(agents, key=lambda d: d['fitness'], reverse=True)

	agents = agents_order_by_fitness[:number_who_survive]
	print(len(agents))
	# Replace the number who died with random breedings of the survivors

	for sex in range(0, number_who_died):
		agent1 = random.choice(agents)
		agent2 = random.choice(agents)
		baby_agent = breed(agent1, agent2, nx, ny)
		agents.append(baby_agent)
		print(len(agents))

	return agents

def mutate(agent_pixels, percentage_pixels_mutate, nx, ny):
	'''
	Randomly mutate a certain percentage of the agent's pixels randomly
	'''
	random_mutation_pixels = np.random.choice(range(0, nx*ny), round((nx*ny)*percentage_pixels_mutate), replace = False)
	#print(random_mutation_pixels)
	new_pixels = agent_pixels
	for pixel in random_mutation_pixels:
		row_number = pixel//ny
		column_number = pixel%ny
		#print(agent_pixels[row_number][column_number])
		replace_with = random.random(3)
		#print(replace_with)
		new_pixels[row_number][column_number] = replace_with

	new_fitness = calculate_fitness(agent_pixels)
	return new_pixels, new_fitness

if __name__ == '__main__':

	nx = 2
	ny = 2
	pop_size = 3
	skim_percentage = 0.5
	percentage_pixels_mutate = 0.3 # What percentage of all pixels get mutated if at all

	agents = []
	for agent in range(0, pop_size):
		agents.append(initialise_agent(nx, ny))
	

	#for agent in agents:
	#	plot_agent(agent)

	for agent in agents:
	 	agent.update({'fitness' : calculate_fitness(agent)})

	agents = make_next_gen(skim_percentage, agents, pop_size, nx, ny)

	new_agents =[]
	for i, agent in enumerate(agents):
		#print("before mutation")
		#print(agent)
		agent_pixels = agent.get('pixels')
		new_pixels, new_fitness = mutate(agent_pixels, percentage_pixels_mutate, nx, ny)
		#agent.update({'pixels' : new_pixels, 'fitness' : new_fitness})
		new_agents.append({'pixels' : new_pixels, 'fitness' : new_fitness})
		print("after mutation")
		print(new_pixels)
	#print("after mutation")
	#print(agents)
	print("new after mutation")
	print(new_agents)

	# print("dfjdsfl")
	# init_agent = initialise_agent(nx, ny)

	# print(init_agent)
	# init_agent.get('pixels')[1][1] = random.random(3)
	# print("dfjdsfl")
	# print(init_agent)


