import pandas as pd
import numpy as np
from random import *

from collections import OrderedDict

from pylab import imshow, show, get_cmap
from numpy import random

import matplotlib.pyplot as plt

from tqdm import tqdm

random.seed(1)

def calculate_fitness(agent_pixels, number_pixels):

	# Higher fitness = lighter colour (closer to [1,1,1])
	fitness = agent_pixels.mean()

	# fewer colours the better (max is 11 with 1 dp):

	#print(len(set([round(a[0],1) for a in agent_pixels])))
	#print(agent_pixels)

	#fitness = (11 - len(set([round(a[0],1) for a in agent_pixels])))/11
	#print(pd.value_counts([round(a[0],1) for a in agent_pixels])

	return fitness

def plot_agent(agent_pixels, agent_fitness, number_pixels):
	imshow(agent_pixels.reshape(nx, ny, 3), interpolation='nearest')
	thisfitness = calculate_fitness(agent_pixels, number_pixels)
	plt.title("Fitness = " + str(round(thisfitness,3)) + ", given fitness = " +  str(round(agent_fitness,3)))
	show()


def initialise_agent(number_pixels):
	init_agent = {'pixels': random.random((number_pixels, 3))}
	return init_agent

def breed(agent1, agent2, prob_mutation, number_pixels):
	'''
	Make a baby between agent1 and agent2
	Update agent1 with the random_half_pixels from agent2
	Calculate it's fitness
	'''

	random_half_pixels = np.random.choice(range(0, number_pixels), 
		round((number_pixels)/2), replace = True)

	baby_agent = {'pixels' : (agent1['pixels']).copy()}
	for pixel in random_half_pixels:
		baby_agent['pixels'][pixel]= (agent2['pixels'][pixel]).copy()

	random_pixels = np.random.choice(range(0, number_pixels), 
		round(prob_mutation*number_pixels), replace = True)

	baby_agent['pixels'][random_pixels] = random.random(
		(round(prob_mutation*number_pixels), 3)
	)

	baby_agent['fitness'] = calculate_fitness(
		baby_agent['pixels'], number_pixels
	)
	
	return baby_agent

def make_next_gen(skim_percentage, agents, pop_size, number_pixels):

	number_who_survive = round(skim_percentage*len(agents))

	# Always keep at least 2 for breeding
	if number_who_survive < 2:
		number_who_survive = 2

	agents_ordered = sorted(agents, key=lambda d: d['fitness'], reverse=True)

	# Replace the ones with the lowest fitness with random breedings of the highest fitness

	for index, elem in enumerate(agents_ordered):
		if index >number_who_survive:
			
			agent1 = random.choice(list(agents_ordered[:number_who_survive]))
			agent2 = random.choice(list(agents_ordered[:number_who_survive]))

			baby = breed(agent1, agent2, prob_mutation, number_pixels)

			agents_ordered[index]['pixels'] = baby['pixels']
			agents_ordered[index]['fitness'] = baby['fitness']

	return agents_ordered

def get_population_mean_fitness(agents):
	fitnesses = []
	for agent in agents:
		fitnesses.append(agent['fitness'])
	return np.mean(fitnesses)

def plot_fittest(agents, nx, ny):
	agents_ordered = sorted(agents, key=lambda d: d['fitness'], reverse=True)
	plot_agent(agents_ordered[len(agents)-1]['pixels'], agents_ordered[len(agents)-1]['fitness'], nx*ny)
	
if __name__ == '__main__':

	save_details = []

	#for rand_params in tqdm(range(0,500)):

	nx = 10
	ny = 10
	number_pixels = nx*ny
	pop_size = 30
	assert pop_size >= 3, "Pick a population size >=3"
	skim_percentage = 0.25#uniform(0, 1) # percentage of agents who survive each year
	prob_mutation = 0.01#uniform(0, 0.2)

	#skim_percentage = uniform(0.2, 0.6) # percentage of agents who survive each year
	#prob_mutation = uniform(0.001, 0.05)

	num_iterations = 1

	agents = []
	for agent in range(0, pop_size):
		agents.append(initialise_agent(number_pixels))

	for agent in agents:
	 	agent.update({'fitness' : calculate_fitness(agent['pixels'], number_pixels)})

	mean_pixel_begin = get_population_mean_fitness(agents)
	plot_fittest(agents, nx, ny)

	save_mean = []
	for iteration in tqdm(range(0,num_iterations)):
		agents = make_next_gen(skim_percentage, agents, pop_size, number_pixels)
		mean_pixel = get_population_mean_fitness(agents)
		save_mean.append([iteration, mean_pixel])

	#mean_pixel_end = get_population_mean_fitness(agents)
	#save_details.append([number_pixels,pop_size,skim_percentage,prob_mutation,num_iterations,mean_pixel_begin,mean_pixel_end])

	save_mean = pd.DataFrame(save_mean, columns = ['Number of iterations','End mean pixel'])
	save_mean.to_csv('Save means each iteration_new.csv')

	plt.plot(save_mean['Number of iterations'], save_mean['End mean pixel'])
	plt.xlabel("Iteration")
	plt.ylabel("Mean population fitness")
	plt.show()

	plot_fittest(agents, nx, ny)

	#print(mean_pixel)
	#print(skim_percentage)
	#print(prob_mutation)
	#print(num_iterations)

	# save_details = pd.DataFrame(save_details, columns = ['Number of pixels','Population size','skim_percentage','prob_mutation','Number of iterations','Beginning mean pixel','End mean pixel'])
	# save_details.to_csv('Looking for parameters3.csv')

	# x = save_details['skim_percentage']
	# y = save_details['prob_mutation']
	# maxs = np.max(save_details['End mean pixel'])
	# mins = np.min(save_details['End mean pixel'])
	# colors = [[(s-mins)/(maxs-mins),0,0] for s in save_details['End mean pixel']]
	
	# plt.scatter(x, y, c=colors)
	# plt.xlabel("skim_percentage")
	# plt.ylabel("prob_mutation")
	# plt.title("highest is "+str(round(maxs,3))+" (red) lowest is "+str(round(mins,3))+" (black)")
	# plt.show()

	# blue is end result 0 , magenta is 1
	#plt.scatter(1,1, c=1, alpha=0.5) 
	#plt.show()
	#a = np.array([[0, 1], [2, 3]], order='F')
	#a.resize((2, 1))

