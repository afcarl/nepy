'''

ne.py NeuroEvolution implemented in Python

@licstart   The following is the entire license notice for
the Python code in this page.

Copyright (C) 2016 jin yeom

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

As additional permission under GNU GPL version 3 section 7, you
may distribute non-source (e.g., minimized or compacted) forms of
that code without the copy of the GNU GPL normally required by
section 4, provided you include this license notice and a URL
through which recipients can access the Corresponding Source.

@licend    The above is the entire license notice
for the Python code in this page.

'''

# module:   ne.py
# author:   Jin Yeom
# since:    01/02/17

import random
import numpy as np
from deap import base
from deap import creator
from deap import tools

'''
    MLP (MultiLayer Perceptron) is a neural network with multiple hidden layers.
    It is initialized given a number of inputs, number of hidden layers, number of
    neurons in a hidden layer, and a number of outputs. Its feedforwarding process
    is done with matrix multiplication. This class is meant to be used for
    neuroevolutions; it is initialized with no connection weights.
'''
class MLP(object):
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.num_outputs = num_outputs
        self.weights = []
        self.bias = -1

    # num_weights returns the number of connection weights in this network.
    def num_weights(self):
        return (self.num_inputs+1) * self.num_hidden_neurons +\
            (self.num_hidden_neurons+1) * self.num_hidden_neurons * (self.num_hidden_layers-1) +\
            (self.num_hidden_neurons+1) * self.num_outputs 

    # flush clears the neural network's connection weights.
    def flush(self):
        if len(self.weights) > 0:
            self.weights = []

    # decode takes a list of connection weights and decodes it into matrices of weights
    # that will be used for computation during feed forwarding. The argument list of 
    # weights must have the valid number of weights.
    def decode(self, weights):
        assert len(weights) == self.num_weights(), 'invalid number of weights: %d != %d' % (len(weights), self.num_weights())
        
        prev_iter = 0
        next_iter = (self.num_inputs+1) * self.num_hidden_neurons

        self.weights.append(np.array(weights[:next_iter]).reshape(self.num_inputs+1, self.num_hidden_neurons))

        for i in range(self.num_hidden_layers-1):
            prev_iter = next_iter
            next_iter = (self.num_hidden_neurons+1) * self.num_hidden_neurons
            self.weights.append(np.array(weights[prev_iter:next_iter]).reshape(self.num_hidden_neurons+1, self.num_hidden_neurons))

        self.weights.append(np.array(weights[next_iter:]).reshape(self.num_hidden_neurons+1, self.num_outputs))

    # feedforward takes a list of numbers as an input vector and feeds it forward via matrix
    # multiplication with weights of the network; returns output vector as a list.
    def feedforward(self, inputs):
        assert len(self.weights) > 0, 'connection weights not initialized' 
        assert len(inputs) == self.num_inputs, 'invalid number of inputs: %d != %d' % (len(inputs), self.num_inputs)

        sigmoid = lambda x: 1. / (1. + np.exp(-x))

        # input layer -> hidden layer
        layer_iter = 0
        input_layer = np.hstack((np.array(inputs), np.array(self.bias)))
        hidden_layer = sigmoid(input_layer.dot(self.weights[layer_iter]))

        # hidden layer (n) -> hidden layer (n + 1)
        while layer_iter < self.num_hidden_layers-1:
            layer_iter += 1
            hidden_layer = np.hstack((hidden_layer, np.array(self.bias)))
            hidden_layer = sigmoid(hidden_layer.dot(self.weights[layer_iter]))
        
        # hidden layer -> output layer
        layer_iter += 1
        hidden_layer = np.hstack((hidden_layer, np.array(self.bias)))
        output_layer = sigmoid(hidden_layer.dot(self.weights[layer_iter]))
        
        return output_layer.tolist() 


'''
    neuroevolution() is an implementation of conventional direct encoding NeuroEvolution
    algorithm. A neural network's genotype is encoded as a list of connection weights. This
    function can be executed by providing a neural network (MLP), an evaluation function which
    takes an MLP as an argument and returns a tuple of scores, number of populations, number of
    generations, rates of crossover and mutation, and a boolean indicator for verbose mode.

    By default, neuroevolution() evaluates an individual based on an inverse relationship
    between a fitness score and an individual's fitness; each connection weight is initialized
    with random float from gaussian distribution with 'mu=0.0' and 'sigma=8.0'. 
'''
def neuroevolution(nn, eval_fn, n_pop, n_gen, r_xover, r_mut, verbose=False):
    assert isinstance(nn, MLP)
    assert callable(eval_fn)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register('attr_float', random.gauss, 0.0, 8.0)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=nn.num_weights()) 
    toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=n_pop)

    population = toolbox.population()

    # eval_nnet is a closure that decodes an individual genotype into a neural network,
    # then evaluates the neural network using an argument evaluation function.
    def eval_nnet(ind):
        nn.flush()
        nn.decode(ind)
        score = eval_fn(nn)
        return score

    toolbox.register('evaluation', eval_nnet)
    toolbox.register('selection', tools.selTournament, tournsize=3)
    toolbox.register('crossover', tools.cxTwoPoint)
    toolbox.register('mutation', tools.mutGaussian, mu=0., sigma=1., indpb=0.1)

    # stats records significant statistical information through evolution, such as
    # the maximum and minimum fitness values, and mean of them in each generation.
    stats = tools.Statistics()
    stats.register('max', np.max)
    stats.register('min', np.min)
    stats.register('mean', np.mean)
    stats.register('std', np.std)
    
    # hof keeps track of best individuals throughout evolution. It keeps the top 10
    # best individuals over generations.
    hof = tools.HallOfFame(10)

    for i in range(n_gen):
        offspring = map(toolbox.clone, toolbox.selection(population, n_pop))
       
        # crossover
        for child0, child1 in zip(offspring[::2], offspring[1::2]):
            if random.random() < r_xover:
                toolbox.crossover(child0, child1)
                del child0.fitness.values
                del child1.fitness.values

        # mutation
        for child in offspring:
            if random.random() < r_mut:
                toolbox.mutation(child)            
                del child.fitness.values

        # evaluate offsprings that have invalid fitness values
        invalid_offspring = [child for child in offspring if not child.fitness.valid]
        fitness_vals = toolbox.map(toolbox.evaluation, invalid_offspring)
        for child, f_val in zip(invalid_offspring, fitness_vals):
            child.fitness.values = f_val

        if verbose:
            print stats.compile([child.fitness.values for child in offspring])

        # update hall of fame and the population 
        hof.update(population)
        population[:] = offspring

    if verbose:
        print '=== Hall of Fame ==='
        for i, best in enumerate(hof):
            score = ', '.join(format(f, '.5f') for f in best.fitness.values)
            print '%d | score=(%s)' % (i, score)

    return hof

# When this module is executed directly, demonstrate XOR test.
if __name__ == '__main__':
    nnet = MLP(2, 1, 2, 1)

    def eval_fn(nnet):
        err = 0.0
        err += (nnet.feedforward([1.0, 1.0])[0] - 0.0) ** 2
        err += (nnet.feedforward([1.0, 0.0])[0] - 1.0) ** 2
        err += (nnet.feedforward([0.0, 1.0])[0] - 1.0) ** 2
        err += (nnet.feedforward([0.0, 0.0])[0] - 0.0) ** 2
        return err/4.,

    neuroevolution(nnet, eval_fn, 50, 50, 0.2, 0.2, verbose=True)
