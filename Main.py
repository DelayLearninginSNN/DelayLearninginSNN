import Population
from Population import *
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_training_phase(img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, par, seed):
    save_delays = False
    interval = 200
    rng = np.random.default_rng(seed)
    static_rng = np.random.default_rng(1)
    name = f"TrainingPhase_img-{img}_layers-{layers}_train_inst-{train_inst}_traindigits-{train_digits}_test_inst-{test_inst}_testdigits-{test_digits}_w-{w}_th-{th}_p-{p}_par-{par}_seed-{seed}"
    test = [y for y in test_digits for x in range(test_inst)]
    train = [static_rng.choice(train_digits) for x in range(train_inst)]
    counter = 0
    sequence = [test_inst for x in range(len(test_digits))] + [train_inst] + [test_inst for x in
                                                                              range(len(test_digits))]
    breaks = []
    for step in sequence:
        breaks.append(counter + step)
        counter += step
    input = Util.create_mnist_sequence_input(train, test, interval, breaks, img)

    pop = Population((img ** 2 * layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1, 40)), n_layers=layers, w=w, p=p, partial=par, trainable=False,
                                        seed=seed)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img ** 2) if rng.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[rng.integers(1, 40) for x in range(len(ij))], trainable=False)

    training_change = [test_inst * len(test_digits), test_inst * len(test_digits) + train_inst]
    durations = []

    for i, l in enumerate(list(map(list, zip(*input)))):
        if i in training_change:
            durations.append(min(l) - 1)
    pop.run(durations[0], path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False,
            save_neuron_data=True)
    for syn in pop.synapses:
        syn.trainable = True
    pop.run(durations[1] - durations[0], path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False,
            save_neuron_data=True)
    for syn in pop.synapses:
        syn.trainable = False
    pop.run(max([max(x) for x in input]) + interval - durations[1], path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False,
            save_neuron_data=True)


'''
    Example code of running training/testing.
    Param:
    img (image size): 10 by 10 pixels
    layers (number of model layers): 2
    train_inst (number of training instances of each digit): 10
    train_digits (set of training digits): [0, 1]
    test_inst (number of training instances of each digit): 10
    test_digits (set of test digits): [0, 1]
    w (connections weights): 4
    th (threshold for PG matching): 0.7
    p (connectivity): 0.1
    partial (using part of the training method f(x) or f(x) and g(x)): True
    seed: 1 
    
'''
run_training_phase(10, 2, 10, [0, 1], 10, [0, 1], 4, 0.7, 0.1, True, 1)
