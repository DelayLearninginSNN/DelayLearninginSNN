import gzip
import json
import shutil

import Data
import Population
from Population import *
import numpy as np
import multiprocessing as mp
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img = [10]
layers = [2]
train_inst = [20]
train_digits = [[0,1]]
test_inst = [25]
test_digits = [[0,1]]
w = [6]
th = [0.7]
p = [0.1]
par = [True]
seed = [320, 335]
seed.reverse()

params = [img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, par, seed]
combos = list(itertools.product(*params))
#combos = combos[:250]


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
    input = Data.create_mnist_sequence_input(train, test, interval, breaks, img)

    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=False, seed=seed)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if rng.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[rng.integers(1,40) for x in range(len(ij))], trainable=False)


    training_change = [test_inst*len(test_digits), test_inst*len(test_digits) + train_inst]
    durations = []

    for i, l in enumerate(list(map(list, zip(*input)))):
        if i in training_change:
            durations.append(min(l)-1)

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
    pop.run(max([max(x) for x in input]) + interval - durations[1], path='network_plots/', name=name, record_PG=True, save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False, save_neuron_data=True)



def run_0_8_0(img, layers, num, inst, w, th, p, par, train, seed):
    name = f"MNIST_img-{img}_layers-{layers}_num-{num}_inst-{inst}_w-{w}_th-{th}_p-{p}_par-{par}_train-{train}"
    input = Data.create_mnist_input(inst, num, 200, image_size=img)
    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=train)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if np.random.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[np.random.randint(1,40) for x in range(len(ij))], trainable=train)
    pop.run(max([max(inp) for inp in input]) + 100, path='network_plots/', name=name, record_PG=True, save_post_model=False, PG_duration=100, PG_match_th=th, save_delays=False, save_synapse_data=False, save_neuron_data=True)


#if __name__ == '__main__':
#    with mp.Pool(30) as p:
#        p.starmap(run_training_phase, combos)





#for folder in ["G:/USABLE RESULTS/unseen digit 0.7/", "G:/USABLE RESULTS/unseen digit 0.8/", "G:/USABLE RESULTS/unseen digit 0.9/"]:
#    if os.path.isdir(os.path.join(folder)):
#        Data.simplify_PG_data(folder)
#        Data.simplify_neuron_data(folder)

'''
d = "G:/USABLE RESULTS/training phase 0.7"
for dir in os.listdir(d):
    p = os.path.join(d,dir, "neuron_data.json")
    if os.path.exists(p):
        f = open(p)
        data = json.load(f)
        f.close()
        if data["0"]["duration"] < 24800:
            print(p)
            print(data["0"]["duration"])
            shutil.rmtree(os.path.join(d,dir))
    else:
        shutil.rmtree(os.path.join(d, dir))
'''





#ddir = "G:/USABLE RESULTS/training phase 0.9/"
#param = []
#for dir in os.listdir(ddir):
#    param.append((ddir, dir, 0.7, 0.9))



#if __name__ == '__main__':
#   with mp.Pool(30) as p:
#        p.starmap(Data.change_threshold, param)





Data.compile_results("G:/USABLE RESULTS/training phase 0.8", "training phase_0.8", 2, 25, 20)
Data.compile_results("G:/USABLE RESULTS/training phase 0.9", "training phase_0.9", 2, 25, 20)

Data.compile_results("G:/USABLE RESULTS/unseen digit 0.7", "unseen digit 0.7", 3, 25, 20)
Data.compile_results("G:/USABLE RESULTS/unseen digit 0.8", "unseen digit 0.8", 3, 25, 20)
Data.compile_results("G:/USABLE RESULTS/unseen digit 0.9", "unseen digit 0.9", 3, 25, 20)