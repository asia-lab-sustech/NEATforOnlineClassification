from __future__ import print_function
import os
import neat
from numpy import median, std
from random import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler#Scaling
import visualize
from collections import Counter
from fitness import *
import sys

def Feature_Scaling(data,label):
    X=data.drop(labels=label, axis=1, inplace=False)
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X[label]=data[label]
    return X

def code_answer(attributer):
    ans = np.zeros((attributer.shape[0],len(Counter(attributer))), dtype="int")
    attributer=list(attributer)
    first=0
    for i in range(len(attributer)):
        if attributer[i] == first: 
            ans[i][0] = 1
        else:
            ans[i][1] = 1
    return ans

fitnessways=['accuracy','badaccgoodacc','profit','profit_history']
eval_fitness=[eval_genomes1,eval_genomes2,eval_genomes3,eval_genomes4]
label='loan_status'
global inputs,outputs
inputs = []
outputs = []
data=pd.read_csv('~//data//Anainfo20172018.csv')

fitness_index=int(sys.argv[1])
intervals=int(sys.argv[2])
arg=[]
arg.append(float(sys.argv[3]))
arg.append(float(sys.argv[4]))

def run(config_file):
    # Load configuration.
    print(fitnessways[fitness_index]+'_interval'+str(intervals)+'_args'+str(arg)+'_data20172018')
    data['issue_d'] = pd.to_datetime(data['issue_d'],errors = 'coerce')
    train=data
    train.drop('issue_d', axis=1, inplace=True)

    global inputs,outputs
    outputs=code_answer(train[label])
    X=train.drop(labels=label, axis=1, inplace=False)
    inputs=X.values.tolist()
    deal_data(inputs,outputs)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_fitness[fitness_index],eval_test_all,intervals,arg,len(outputs),500000)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
