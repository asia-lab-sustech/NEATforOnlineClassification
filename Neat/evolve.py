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

label='loan_status'
global inputs,outputs
inputs = []
outputs = []
data=pd.read_csv('~//data//Anainfo20172018.csv')


# def eval_genomes1(genomes,data_idx,config):
#     #print('The data index is %d',data_idx)
#     global inputs,outputs
#     eval_input=inputs[data_idx:data_idx+5000]
#     eval_output=outputs[data_idx:data_idx+5000]
#     for genome_id, genome in genomes:
#         AUX = 0.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for i,o in zip(eval_input,eval_output):
#             output = net.activate(i)
#             aux = np.argmax(output)
#             if(aux == np.argmax(o)): 
#                 AUX+=(output[0] - output[1]) ** 2
#             else:
#                 AUX-=(output[0] - output[1]) ** 2
#         genome.fitness = AUX

# def eval_genomes(genomes,data_idx,intervals,config):
#     #print('The data index is %d',data_idx)
#     global inputs,outputs
#     eval_input=inputs[data_idx:data_idx+intervals]
#     eval_output=outputs[data_idx:data_idx+intervals]
#     for genome_id, genome in genomes:
#         AUX = 0.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for i,o in zip(eval_input,eval_output):
#             output = net.activate(i)
#             aux = np.argmax(output)
#             if(aux == np.argmax(o)): 
#                 AUX+=1
#         genome.fitness = AUX/len(eval_input)

# def eval_genomes(genomes,data_idx,intervals,config):
#     #print('The data index is %d',data_idx)
#     global inputs,outputs
#     eval_input=inputs[data_idx:data_idx+intervals]
#     eval_output=outputs[data_idx:data_idx+intervals]
#     for genome_id, genome in genomes:
#         AUX = 0.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         pred=[]
#         eval=[]
#         for i,o in zip(eval_input,eval_output):
#             output = net.activate(i)
#             aux = np.argmax(output)
#             pred.append(aux)
#             eval.append(np.argmax(o))
#         result=accuracy_function(eval,pred)
#         genome.fitness = result[1]+result[2]
#         #genome.fitness = result[0]

#profit
# def eval_genomes(genomes,data_idx,intervals,config):
#     #print('The data index is %d',data_idx)
#     global inputs,outputs
#     eval_input=inputs[data_idx:data_idx+intervals]
#     eval_output=outputs[data_idx:data_idx+intervals]
#     for genome_id, genome in genomes:
#         AUX = 0.0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         pred=[]
#         eval=[]
#         for i,o in zip(eval_input,eval_output):
#             output = net.activate(i)
#             aux = np.argmax(output)
#             pred.append(aux)
#             eval.append(np.argmax(o))
        
#         profit=0
#         for i in range(len(eval)):
#             result=eval[i]
#             #print(eval_input[i],eval_output[i],result)
#             apply_money=eval_input[i][0]
#             return_money=eval_input[i][1]*eval_input[i][2]
#             if result==pred[i]:
#                 if result==0:
#                     profit=profit+apply_money
#                 else:
#                     profit=profit+return_money-apply_money
#             else:
#                 if result==0:
#                     profit=profit-apply_money
#                 else:
#                     profit=profit-(return_money-apply_money)

#         genome.fitness = profit

#profit+past_profit
def eval_genomes(genomes,data_idx,intervals,config):
    #print('The data index is %d',data_idx)
    global inputs,outputs
    eval_input=inputs[data_idx:data_idx+intervals]
    eval_output=outputs[data_idx:data_idx+intervals]
    for genome_id, genome in genomes:
        AUX = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        pred=[]
        eval=[]
        for i,o in zip(eval_input,eval_output):
            output = net.activate(i)
            aux = np.argmax(output)
            pred.append(aux)
            eval.append(np.argmax(o))
        
        profit=0
        for i in range(len(eval)):
            result=eval[i]
            #print(eval_input[i],eval_output[i],result)
            apply_money=eval_input[i][0]
            return_money=eval_input[i][1]*eval_input[i][2]
            if result==pred[i]:
                if result==0:
                    profit=profit+apply_money
                else:
                    profit=profit+return_money-apply_money
            else:
                if result==0:
                    profit=profit-apply_money
                else:
                    profit=profit-(return_money-apply_money)
        genome.pastfitness=genome.fitness if genome.fitness !=None else 0
        genome.fitness = 0.000001*profit+0.1*genome.pastfitness
        

def eval_test_next(genome,idx,intervals,config):
    global inputs,outputs
    acc=[]
    eval_input=inputs[idx:idx+intervals]
    eval_output=outputs[idx:idx+intervals]
    accuracy = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    pred=[]
    eval=[]
    for i,o in zip(eval_input,eval_output):
        output = net.activate(i)
        aux = np.argmax(output)
        pred.append(aux)
        eval.append(np.argmax(o))
        #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
    return accuracy_function(eval,pred)

def accuracy_function(eval,pred):
    
    good_index= np.where(np.array(eval)==1)
    bad_index=np.where(np.array(eval)==0)
    accuracy=0
    good_acc=0
    bad_acc=0

    for i in range(len(eval)):
        result=eval[i]

        if result==pred[i]:

            accuracy=accuracy+1

            if result==0:
                bad_acc=bad_acc+1
            else:
                good_acc=good_acc+1
    #print(accuracy,good_acc,bad_acc)
    good_acc=good_acc/len(list(good_index[0])) if len(list(good_index[0])) != 0 else 0
    bad_acc=bad_acc/len(list(bad_index[0])) if len(list(bad_index[0])) != 0 else 0
    return accuracy/len(pred),good_acc,bad_acc

#print('2007-2013 ev-profit0.000001*profit+0.5*genemo.pastfitness  alldata3   interval 500   T=10')
#print('2018 ev-0.000001*profit+0.5*genome.pastfitness  Anainfo2018fifo-data   interval 1000   T=10')
#print('0.000001profit0.5past20172018I1000T10')
def run(config_file):
    # Load configuration.
    intervals=1000
    print('0.000001profit0.1past20172018I'+str(intervals)+'T10')
    data['issue_d'] = pd.to_datetime(data['issue_d'],errors = 'coerce')
    #train = data.loc[data['issue_d'] <  data['issue_d'].quantile(0.9)]
    #test =  data.loc[data['issue_d'] >= data['issue_d'].quantile(0.9)]
    train=data
    train.drop('issue_d', axis=1, inplace=True)
    #test.drop('issue_d', axis=1, inplace=True)
    global inputs,outputs
    outputs=code_answer(train[label])
    X=train.drop(labels=label, axis=1, inplace=False)
    inputs=X.values.tolist()
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.  
    winner = p.run(eval_genomes,eval_test_next,intervals,len(outputs),500000)
    #print(stats.get_fitness_mean())
    #print(stats.get_fitness_best())
    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    #print('\nOutput:')
    # test_inputs = []
    # test_outputs = code_answer(test[label])
    # X_=test.drop(labels=label, axis=1, inplace=False)
    # test_inputs=X_.values.tolist()
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # accuracy = 0
    # for i,o in zip(test_inputs,test_outputs):
    #     output = winner_net.activate(i)
    #     aux = np.argmax(output)
    #     if(aux == np.argmax(o)): accuracy+=1
    #     #print("input {!r}, expected output {!r}, got {!r}, {!r}".format(i, o, aux,output))
    # accuracy = accuracy/len(test_inputs)
    # print('Final accuracy {!r}'.format(accuracy))




if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
