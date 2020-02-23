# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 22:35:22 2016

@author: Dmitry
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from pylab import plot, legend, subplot, grid, xlabel, ylabel, show, title, savefig
from pyneurgen.nodes import BiasNode, Connection
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import NARXRecurrent

##Load data
df = pd.read_csv('C:/Users/Dmitry/Desktop/Schultz group/\
Experiments/Modeling/Neral network/20160401_data_for_NN.csv')

#unque_sensors
Sensor_arr = df['Sensor'].unique()

#prepare_dummy variables for each sensor and run neuron network training==
#=========================================================================
for sn in Sensor_arr:
    Sel_part1 = 'Sensor == "'
    Sel_part2 = '"'
    Sel = Sel_part1 + sensor + Sel_part2
    df_sel_sen = df.query(Sel)
    dummies = pd.get_dummies(df_sel_sen['Stimulus'])
#select relyable stimulations    
    dummies_select = dummies[['Mock','EGF_6_25ngml','EGF_12_5ngml','EGF_25ngml',
                              'EGF_50ngml','EGF_100ngml',
                              'IGF_6_25ngml','IGF_100ngml','TNFa_100ngml',
                              'EGF_IGF_100ngml_100ngml','EGF_IGF_100ngml_6_25ngml',
                              'EGF_IGF_6_25ngml_100ngml','EGF_TNFa_100ngml_100ngml',
                              'IGF_TNFa_100ngml_100ngml']]
'''generation dummy variable for each stimulatin
concentration was normalyzed to have range between -1 and 1.
where -1 is 0 ng/ml
      -0.875 is 6.25 ng/ml
      -0.5 is 25 ng/ml
      1 is 100 ng/ml
''' 
    df_sel_sen_with_dummies = df_sel_sen.join(dummies_select)
    df_sel_sen_with_dummies['All selected'] = df_sel_sen_with_dummies['Mock'] + \
                                            df_sel_sen_with_dummies['EGF_6_25ngml'] + \
                                            df_sel_sen_with_dummies['EGF_12_5ngml'] + \
                                            df_sel_sen_with_dummies['EGF_25ngml'] + \
                                            df_sel_sen_with_dummies['EGF_50ngml'] + \
                                            df_sel_sen_with_dummies['EGF_100ngml'] + \
                                            df_sel_sen_with_dummies['IGF_6_25ngml'] + \
                                            df_sel_sen_with_dummies['IGF_100ngml'] + \
                                            df_sel_sen_with_dummies['TNFa_100ngml'] + \
                                            df_sel_sen_with_dummies['EGF_IGF_100ngml_100ngml'] + \
                                            df_sel_sen_with_dummies['EGF_IGF_100ngml_6_25ngml'] + \
                                            df_sel_sen_with_dummies['EGF_IGF_6_25ngml_100ngml'] + \
                                            df_sel_sen_with_dummies['EGF_TNFa_100ngml_100ngml'] + \
                                            df_sel_sen_with_dummies['IGF_TNFa_100ngml_100ngml']
    
    df_sel_sen_with_dummies_selected = df_sel_sen_with_dummies[df_sel_sen_with_dummies['All selected'] >= 0.5].copy()
    df_sel_sen_with_dummies_selected = df_sel_sen_with_dummies_selected.drop('Sensor',1)
    
    df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_100ngml__EGF'] = df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_100ngml__IGF'] = df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_6_25ngml__EGF'] = df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_6_25ngml'] * 1
    df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_6_25ngml__IGF'] = df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_6_25ngml'] * -0.875
    df_sel_sen_with_dummies_selected['EGF_IGF_6_25ngml_100ngml__EGF'] = df_sel_sen_with_dummies_selected['EGF_IGF_6_25ngml_100ngml'] * -0.875
    df_sel_sen_with_dummies_selected['EGF_IGF_6_25ngml_100ngml__IGF'] = df_sel_sen_with_dummies_selected['EGF_IGF_6_25ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['EGF_TNFa_100ngml_100ngml__EGF'] = df_sel_sen_with_dummies_selected['EGF_TNFa_100ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['EGF_TNFa_100ngml_100ngml__TNF'] = df_sel_sen_with_dummies_selected['EGF_TNFa_100ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['IGF_TNFa_100ngml_100ngml__IGF'] = df_sel_sen_with_dummies_selected['IGF_TNFa_100ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['IGF_TNFa_100ngml_100ngml__TNF'] = df_sel_sen_with_dummies_selected['IGF_TNFa_100ngml_100ngml'] * 1
    df_sel_sen_with_dummies_selected['Mock'] = df_sel_sen_with_dummies_selected['Mock'] * -1
    df_sel_sen_with_dummies_selected['EGF_6_25ngml'] = df_sel_sen_with_dummies_selected['EGF_6_25ngml'] * -0.875
    df_sel_sen_with_dummies_selected['EGF_12_5ngml'] = df_sel_sen_with_dummies_selected['EGF_12_5ngml'] * -0.75
    df_sel_sen_with_dummies_selected['EGF_25ngml'] = df_sel_sen_with_dummies_selected['EGF_25ngml'] * -0.5
    df_sel_sen_with_dummies_selected['EGF_50ngml'] = df_sel_sen_with_dummies_selected['EGF_50ngml'] * 3
    df_sel_sen_with_dummies_selected['EGF_100ngml'] = df_sel_sen_with_dummies_selected['EGF_100ngml'] * 1
    df_sel_sen_with_dummies_selected['IGF_100ngml'] = df_sel_sen_with_dummies_selected['IGF_100ngml'] * 1
    df_sel_sen_with_dummies_selected['IGF_100ngml'] = df_sel_sen_with_dummies_selected['IGF_100ngml'] * 1
    df_sel_sen_with_dummies_selected['IGF_6_25ngml'] = df_sel_sen_with_dummies_selected['IGF_6_25ngml'] * -0.875
    df_sel_sen_with_dummies_selected['TNFa_100ngml'] = df_sel_sen_with_dummies_selected['TNFa_100ngml'] * 1
    
    df_sel_sen_with_dummies_selected['EGF'] = df_sel_sen_with_dummies_selected['Mock'] + \
                                            df_sel_sen_with_dummies_selected['EGF_6_25ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_12_5ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_25ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_50ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_100ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_100ngml__EGF'] + \
                                            df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_6_25ngml__EGF'] + \
                                            df_sel_sen_with_dummies_selected['EGF_IGF_6_25ngml_100ngml__EGF']
                                            
    df_sel_sen_with_dummies_selected['IGF'] = df_sel_sen_with_dummies_selected['Mock'] + \
                                            df_sel_sen_with_dummies_selected['IGF_100ngml'] + \
                                            df_sel_sen_with_dummies_selected['IGF_6_25ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_100ngml__IGF'] + \
                                            df_sel_sen_with_dummies_selected['EGF_IGF_100ngml_6_25ngml__IGF'] + \
                                            df_sel_sen_with_dummies_selected['EGF_IGF_6_25ngml_100ngml__IGF'] + \
                                            df_sel_sen_with_dummies_selected['IGF_TNFa_100ngml_100ngml__IGF']
                                            
    df_sel_sen_with_dummies_selected['TNF'] = df_sel_sen_with_dummies_selected['Mock'] + \
                                            df_sel_sen_with_dummies_selected['TNFa_100ngml'] + \
                                            df_sel_sen_with_dummies_selected['EGF_TNFa_100ngml_100ngml__TNF'] + \
                                            df_sel_sen_with_dummies_selected['IGF_TNFa_100ngml_100ngml__TNF']
                                            
    
    data = df_sel_sen_with_dummies_selected[['Stimulus','EGF','IGF', 'TNF', 'time', 'Norm_to_Mean_abs_0']].copy()
    
    data_select = data[['EGF','IGF', 'TNF']]
    data_select[data_select == 0] = -1
    data_select[data_select == 3] = 0
    data[['EGF','IGF', 'TNF']] = data_select
    data = data.reset_index(drop=True)
    data.iloc[0:14, 1:4] = -1.0
    data[data['time'] == 0][['EGF','IGF', 'TNF']]
    
    all_targets = np.zeros((len(data),1))
    all_inputs = np.zeros((len(data),4))
    
    for n in range(len(all_targets)):
        all_targets[n][0] = data.Norm_to_Mean_abs_0[[n]]
 
##### time is normalized between 0 and 1  as well   
    for n in range(len(all_targets)):    
        all_inputs[n][0] = data.EGF[[n]]
        all_inputs[n][1] = data.IGF[[n]]
        all_inputs[n][2] = data.TNF[[n]]
        all_inputs[n][3] = data.time[[n]]/16200.
    
### return indexes of test data   
    def index_test(df=data):
        df_sel_sen = df.query('Stimulus in ("EGF_12_5ngml","EGF_50ngml",\
        "EGF_IGF_6_25ngml_100ngml","IGF_TNFa_100ngml_100ngml")')
        return df_sel_sen.index.values

### retern inexes of training data   
    def index_learn(df=data):
        df_sel_sen = df.query('Stimulus in ("Mock","EGF_6_25ngml","EGF_25ngml",\
                              "EGF_100ngml",\
                              "IGF_6_25ngml","IGF_100ngml","TNFa_100ngml",\
                              "EGF_IGF_100ngml_6_25ngml",\
                              "EGF_IGF_100ngml_100ngml","EGF_TNFa_100ngml_100ngml")')
        return df_sel_sen.index.values
    
    indexes_test = index_test ()
    indexes_learn = index_learn()
    
### building NARX neural network======================================================
###======================================================================================

#optimize the network architecture I used 2 hiden layers with 1-20 nodes
    number_nodes_to_try = 20
    num_nodes = range(1,number_nodes_to_try + 1)
    xs, ys=np.meshgrid(num_nodes, num_nodes)
    xs = np.concatenate([xs[0],xs[1],xs[2],xs[3], xs[4], xs[5], xs[6],
                         xs[7], xs[8], xs[9], xs[10], xs[11], xs[12], xs[13],
                        xs[14], xs[15], xs[16], xs[17], xs[18], xs[19], xs[20]])
    ys = np.concatenate([ys[0],ys[1],ys[2],ys[3], ys[4], ys[5], ys[6],
                         ys[7], ys[8], ys[9], ys[10], ys[11], ys[12], ys[13],
                        ys[14], ys[15], ys[16], ys[17], ys[18], ys[19], ys[20]])

#createnetwork and run training                     
    for l1, l2 in zip (xs, ys):
        output_order = 6
        incoming_weight_from_output = 1.
        input_order = 0
        incoming_weight_from_input = 0.
            
        net = NeuralNet()
        net.init_layers(4, [l1,l2], 1, NARXRecurrent(
                    output_order,
                    incoming_weight_from_output,
                    input_order,
                    incoming_weight_from_input))
        
        net.randomize_network()
        net.set_halt_on_extremes(True)
        
        
        #   Set to constrain beginning weights to -.5 to .5
        #       Just to show we can
        
        net.set_random_constraint(.5)
        net.set_learnrate(.1)
        net.set_all_inputs(all_inputs)
        net.set_all_targets(all_targets)
        
        net.set_learn_range(indexes_learn)
        net.get_learn_range()
        net.set_test_range(indexes_test)
        net.get_test_range()
        net.layers[0].set_activation_type('tanh')
        net.layers[1].set_activation_type('tanh')
        net.layers[2].set_activation_type('tanh')
        net.layers[3].set_activation_type('tanh')
        
        ###training network
        net.learn(epochs=1200, show_epoch_results=True,
            random_testing=True)
        
        mse = net.test()
        
        #extract predicted velues
        all_learn = [item[1][0] for item in net.get_learn_data()]
        learn_positions = [item[0][3] for item in net.get_learn_data()]
        
        test_positions = [item[0][3] for item in net.get_test_data()]
        all_targets1 = [item[0][0] for item in net.test_targets_activations]
        allactuals = [item[1][0] for item in net.test_targets_activations]

        #   This is quick and dirty, but it will show the results
        subplot(3, 1, 1)
        plot(learn_positions, all_learn,'bo')
        title("all_targets")
        grid(True)
        
        
        subplot(3, 1, 2)
        plot(test_positions, all_targets1, 'bo', label='targets')
        plot(test_positions, allactuals, 'ro', label='actuals')
        grid(True)
        
        
        subplot(3, 1, 3)
        plot(range(1, len(net.accum_mse) + 1, 1), net.accum_mse)
        xlabel('epochs')
        ylabel('mean squared error')
        grid(True)
        title("Mean Squared Error by Epoch")
        savefig('Out6_weight1_4_10_4_1_epoch1200.pdf', bbox_inches='tight')
        show()
        a = 'blabla'
        net.save(a)
