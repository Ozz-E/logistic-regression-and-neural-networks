import matplotlib.pyplot as plt
import os

def plot_cost_train_test(train_cost, test_cost, ax1):

    """ This function plots the training and test cost of the neural network.

    """    

    ax1.set_title('Cost vs Iterations')
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax1.plot(train_cost, c='red', label='train cost')
    ax1.plot(test_cost, c='blue', label='test cost')
    
    # add legend to the subplot
    ax1.legend()
