import pandas as pd
import argparse
import csv
import warnings

# ignore all the python3 inbuilt warnings
warnings.filterwarnings("ignore")

"""
Reading and parsing arguments passed from command line
Arguments : 
        data
        output
        maxIteration
"""
parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--output")
parser.add_argument("--maxIteration")
args = parser.parse_args()
print("data: {}".format(args.data))
print("output: {}".format(args.output))
print("maximum iteration: {}".format(args.maxIteration))

# Parsed input data
input_file = args.data
output_path = args.output
max_iteration = int(args.maxIteration)


# data_frame = pd.read_csv("Example.tsv", sep='\t', header=None)
# data_frame = pd.read_csv("Gauss2.tsv", sep='\t', header=None)
data_frame = pd.read_csv(input_file, sep='\t', header=None)
# print(data_frame)


def gradient_function(d, x0=1):
    dw0 = 0
    dw1 = 0
    dw2 = 0
    for (x1, x2, yi) in zip(d['x1'], d['x2'], d['y']):
        dw0 += (yi-(activation_function(d['w0'], d['w1'], d['w2'], x1, x2, x0=1)))*x0
        dw1 += (yi-(activation_function(d['w0'], d['w1'], d['w2'], x1, x2, x0=1)))*x1
        dw2 += (yi-(activation_function(d['w0'], d['w1'], d['w2'], x1, x2, x0=1)))*x2
    return dw0, dw1, dw2


def calculate_missclassification(d, x0=1):
    missclassfication_count = 0
    for (x1, x2, yi) in zip(d['x1'], d['x2'], d['y']):
        if yi != activation_function(d['w0'], d['w1'], d['w2'], x1, x2, x0):
            missclassfication_count += 1
    return missclassfication_count


def activation_function(w0, w1, w2, x1, x2, x0=1):
    return 1 if (linear_sum(w0, w1, w2, x1, x2, x0)) > 0 else 0


def linear_sum(w0, w1, w2, x1, x2, x0):
    return w0 * x0 + w1 * x1 + w2 * x2


def update_weights(d, learning_rate):
    dw0, dw1, dw2 = gradient_function(d, x0=1)
    d['w0'] = d['w0'] + (learning_rate * dw0)
    d['w1'] = d['w1'] + (learning_rate * dw1)
    d['w2'] = d['w2'] + (learning_rate * dw2)
    return d


def get_data_features(d, d_weight):
    for i in data_frame:
        if i == 0:
            d['y'] = data_frame[i]
            d_weight['w{}'.format(i)] = 0
        else:
            d['x{}'.format(i)] = data_frame[i]
            d_weight['w{}'.format(i)] = 0
        d.update(d_weight)
    classify_y(d)
    return d


def classify_y(d):
    for k in range(len(d['y'])):
        if d['y'][k] == 'A':
            d['y'][k] = 1
        else:
            d['y'][k] = 0


def run():
    # Initializing directory with x0. Value of x0 is always 1
    d = {'x0': 1}
    # Directory for storing variable length weight values
    d_weight = {}
    d_constant = get_data_features(d, d_weight)
    d_annealing = d_constant
    # max_iteration = 100
    # output_path = 'output_Example.tsv'
    with open(output_path, 'w') as tsv_file:
        missclassfication_count_constant = calculate_missclassification(d_constant, x0=1)
        tsv_file.write(str(missclassfication_count_constant))
        for i in range(max_iteration):
            d_constant = update_weights(d_constant, learning_rate=1)
            missclassfication_count_constant = calculate_missclassification(d_constant, x0=1)
            tsv_file.write('\t'+str(missclassfication_count_constant))
        tsv_file.write('\n')
        itr_count = 0
        d_annealing['w0'] = 0
        d_annealing['w1'] = 0
        d_annealing['w2'] = 0
        missclassfication_count_annealing = calculate_missclassification(d_annealing, x0=1)
        tsv_file.write(str(missclassfication_count_annealing))
        for i in range(max_iteration):
            itr_count += 1
            d_annealing = update_weights(d_annealing, learning_rate=1/itr_count)
            missclassfication_count_annealing = calculate_missclassification(d_annealing, x0=1)
            tsv_file.write('\t'+str(missclassfication_count_annealing))
    print("Program Completed!!!!")


run()
