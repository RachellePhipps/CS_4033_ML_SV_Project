import pandas as pd
import numpy as np
import math
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv_cleanup
import decision_trees
import random

TESTSPLIT = 0.2
VALIDATIONSPLIT = 0.25
FILENAME = 'historic_deals.csv'
FILEPATH = ''
XCOLUMNS = ['ReleaseDate', 'OriginalPrice', 'DiscountPCT']
#XCOLUMNS = ['date_numeric', 'OriginalPrice']
YCOLUMN = 'DiscountPCT'

class RandomForest():
    def __init__(self, target_classes, examples, attributes, num_trees=5, depth=0):
        self.classes = target_classes
        self.training_data = examples
        self.attributes = attributes
        self.generate_trees(num_trees)
        
    def generate_trees(self, num_trees):
        self.trees = []

        for tree in range(num_trees):
            randomized_dataframe = self.training_data.sample(frac=1, replace=True)

            self.trees.append(decision_trees.DecisionTreeNode(self.classes, randomized_dataframe, self.select_attribute_subset()))
            self.trees[tree].loop_and_create_tree()
    
    def predict(self, example):
        sum = 0.0
        for tree in self.trees:
            sum += tree.predict(example)
        
        return sum / float(len(self.trees))

    def select_attribute_subset(self):
        num_attributes_selected = int(math.sqrt(len(self.attributes)))

        possible_attributes = self.attributes.copy()
        attributes_chosen = []

        for attribute in range(num_attributes_selected):
            a = random.choice(possible_attributes)
            possible_attributes.remove(a)
            attributes_chosen.append(a)
        

        return attributes_chosen


if __name__ == '__main__':
    training_set, testing_set = decision_trees.train_test_split(csv_cleanup.process_file(FILENAME, XCOLUMNS, YCOLUMN, FILEPATH)[0], training_size=0.8)

    classes = decision_trees.generate_class_cutoffs(20, difference=5)
    attributes = decision_trees.create_real_attribute(training_set, 'date_numeric') + decision_trees.create_real_attribute(training_set, 'OriginalPrice')
    
    forest = RandomForest(classes, training_set, attributes, num_trees=50)

    print(forest.predict(testing_set.iloc[0]))
    print(decision_trees.evaluate_efficacy(forest, testing_set))