import pandas as pd
import numpy as np
import math
import csv_cleanup
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TESTSPLIT = 0.2
VALIDATIONSPLIT = 0.25
FILENAME = 'historic_deals.csv'
FILEPATH = ''
XCOLUMNS = ['ReleaseDate', 'OriginalPrice', 'DiscountPCT']
#XCOLUMNS = ['date_numeric', 'OriginalPrice']
YCOLUMN = 'DiscountPCT'

class DecisionTreeNode():
    def __init__(self, classes, examples, attributes, depth=0):
        self.classes = classes
        self.examples = examples
        self.attributes = attributes
        self.best_attribute = None
        self.pure = False
        self.depth = depth
        self.calculate_num_each_class()
        self.children = []
        #print(self.num_each_class)
        
    def __repr__(self):
        if self.pure:
            return 'Pure: Depth: ' + str(self.depth) + str(self.best_attribute) + "\n(" + str(self.children) + ")"
        else:
            return 'Impure: Depth: ' + str(self.depth) + ' ' + str(self.best_attribute) + "\n(" + str(self.children) + ")"

    def loop_and_create_tree(self):
        if self.check_end_conditions():
            return True
        
        next_attribute = self.pick_next_attribute()
        sets = self.split_set_binary(next_attribute)
        sets_attributes = self.what_attributes_are_left(next_attribute)
       

        for set_index in range(len(sets)):
            child = DecisionTreeNode(self.classes, sets[set_index], sets_attributes[set_index], depth=self.depth+1)
            self.children.append(child)
            child.loop_and_create_tree()
    
    def follow_tree(self, example):
        if self.check_end_conditions():
            return self.most_common_category
        else:
            if self.best_attribute.has_attribute(example):
                #TODO: adapt for nonbinary case
                return (self.children[0]).follow_tree(example)
            else:
                return (self.children[1]).follow_tree(example)

    def predict(self, example):
        return self.follow_tree(example)
        
    def calculate_num_each_class(self):
        'Calculates how many examples correspond to each class'
        self.num_examples = len(self.examples)
        self.num_each_class = []
        for category in self.classes:
            self.num_each_class.append(0)
        
        

        for index, row in self.examples.iterrows():
            self.num_each_class[self.which_class(row[YCOLUMN])] = self.num_each_class[self.which_class(row[YCOLUMN])] + 1
        
        return self.num_each_class
    
    def which_class(self, target_attribute):
        'Given a target attribute, assigns it to the index of the class the example belongs to'
        for category in range(len(self.classes)):
            if target_attribute < self.classes[category]:
                return category
    def what_attributes_are_left(self, chosen_attribute):
        ''' Figure out which attributes go in which sets after a real-valued attribute is used'''
        #TODO: Adapt for non-real valued binary versions

        attributes_in_set1 = []
        attributes_in_set2 = []

        for attribute in self.attributes:
            if attribute.col != chosen_attribute.col:
                attributes_in_set1.append(attribute)
                attributes_in_set2.append(attribute)

            else:
                if attribute.cutoff < chosen_attribute.cutoff:
                    attributes_in_set1.append(attribute)

                elif attribute.cutoff > chosen_attribute.cutoff:
                    attributes_in_set2.append(attribute)

        return attributes_in_set1, attributes_in_set2


    def check_end_conditions(self):
        num_categories_present = 0
        most_common_category_number = 0
        self.most_common_category = self.classes[0]
        for category_index in range(len(self.num_each_class)):
            if self.num_each_class[category_index] != 0:
                num_categories_present += 1
                if most_common_category_number < self.num_each_class[category_index]:
                    most_common_category_number = self.num_each_class[category_index]
                    self.most_common_category = self.classes[category_index]
        
        if num_categories_present <= 1:
            self.confidence = most_common_category_number
            self.pure = True
            #print('pure node')

            return True
        if len(self.attributes) == 0:
            #print('impure node')
            return True
        
        
        return False
    
    def pick_next_attribute(self):
        self.best_attribute = self.attributes[0]
        best_attribute_entropy_gain = 0
        for attribute in self.attributes:
            entropy_gain = self.evaluate_entropy_gain_binary(attribute)
            #print(entropy_gain)
            if best_attribute_entropy_gain < entropy_gain:
                self.best_attribute = attribute
                best_attribute_entropy_gain = entropy_gain

        #print(best_attribute_entropy_gain)
        return self.best_attribute
        
    def evaluate_entropy(self):
        entropy = 0
        for category in self.num_each_class:
            if category != 0:
                entropy -= (category/self.num_examples)*math.log2(category/self.num_examples)
        
        return entropy
    
    def evaluate_entropy_gain_binary(self, attribute):
        #TODO: adapt for non-binary attributes
        sets = self.split_set_binary(attribute)
        sets_attributes = self.what_attributes_are_left(attribute)

        nodes = []
        entropy = 0

        for set_index in range(len(sets)):
            nodes.append(DecisionTreeNode(self.classes, sets[set_index], sets_attributes[set_index]))
            #TODO: this shouldn't be attributes as the last thing. It should reduce the # of attributes.
            entropy += ((nodes[-1].num_examples)/self.num_examples)*(nodes[-1].evaluate_entropy())
        
        information_gain = self.evaluate_entropy() - entropy
        return information_gain
            
    
    def split_set_binary(self, attribute):
        '''The absolute worst way to do this'''
        '''attribute_indices = []
        not_attribute_indices = []
        for index, row in self.examples.iterrows():
            if attribute.has_attribute(row):
                attribute_indices.append(index)
            else:
                not_attribute_indices.append(index)
        print(len(attribute_indices), len(not_attribute_indices))
        '''
        set1 = self.examples.loc[lambda df: attribute.has_attribute_dataframe(df)]
        set2 = self.examples.loc[lambda df: ~(attribute.has_attribute_dataframe(df))]
        
        #print(self.examples.shape, set1.shape, set2.shape)
        

        return set1, set2


class Attribute():
    def __init__(self):
        pass
    def has_attribute(self, example):
        return False
    def has_attribute_dataframe(self, example):
        return False

class RealAttribute(Attribute):
    def __init__(self, column, cutoff):
        self.col = column
        self.cutoff = cutoff
    
    def __repr__(self):
        return self.col + ": " + str(self.cutoff)

    def has_attribute_dataframe(self, example):
        return example[self.col] < self.cutoff
    
    def has_attribute(self, example):
        if example[self.col] < self.cutoff:
            return True
        else:
            return False


def generate_class_cutoffs(num_classes, bottom=0, difference=10):
    classes = []
    for num in range(num_classes):
        classes.append(bottom + (num+1)*difference)
    
    return classes

def break_real_attribute_into_cutoffs(num_classes, dataframe, attribute):
    #TODO: CHANGE THIS TO MATCH C4.5?
    minimum = dataframe[attribute].min()
    maximum = dataframe[attribute].max()

    return generate_class_cutoffs(num_classes, minimum, (maximum-minimum)/num_classes)

def create_real_attribute(dataframe, attribute, num_classes=5):
    broken_attribute = []
    for cutoff in break_real_attribute_into_cutoffs(num_classes, dataframe, attribute):
        broken_attribute.append(RealAttribute(attribute, cutoff))
    
    return broken_attribute

def evaluate_efficacy(model, test_set):
    total_error = 0
    squared_error = 0
    num_examples = 0
    model_predictions_y = []
    actual_discount_y = []
    error_y = []
    indices = []

    for index, row in test_set.iterrows():
        num_examples += 1

        # Building our graphs' datasets
        model_predictions_y.append(model.predict(row))
        actual_discount_y.append(row[YCOLUMN])
        indices.append(index)

        error = model_predictions_y[-1] - row[YCOLUMN]
        error_y.append(error)
        
        total_error += abs(error)
        squared_error += error*error
        

        #if num_examples < 10:
        #    print(error, num_examples)

    mean_error = total_error/(num_examples)
    mean_squared_error = squared_error/(num_examples)

    plt.scatter(indices, model_predictions_y, label="model predictions", c='black', s=2)
    plt.scatter(indices, actual_discount_y, label="actual discount", c='red', s=2)
    plt.scatter(indices, error_y, s=2)
    plt.show()
    plt.scatter(indices[:50], model_predictions_y[:50], label="model predictions", c='black', s=2)
    plt.scatter(indices[:50], actual_discount_y[:50], label="actual discount", c='red', s=2)
    plt.scatter(indices[:50], error_y[:50], s=2)
    plt.show()

    return mean_error, mean_squared_error

def train_test_split(dataframe, training_size=0.8):
    # This was partially stolen from Rachelle 

    size = dataframe.shape[0]

    randomized_dataframe = dataframe.sample(frac=1)

    train = randomized_dataframe[:int(training_size * size)]
    test = randomized_dataframe[int(training_size * size):]

    return train, test

if __name__ == '__main__':
    training_set, testing_set = train_test_split(csv_cleanup.process_file(FILENAME, XCOLUMNS, YCOLUMN, FILEPATH)[0], training_size=0.8)

    classes = generate_class_cutoffs(20, difference=5)
    attributes = create_real_attribute(training_set, 'date_numeric') + create_real_attribute(training_set, 'OriginalPrice')
    print(classes)
    print(attributes)
    print("-" * 40)
    root = DecisionTreeNode(classes, training_set, attributes)
    '''print(root.evaluate_entropy())
    #print(root.split_set_binary(attributes[6]))

    print(root.pick_next_attribute())
    print(root)
    print('-' * 40)'''
    root.loop_and_create_tree()
    #print(root)
    print(testing_set.iloc[0])
    #print(root.classes)
    print(root.follow_tree(testing_set.iloc[0]))
    print(evaluate_efficacy(root, testing_set))
    