# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from nltk import bigrams, trigrams
from nltk.corpus import stopwords
from scipy import stats
from random import randint

# classes
classes = None

# count
count = 0

# number of trees in random forest
num_of_trees = 15

# random forest flag
is_random_forest = True

# early stopping threshold
early_stopping_thr = 30

class Node(object):
    def __init__(self, best_attribute, best_threshold):
        global count        
        count += 1
        self.attribute = best_attribute
        self.threshold = best_threshold
        self.left = None
        self.right = None
        self.distribution = None
        self.is_leaf = False
    
    def set_left(self, node):
        '''sets the right child'''
        self.left = node

    def set_right(self, node):
        '''sets the right child'''
        self.right = node
    
    def set_distribution(self, distribution):
        '''sets the distribution for leaf node'''
        self.distribution = distribution
        self.is_leaf = True
        
# data processing
def club(x):
    return pd.Series(dict(product_attributes = "%s" % ', '.join(x['product_attributes'])))
        
def data_processing(training_data, description_data, attribute_data):
    '''loads the data from the multiple csv files and merges them'''    
    training_data = pd.read_csv("train.csv", encoding = "ISO-8859-1")
    description_data = pd.read_csv("product_descriptions.csv", encoding="ISO-8859-1")
    attribute_data = pd.read_csv("attributes.csv", encoding="ISO-8859-1")
    
    '''merges the training, test, product description and attributes data'''
    #merge training and descriptions data
    training_data = pd.merge(training_data, description_data, on="product_uid", how="left")
    
    # extract brand names and merge
    brand_data = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "product_brand"})
    training_data = pd.merge(training_data, brand_data, on="product_uid", how="left")
    training_data.product_brand.fillna("", inplace=True)
    
    attribute_data.value.fillna("", inplace=True)
    attribute_data['product_attributes'] = attribute_data.name.str.cat(' '+ attribute_data.value) 
    temp = training_data.join(attribute_data.groupby('product_uid').apply(club), on = 'product_uid' ,rsuffix='_r')
    
    #reordering of columns
    cols = temp.columns.values.tolist()
    relevance = cols.pop(cols.index('relevance'))
    search_term = cols.pop(cols.index('search_term'))
    cols.append(search_term)
    cols.append(relevance)
    
    # merged training data
    training_data = temp[cols]
    
    # dropping id
    del training_data['id']
    
    return training_data
    
def extract_features(product_title, product_description, product_brand, product_attributes, search_term):
    '''extracts features from pairs like (search_term, product_title), (search_term, product_description)
    features that will be built:
    1. title_count
    2. description_ng1_count
    3. description_ng2_count
    4. description_ng3_count
    5. brand_count
    6. attributes_count
    7. sterms_prop_in_title
    8. sterms_prop_in_desc
    9. sterms_prop_in_atts
    10.sterms_freq_in_brand
    11.sterms_matched
    @return returns a list of features'''
    # converting all the attributes to lowercase
    
    product_title = product_title.lower()
    if isinstance(product_description, str):
        product_description = product_description.lower()
    else:
        product_description = ''
    if isinstance(product_brand, str):
        product_brand = product_brand.lower()
    else:
        product_brand = ''
    if isinstance(product_attributes, str):
        product_attributes = product_attributes.lower()
    else:
        product_attributes = ''
    search_term = search_term.lower()

    stop_words = stopwords.words('english')
    tokens = search_term.split()
    search_len = len(tokens)
    
    #initializing variables to keep track of the frequency of search terms in various attributes
    title_count = 0
    brand_count = 0
    attributes_count = 0
    description_ng1_count = 0
    description_ng2_count = 0
    description_ng3_count = 0
    
    #initializing variables to keep track of how many search terms were found in various attributes
    tmatch = 0
    dmatch = 0
    bmatch = 0
    amatch = 0
    
    # iterates thrice for n = 1,2,3 in n-gram
    for n in range(1,4):
        if n == 1:
            words = tokens
        elif n == 2:
            words = bigrams(tokens) #when n-gram is 2-grams
        else :
            words = trigrams(tokens) #when n-gram is 3-grams
        
        for word in words:
            if word not in stop_words:
                if len(word) == 1:
                    word = " " + word + " "
                if(len(word) > 2 and n == 1):
                    # flags to check if a word in search_term was found in various attributes
                    found_in_title = False
                    found_in_desc = False
                    found_in_brand = False
                    found_in_atts = False
                    
                    for end_pointer in range(len(word), 2, -1):
                        trimmed_word = word[:end_pointer]
                        if not found_in_title:
                            tcount = product_title.count(trimmed_word)
                            if tcount > 0:
                                found_in_title = True
                                tmatch += 1
                        if not found_in_desc:
                            dcount = product_description.count(trimmed_word)
                            if dcount > 0:
                                found_in_desc = True
                                dmatch += 1
                        if not found_in_brand:
                            bcount = product_brand.count(trimmed_word)
                            if bcount > 0:
                                found_in_brand = True
                                bmatch += 1
                        if not found_in_atts:
                            acount = product_attributes.count(trimmed_word)
                            if acount > 0:
                                found_in_atts = True
                                amatch += 1
                else:
                    # if size of letter is 2 or its 2-gram or 3-gram
                    tcount = product_title.count(' '.join(word))
                    dcount = product_description.count(' '.join(word))                    
                    bcount = product_brand.count(' '.join(word))
                    acount = product_attributes.count(' '.join(word))
                
                title_count += tcount
                brand_count += bcount
                attributes_count += acount
                # the frequency of words in search_term are broken down for each n-gram
                if n == 1:
                    description_ng1_count += dcount
                elif n == 2:
                    description_ng2_count += dcount
                elif n == 3:
                    description_ng3_count += dcount
            else:
                # if a word is a stopword, decreasing the length of the search_term
                search_len -= 1
    # finding proportion of search_term in various attributes
    sterms_prop_in_title = round(tmatch / search_len, 6) if tmatch > 0 else 0
    sterms_prop_in_desc = round(dmatch / search_len, 6) if dmatch > 0 else 0
    sterms_prop_in_atts = round(amatch / search_len, 6) if amatch > 0 else 0
    num_sterms_matched_in_brand = bmatch
    num_sterms_matched_in_title = tmatch
    return list((title_count, description_ng1_count, description_ng2_count, description_ng3_count, brand_count, attributes_count,
                 sterms_prop_in_title, sterms_prop_in_desc, sterms_prop_in_atts, num_sterms_matched_in_title, 
                 num_sterms_matched_in_brand))

# feature extraction
def feature_extraction(training_data):
    '''extracts features from the training'''
    product_details = {}
    ex_training_data = [] # to store extracted features from training data
    
    for index, example in training_data.iterrows():
        # storing product details for reference during test phase
        if example['product_uid'] not in product_details:
            product_details[example['product_uid']] = [example['product_description'],
                            example['product_brand'], example['product_attributes']]
        # extracting features
        xrow = extract_features(example['product_title'], example['product_description'], example['product_brand'],
                                example['product_attributes'], example['search_term'])
        xrow.append(example['relevance'])
        ex_training_data.append(xrow)
    ex_training_data = np.array(ex_training_data)
    return (ex_training_data, product_details)

# testing function
def test_feature_extraction():
    product_title = "3 in. x 3 in. x 3 in. x 2 in. ABS DWV Hub x Hub x Hub Sanitary Tee"
    product_description = ""
    product_brand = ""
    product_attributes = ""
    search_term = "2 in. abs dwv hub x hub x hub sanitary tee"
    print(extract_features(product_title, product_description, product_brand, product_attributes, search_term))
    
def gen_stats(training_data):
    '''generating and writing statistics to file'''
    fo = open('stats.txt', 'w')
    count = 1
    for column in training_data.T:
        fo.write("column "+ str(count) + '\n')
        count += 1
        fo.write("\n")
        temp = np.unique(column)
        for row in temp:
            fo.write(str(row.tolist()))
            fo.write(" ")
        fo.write("\n")
        temp = stats.itemfreq(column)
        for row in temp:
            fo.write(str(row.tolist()))
            fo.write("\n")
        fo.write("\n")
    fo.close()

def cal_entropy(examples):
    '''to measure the entropy of a node in the decision tree'''
    # generate the count of each class
    s = stats.itemfreq(examples[:,examples.shape[1]-1])
    entropy = 0.0
    for row in s:
        x = row[1]/examples.shape[0]
        entropy += -(x) * np.log2(x)
    return entropy

def get_left_examples(examples, attribute, threshold):
    '''splits the examples into left and right based on threshold, and returns
    the left examples'''
    boolean = examples[:, attribute] < threshold
    left_examples = examples[boolean, :]
    return left_examples

def get_right_examples(examples, attribute, threshold):
    '''splits the examples into left and right based on threshold, and returns
    the right examples'''
    # all the examples whose attrribute value is >= threshold goes to the right
    # child
    boolean = examples[:, attribute] >= threshold
    right_examples = examples[boolean, :]
    return right_examples

def cal_infogain(examples, attribute, threshold):
    '''measure inormation gain for a node after splitting it based on given
    attribute and threshold'''
    entropy_of_parent = cal_entropy(examples)
    # get examples that would go to the left and right node
    left_examples = get_left_examples(examples, attribute, threshold)
    right_examples = get_right_examples(examples, attribute, threshold)
    #calculate the fraction of examples that go to the left and right node
    left_fraction = left_examples.shape[0]/examples.shape[0]
    right_fraction = right_examples.shape[0]/examples.shape[0]
    
    entropy_of_left = cal_entropy(left_examples)
    entropy_of_right = cal_entropy(right_examples)
    # calculate the information gain
    infogain = entropy_of_parent - (left_fraction * entropy_of_left +
    right_fraction * entropy_of_right)
    return infogain

def get_distribution(examples):
    '''returns the distribution of the examples based on examples belonging to 
    each classes'''
    dist = {}
    # generate the count of each class
    s = stats.itemfreq(examples[:,examples.shape[1]-1])
    for row in s:
        dist[row[0]] = row[1]/examples.shape[0]
    # setting the count of missing classes in examples to 0
    for cls in classes:
        if cls not in dist:
            dist[cls] = 0.0
    return dist

def choose_random_attribute(examples, attributes):
    '''chooses a random attribute from the set of attributes and the best
    threshold based on certain criteria'''
    rand_attribute = randint(0, attributes.shape[0] - 1)
    max_gain = best_threshold = -1
    # generating the probable thresholds
    col = examples[:, rand_attribute]
    unique_values = np.unique(col)
    last_val = unique_values[unique_values.size - 1]
    unique_values = np.append(unique_values, last_val + 1)
    # finding the threshold that gives the maximum infogain
    for thr in unique_values:
        infogain = cal_infogain(examples, rand_attribute, thr)
        if infogain > max_gain and infogain > 0:
            max_gain = infogain
            best_threshold = thr
    return (rand_attribute, best_threshold)

def choose_attribute(examples, attributes):
    '''chooses the best attribute and threshold from the set of attributes
    based on certain criteria'''
    max_gain = best_attribute = best_threshold = -1
    # attributes ranges from 0 to n - 1, where n is the number of attributes
    for att in attributes:
        col = examples[:, att]
        unique_values = np.unique(col)
        last_val = unique_values[unique_values.size - 1]
        unique_values = np.append(unique_values, last_val + 1)
        for thr in unique_values:
            infogain = cal_infogain(examples, att, thr)
            if infogain > max_gain and infogain > 0:
                max_gain = infogain
                best_attribute = att
                best_threshold = thr
    return (best_attribute, best_threshold)

def is_homogeneous(examples):
    '''checks if all the examples belong to the same class'''
    col = examples[:, examples.shape[1]-1]
    unique_values = np.unique(col)
    if unique_values.shape[0] == 1:
        return True
    else:
        return False

def DTL(examples, attributes, default):
    '''builds a decision tree from the training data'''
    if examples.shape[0] < early_stopping_thr:
        leaf_node = Node(None, None)
        leaf_node.set_distribution(default)
        return leaf_node
    elif is_homogeneous(examples):
        leaf_node = Node(None, None)
        leaf_node.set_distribution(get_distribution(examples))
        return leaf_node
    else:
        if is_random_forest:
            best_attribute, best_threhold = choose_random_attribute(examples, attributes)
        else:
            best_attribute, best_threhold = choose_attribute(examples, attributes)
        # for cases when no possbible split can give any information gain
        if best_threhold == -1:
            leaf_node = Node(None, None)
            leaf_node.set_distribution(get_distribution(examples))
            return leaf_node
        tree = Node(best_attribute, best_threhold)
        examples_left = get_left_examples(examples, best_attribute, best_threhold)
        examples_right = get_right_examples(examples, best_attribute, best_threhold)
        tree.set_left(DTL(examples_left, attributes, get_distribution(examples)))
        tree.set_right(DTL(examples_right, attributes, get_distribution(examples)))
        return tree

def generate_classes(examples):
    '''get unique classes from the class column of training data and store it
    in a global variable for future reference'''
    global classes
    classes = np.unique(examples[:, examples.shape[1]-1]).tolist()
    
def print_tree(node):
    '''prints the decision tree in depth first order'''
    print()
    '''print's the tree'''
    if node.is_leaf:
        print("distribution:", node.distribution)
        return
    print("attribute:", node.attribute, "threshold:", node.threshold)
    if node.left != None:
        print_tree(node.left)
    if node.right != None:
        print_tree(node.right)
        
def traverse_dt(node, test_record):
    '''deprecated : traverses the decision tree based on the test record and returns
    a predicted class for the test record'''
    if node.is_leaf:
        # get the max probability among all the classes
        max_prob = sorted(node.distribution.values()).pop()
        tied_classes = []
        # get all the classes with max probability
        for key, value in node.distribution.items():
            if value == max_prob:
                tied_classes.append(key)
        # if only one class has max probability
        if len(tied_classes) == 1:
            return tied_classes.pop()
        else:
            return tied_classes[randint(0, len(tied_classes) - 1)]
    else: # if the node is not a leaf node
        att = node.attribute
        thr = node.threshold
        if test_record[att] < thr:
            return traverse_dt(node.left, test_record)
        else:
            return traverse_dt(node.right, test_record)

def traverse_dt1(node, test_record):
    '''traverses the decision tree based on the test record and returns
    a predicted class for the test record'''
    predicted_relevance = 0
    if node.is_leaf:
        for key, value in node.distribution.items():
            predicted_relevance += key * value
        return predicted_relevance
    else: # if the node is not a leaf node
        att = node.attribute
        thr = node.threshold
        if test_record[att] < thr:
            return traverse_dt1(node.left, test_record)
        else:
            return traverse_dt1(node.right, test_record)

def test_rf(trees, product_details, test_file):
    '''predit the relevance for each test record using random forest'''
    id_list = []
    relevance_list = []
    test_data = pd.read_csv(test_file, encoding="ISO-8859-1")
    for index, row in test_data.iterrows():
        id = row['id']
        product_uid = row['product_uid']
        product_title = row['product_title']
        search_term = row['search_term']
        if product_uid not in product_details:
            product_description = ""
            product_brand = ""
            product_attributes = ""
        else:
            product_description = product_details[product_uid][0]
            product_brand = product_details[product_uid][1]
            product_attributes = product_details[product_uid][2]
        test_record = extract_features(product_title, product_description, product_brand,
                                product_attributes, search_term)
        # iterate over all the trees in the random forest
        predicted_relevance = 0
        for i in range(num_of_trees):
            predicted_relevance += traverse_dt1(trees[i], test_record)
        predicted_relevance = predicted_relevance/num_of_trees
        id_list.append(id)
        relevance_list.append(predicted_relevance)
    output = {'id': id_list, 'relevance': relevance_list}
    pd_output = pd.DataFrame(output, columns=['id', 'relevance'])
    pd_output.to_csv('result.csv', encoding='utf-8', index = False)

def test_dt(tree, product_details, test_file):
    '''predit the relevance for each test record using decision tree'''
    id_list = []
    relevance_list = []
    test_data = pd.read_csv(test_file, encoding="ISO-8859-1")
    for index, row in test_data.iterrows():
        id = row['id']
        product_uid = row['product_uid']
        product_title = row['product_title']
        search_term = row['search_term']
        if product_uid not in product_details:
            product_description = ""
            product_brand = ""
            product_attributes = ""
        else:
            product_description = product_details[product_uid][0]
            product_brand = product_details[product_uid][1]
            product_attributes = product_details[product_uid][2]
        test_record = extract_features(product_title, product_description, product_brand,
                                product_attributes, search_term)
        predicted_relevance = traverse_dt1(tree, test_record)
        id_list.append(id)
        relevance_list.append(predicted_relevance)
    output = {'id': id_list, 'relevance': relevance_list}
    pd_output = pd.DataFrame(output, columns=['id', 'relevance'])
    pd_output.to_csv('result.csv', encoding='utf-8', index = False)
    
# main
def main():
    '''main function'''
    training_data = "train.csv"
    test_file = "test.csv"
    description_data = "product_descriptions.csv"
    attribute_data = "attributes.csv"
    print("stage 1 of 4, reading data")
    training_data = data_processing(training_data, description_data, attribute_data)
    training_data.to_csv('out.csv', encoding='utf-8', index = False)
    
    print("stage 2 of 4, extracting features")
    examples, product_details = feature_extraction(training_data)
    generate_classes(examples)
    
    # building the classifier
    if is_random_forest: # in case of random forest
        print("stage 3 of 4, building random forest")
        trees = []
        for i in range(num_of_trees):
            print("building tree", i)
            trees.append(DTL(examples, np.arange(examples.shape[1] - 1), get_distribution(examples)))
    else: # in case of optimized decision tree
        print("stage 3 of 4, building decision tree")
        tree = DTL(examples, np.arange(examples.shape[1] - 1), get_distribution(examples))

    # print("Total number of nodes", count)
    
    # testing the classifier
    print("stage 4 of 4, predicting relevance for the test data")
    if is_random_forest:
        test_rf(trees, product_details, test_file)
    else:
        test_dt(tree, product_details, test_file)
    print("finished")

#initiating the program
main()
