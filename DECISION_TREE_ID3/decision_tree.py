
import numpy as np
import os
import graphviz
import pandas as pd
from collections import defaultdict
from collections import Counter
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def partition(x):

    x = x.astype(int)
    x_df = pd.DataFrame(x)
    partion_dict = {}
    for index, row in  x_df[0].iteritems():
        if row not in partion_dict:
            partion_dict[row]=[]
        partion_dict[row].append(index) 
    
    return partion_dict
    
    
    
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def entropy(y):
    
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    #y = pd.DataFrame(y)
    
    
    ycount = Counter(y)
    ytot_count = len(y)
    
    Y_entropy = 0
    for key in ycount.keys():
        prob = ycount[key]/ytot_count
        Y_entropy +=  (-prob)*math.log2(prob)
    
    
    return Y_entropy
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    y_entropy = entropy(y)
    
    df2 = pd.DataFrame(x)
    mydict = defaultdict(list)
    for index, row in  df2[0].iteritems():
        mydict[row].append(index)
    sum1 = 0 
    dictlensum = 0
    for values in mydict.values():
        dictlensum+= (len(values))
    
    x_tot = df2.size
    for key,values in mydict.items():
        sum2 = 0
        y2count = Counter(y[values])
        y2_totcount = y[values].size
        #print(key,y2count,y2_totcount)
        probkey = y2_totcount/x_tot
        h_y_x = entropy(y[x == key])
        H_y_x = probkey*h_y_x
        
        sum1 = sum1+H_y_x
        
    mi = (y_entropy - sum1)
    return mi




    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')

   
    
    
    
    
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
                
    """
#-----------------------------------------------------------------------   
    y_label = pd.DataFrame(y)
    df = pd.DataFrame(x)
    attvallist = list()
    mi = []
    if(y_label[0].unique().size == 1):
        label = y_label[0].unique()
        return label[0]
    
    if(attribute_value_pairs==None):
        attvallist = list()
        
        for index in df.columns:
            #print("index :",index,df[index].unique())
            for val in df[index].unique():
                attvallist.append((index,val))
        attribute_value_pairs = attvallist
    
    if(depth == max_depth or len(attribute_value_pairs)==0):
        #print("Entering empty attval list check")
        tempp = 0
        y_label_list = Counter(y)
        for key in y_label_list:
            if(y_label_list[key] > tempp):
                tempp = y_label_list[key]
                y_label_key = key
        #print("entered second check")
        return y_label_key
        
    
    
        
    #print("entering MI")  
    #print("Printign attribute_val_pairs",attribute_value_pairs)
    #print("Attribute value pairs before new mi",attribute_value_pairs)
    for a,v in attribute_value_pairs:
        mi.append(mutual_information((x[:,a] == v),y))
    #print("Mutual info is :",mi)
    attr , value = attribute_value_pairs[np.argmax(mi)]
    #print(attr,value)
    x_partitions = partition(x[:,attr]==value)
    #print(x_partitions.keys())
    attribute_value_pairs = attribute_value_pairs.remove((attr,value))
    #print(attribute_value_pairs)
    
    
    
    tree = {}
   
    for key , values in x_partitions.items():
        x_sub_parts = x.take(values,axis=0)
        y_sub_labels = y.take(values,axis=0)
        decval = bool(key)
        
        tree[(attr,value,decval)] = id3(x_sub_parts,y_sub_labels,attribute_value_pairs=attribute_value_pairs,depth = (depth+1),max_depth = max_depth)
        
    #print(tree)
    return tree
    
    
        
    
    
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    
    for split_key,sub_dict in tree.items():
        #print(split_tree)
        #print(len(split_tree))
        split_index = split_key[0]
        split_value = split_key[1]
        split_decision = split_key[2]
        decision = (x[split_index] == split_value)
        #print(decision)
        
        if (split_decision == decision):
            if (type(sub_dict) is dict):
                val_label = predict_example(x,sub_dict)
            else:
                val_label = sub_dict
            #print(val_label)            
            return val_label
        
        
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    #print(y_true.size)
    #print(len(y_pred))
    n = y_true.size
    
    
    # INSERT YOUR CODE HERE
    sum = 0
    for i in range(0,n):
        if(y_true[i] != y_pred[i]):
            sum = sum+1
    error = (1/n)*sum
    
    return error
    
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    
    if depth == 0:
        print('TREE')
    
    print(tree)

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
   
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    # Learn a decision tree of depth 3
    
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    # Pretty print it to console
    pretty_print(decision_tree)
    
    
    # Visualize the tree and save it as a PNG image
    
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')
 

    # Compute the test error
    
    y_trn_pred = [predict_example(x, decision_tree) for x in Xtrn]
    trn_err = compute_error(ytrn, y_trn_pred)
    
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    #print("PRINTING YPRED")
    #print(y_pred)
    #print(ytst)

    print('Train Error = {0:4.2f}%.'.format(trn_err * 100))
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    trn_err1 = {}
	tst_err1 = {}
    
    for tree_depth in range(1,11):
        #print("Tree_Depth is:",tree_depth)
        decision_tree = id3(Xtrn, ytrn, max_depth=tree_depth)
        y_trn_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err1[tree_depth] = compute_error(ytrn, y_trn_pred)
    
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err1[tree_depth] = compute_error(ytst, y_pred)
    
    #print("printing error dictionary")
    #print(trn_err1)
    #print(tst_err1)
    
    plt.plot(list(trn_err1.keys()), list(trn_err1.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(tst_err1.keys()), list(tst_err1.values()), marker='s', linewidth=3, markersize=12)
    plt.xlabel(' Depth of Tree', fontsize=16)
    plt.ylabel('Train/Test error', fontsize=16)
    plt.xticks(list(tst_err1.keys()), fontsize=12)
    plt.legend(['Tarin Error', 'Test Error'], fontsize=16)
    plt.figure()
    plt.show()
    
#-------------------------------------------(PART C)---Printing Confusion Matrix ----------------------------#
    decision_tree = id3(Xtrn, ytrn, max_depth=1)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
            #print(y_pred)
            #print(ytst_1)
    print("Confusion Matrix for depth:",1)
    print(confusion_matrix(ytst,y_pred))

    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
            #print(y_pred)
            #print(ytst_1)
    print("Confusion Matrix for depth:",3)
    print(confusion_matrix(ytst,y_pred))

    decision_tree = id3(Xtrn, ytrn, max_depth=5)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
            #print(y_pred)
            #print(ytst_1)
    print("Confusion Matrix for depth:",5)
    print(confusion_matrix(ytst,y_pred))  

#-------------------------------------------(PART D) ---- SCIKIT LEARN DECISION TREE----------------------------#
    for depth in range(1,6,2):
        print("SCIKIT DECISION TREE FOR DEPTH:",depth)
        dtrclf = DecisionTreeClassifier(criterion='entropy',max_depth=depth)
        model = dtrclf.fit(Xtrn,ytrn)
        y_pred_scikit = model.predict(Xtst)
        print("Confusion Matrix -> SCikit Learn")
        print(confusion_matrix(ytst,y_pred_scikit))
        
    
