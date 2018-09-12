from math import log
import numpy
from numba import jit, jitclass
from joblib import Parallel, delayed
import threading
from scipy.stats import norm

cache = True

@jit(cache=True, nopython=True)
def numba_predict_all_objects(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, X, dX, keep_proba):

    nof_objects = X.shape[0]
    nof_classes = len(node_tree_results[0])
    summed_prediction_all = numpy.zeros((nof_objects, nof_classes))
    curr_node = 0
    for i in range(nof_objects):
        summed_prediction_all[i] = numba_predict(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, X[i], dX[i], curr_node, keep_proba, p_tree = 1.0, is_max = True)

    return summed_prediction_all

@jit(cache=True, nopython=True)
def numba_predict(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, x, dx, curr_node, keep_proba, p_tree = 1.0, is_max = True):
        """
        function classifies a single object accrding to the trained tree
        """

        node = curr_node
        tree_results = node_tree_results[curr_node]
        tree_feature_index = node_feature_idx[curr_node]
        tree_feature_th = node_feature_th[curr_node]
        true_branch_node = node_true_branch[curr_node]
        false_branch_node = node_false_branch[curr_node]
        #print(curr_node, node, tree_results, tree_feature_index, tree_feature_th, true_branch_node, false_branch_node)

        if (tree_results[0] >= 0):
            #print(summed_prediction, tree_results, p_tree, tree_results * p_tree)
            summed_prediction = tree_results * p_tree
        else:
            summed_prediction = numpy.zeros(2)
            if is_max:
                val = x[tree_feature_index]
                delta = dx[tree_feature_index]
                p_split =     split_probability(val, delta, tree_feature_th)

                is_max_true = True
                is_max_false = False
                if p_split <= 0.5:
                    is_max_true = False
                    is_max_false = True

                p_true = p_tree * p_split
                summed_prediction += numba_predict(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch,x, dx, true_branch_node, keep_proba, p_true, is_max_true)

                p_false = p_tree * (1 - p_split)
                summed_prediction += numba_predict(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch,x, dx, false_branch_node, keep_proba, p_false, is_max_false)
            else:
                is_max_true = False
                is_max_false = False
                val = x[tree_feature_index]
                delta = dx[tree_feature_index]
                p_split =     split_probability(val, delta, tree_feature_th)
                p_true = p_tree * p_split
                if p_true > keep_proba:
                    summed_prediction += numba_predict(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch,x, dx, true_branch_node, keep_proba, p_true, is_max_true)
                p_false = p_tree * (1 - p_split)
                if p_false > keep_proba:
                    summed_prediction += numba_predict(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch,x, dx, false_branch_node, keep_proba, p_false, is_max_false)


        return summed_prediction

############################################################
############################################################
############ Propogate Probabilities functions  ############
############################################################
############################################################

N_SIGMA = 1
X_GAUS = numpy.arange(-N_SIGMA,N_SIGMA,0.1)
#X_GAUS = numpy.append(X_GAUS, N_SIGMA)
GAUS = numpy.array(norm(0,1).cdf(X_GAUS))
GAUS = numpy.append(GAUS, 1)


@jit(cache=cache, nopython=True)
def split_probability(value, delta, threshold):
    """
    Calculate split probability for a single object
    """

    if delta > 0:
        normalized_threshold = (threshold - value)/delta
        if (normalized_threshold <= -N_SIGMA):
            split_proba = 0
        elif (normalized_threshold >= N_SIGMA):
            split_proba = 1
        else:
            x = numpy.searchsorted(a=X_GAUS, v=normalized_threshold,)
            #x = numpy.argmax(X_GAUS > normalized_threshold)
            split_proba = GAUS[x]
    else:
        if (threshold - value) == 0:
            split_proba = 0.5
        elif (threshold - value) > 0:
            split_proba = 1
        elif (threshold - value) < 0:
            split_proba = 0



    return 1-split_proba


@jit(cache=cache, nopython=True)
def split_probability_all(values, deltas, threshold):
    """
    Calculate split probabilities for all rows in values
    """

    nof_objcts = values.shape[0]
    ps = [split_probability(values[i], deltas[i], threshold) for i in range(nof_objcts)]
    ps = numpy.array(ps)

    return ps


@jit(cache=cache, nopython=True)
def return_class_probas(pnode, pY):
    """
    The leaf probabilities for each class
    """

    nof_objects = pY.shape[0]
    nof_classes = pY.shape[1]
    class_probas = numpy.zeros(nof_classes)

    for i in range(nof_objects):
        class_probas += pnode[i] * pY[i,:]

    #class_probas = class_probas/numpy.sum(pnode)
    class_probas = class_probas/len(pnode)
    #class_probas = pY

    return class_probas

############################################################
############################################################
############################ MISC  #########################
############################################################
############################################################

def get_split_objects_numpy(pnode, p_split_right, p_split_left, is_max, n_objects_node, keep_proba):
    pnode_right = pnode*p_split_right
    pnode_left  = pnode*p_split_left

    best_path_objects_right = numpy.where( (p_split_right >= 0.5)  & is_max)[0]
    is_max_right = numpy.zeros(n_objects_node, dtype = int)
    is_max_right[best_path_objects_right] = 1

    best_path_objects_left = numpy.where( (p_split_left >= 0.5)  & is_max)[0]
    is_max_left = numpy.zeros(n_objects_node, dtype = int)
    is_max_left[best_path_objects_left] = 1




    best_right = numpy.unique(numpy.concatenate([best_path_objects_right, numpy.where(pnode_right > keep_proba)[0]]))
    best_left  = numpy.unique(numpy.concatenate([best_path_objects_left,  numpy.where(pnode_left  > keep_proba)[0]]))

    return pnode_right, pnode_left, best_right, best_left, is_max_right, is_max_left


@jit(cache=True, nopython=True)
def get_split_objects(pnode, p_split_right, p_split_left, is_max, n_objects_node, keep_proba):
    pnode_right = pnode*p_split_right
    pnode_left  = pnode*p_split_left

    #best_path_objects_right = numpy.where( (p_split_right >= 0.5)  & is_max)[0]
    #is_max_right = numpy.zeros(n_objects_node)
    #is_max_right[best_path_objects_right] = 1

    #best_path_objects_left = numpy.where( (p_split_left >= 0.5)  & is_max)[0]
    #is_max_left = numpy.zeros(n_objects_node)
    #is_max_left[best_path_objects_left] = 1


    best_right = [0]
    best_left = [0]

    is_max_right = [0]
    is_max_left = [0]

    for i in range(n_objects_node):
        if (p_split_right[i] >= 0.5 and is_max[i] == 1):
            best_right.append(i)
            is_max_right.append(i)
        elif pnode_right[i] > keep_proba:
            best_right.append(i)

        if (p_split_left[i] >= 0.5 and is_max[i] == 1):
            best_left.append(i)
            is_max_left.append(i)
        elif pnode_left[i] > keep_proba:
            best_left.append(i)

    best_right = numpy.array(best_right)
    best_left = numpy.array(best_left)
    is_max_right = numpy.array(is_max_right)
    is_max_left = numpy.array(is_max_left)

    #best_right = numpy.unique(numpy.concatenate([best_path_objects_right, numpy.where(pnode_right > keep_proba)[0]]))
    #best_left  = numpy.unique(numpy.concatenate([best_path_objects_left,  numpy.where(pnode_left  > keep_proba)[0]]))

    return pnode_right, pnode_left, best_right[1:], best_left[1:], is_max_right[1:], is_max_left[1:]


#@jit(cache=True, nopython=True)
def choose_features_jit(nof_features, max_features):
    """
    function randomly selects the features that will be examined for each split
    """
    features_indices = numpy.arange(nof_features)
    #numpy.random.seed()
    features_chosen = numpy.random.choice(features_indices, size=max_features)
    return features_chosen

@jit(cache=True, nopython=True)
def pull_values(A, right, left):
    """
    Splits an array A to two
    according to lists of indicies
    given in right and left
    """
    A_left = A[left]
    A_right = A[right]

    return A_right, A_left

############################################################
############################################################
################ Find best split functions  ################
############################################################
############################################################


@jit(cache=True, nopython=True)
def _gini_init(py):
    """
    calculates the gini impurity
    given an array with probabilities for each object to be in
                                          each class

    gini impurity is
    sum over classes of
    x * (1 - x)
    when x is
    the fraction of objects in a class
    """
    nof_classes = py.shape[1]

    # initializations
    class_p_arr = py[0]*0
    impurity = 0

    # Normalization used to
    # calculate the distribution of
    # classes
    normalization = py.sum()

    # loop over classes
    for class_idx in range(nof_classes):
        # E of number of objects in the class
        py_class = py[:, class_idx]
        class_p = numpy.sum(py_class )
        class_p_arr[class_idx] = class_p

        # fraction of class size
        class_p = class_p / normalization

        # gini impurity
        impurity += class_p*(1-class_p)

    return impurity, normalization, class_p_arr

@jit(cache=True, nopython=True)
def _gini_update(normalization, class_p_arr, py):
    """
    this function is used to claculate
    the change in the gini impurity if we add or remove a single object

    the inputs normalization and class_p_arr,
    contain values from the previous iteration,
    that is before moving the object

    py contains the probabilities of one object to be in each class

    we use +py to add an object and -py to remove
    """
    nof_classes = len(py)

    # initialization
    impurity = 0

    # get the new normalization
    normalization = normalization + py.sum()

    for class_idx in range(nof_classes):
        # get the new E of number of objects in a class
        class_p_arr[class_idx] +=  py[class_idx]

        # fraction of class size
        class_p = class_p_arr[class_idx]/normalization

        # gini impurity
        impurity += class_p*(1-class_p)

    return impurity, normalization, class_p_arr


@jit(cache=True, nopython=True)
def get_best_split(X, py, y, current_score, features_chosen_indices):
    """
    Looks for the best possible split,
    impurity-wise,
    among all feature values,
    among some features selected at random,
    which are given in features_chosen_indices.

    current_score is the impurity of the parent node,
    it is used to calculate the gain,
    the gain needs to be bigger than zero,
    for highest gain split,
    or else we do not split the node
    """

    nof_objects = len(y)

    # Initialize values in case no new best gain is found
    best_gain = 0
    gain = current_score
    best_attribute = 0
    best_attribute_value = 0
    best_right = numpy.arange(1,nof_objects)
    best_left = numpy.arange(0,1)

    # Loop over the possible features
    for feature_index in features_chosen_indices:

        # We first sort the feature values for the selected feature. This allows us to avoid spliting the sample in each
        # iteration. Instead we can just move one object
        feature_values = X[:,feature_index]
        x_asort = numpy.argsort(feature_values)

        # We calculate the impurity when all the objects are on the right node.
        # in each iteration of loop over possible splits, we just update this value by moving one object to the other side of the
        # split (using the _gini_update function).
        impurity_right, normalization_right, class_p_right = _gini_init(py)
        impurity_left, normalization_left, class_p_left = 0, 0, 0*class_p_right

        nof_objects_right = nof_objects
        nof_objects_left = 0

        #print(10,  impurity_right, 0,  impurity_left)
        # In each iteration of this loop we move object by object from the left to the right side of the loop.
        for i in range(nof_objects):

            # Update the number of objects in each side of the split
            nof_objects_left += 1
            nof_objects_right -= 1

            # We only need to calculate the impurity if both sides are not empty (which is a useless split)
            if nof_objects_right>0 and nof_objects_left>0:

                # Update the impurities on both sides
                impurity_right, normalization_right, class_p_right  = _gini_update( normalization_right,  class_p_right, -py[x_asort[i]])
                impurity_left, normalization_left, class_p_left  = _gini_update( normalization_left, class_p_left, py[x_asort[i]])

                # Calculate the gain for the split
                normalization = normalization_right + normalization_left
                p_right = normalization_right / normalization
                p_left = normalization_left / normalization
                gain = current_score - p_right*impurity_right - p_left*impurity_left

                # Check if this is a better gain that the current best gain
                #print(nof_objects_right,  impurity_right, nof_objects_left,  impurity_left)
                if gain > best_gain:

                    # Get the indicies of the object that go left, and of the objects that go right
                    right,left = x_asort[i+1:], x_asort[:i+1]

                    # Save the values of the best split so far
                    best_gain = gain
                    best_attribute = feature_index
                    best_attribute_value = feature_values[x_asort[i]]
                    best_left = left
                    best_right = right

    return gain, best_gain, best_attribute, best_attribute_value, best_left, best_right

############################################################
############################################################
################ DecisionTreeClassifier Class  #############
############################################################
############################################################

class Tree:
    """
    This is the recursive binary tree implementation.
    """
    def __init__(self, feature_index=-1, feature_threshold=None, true_branch=None, false_branch=None, results=None):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.results = results # None for nodes, not None for leaves # TODO: decide what you want to do with this.

    def get_node_list(self, node_list, this_node, node_idx):

        node_idx_right = node_idx + 1
        last_node_left_branch = node_idx
        if type(this_node.true_branch) != type(None):
            last_node_right_branch, node_list = self.get_node_list(node_list, this_node.true_branch, node_idx_right)
            node_idx_left = last_node_right_branch + 1
            last_node_left_branch, node_list = self.get_node_list(node_list, this_node.false_branch, node_idx_left)


        if type(this_node.results) != type(None):
            node_list.append([node_idx, this_node.feature_index, this_node.feature_threshold, None,None, this_node.results])
        else:
            node_list.append([node_idx, this_node.feature_index, this_node.feature_threshold, node_idx_right,node_idx_left, None])

        return last_node_left_branch, node_list



class DecisionTreeClassifier:
    """
    This the the decision tree classifier class.
    """
    def __init__(self, criterion='gini', max_features=None, use_py_gini = True, use_py_leafs = True, max_depth = None, keep_proba = 0.05):
        self.criterion = criterion
        self.max_features = max_features
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba


    def _gini_original(self, py, y):
        """
        function computes the gini impurity of the given sample of objects
        """

        nof_classes = py.shape[1]
        nof_objects = py.shape[0]
        impurity = 0
        for c in [0,1]:
            c_num = numpy.sum(y == c)
            p = c_num/nof_objects
            impurity += p*(1-p)

        return impurity

    def _growDecisionTreeFrom(self, X, y, dX, py_gini, py_leafs, pnode, depth, is_max):
        """
        function grows a recursive disicion tree according to the objects X and their classifications y
        """
        #if self.criterion == "gini":
        #    evaluationFunction = self._gini_original
        #elif self.criterion == "gini_orig":
        #    evaluationFunction = self._gini_original
        #elif self.criterion == "entropy":
        #    evaluationFunction = self._gini_original
        #else:
        #    return Tree()

        if len(X) == 0: return Tree()

        n_objects_node = X.shape[0]

        max_depth = depth + 1
        if self.max_depth:
            max_depth = self.max_depth

        if depth < max_depth:
            features_chosen_indices = choose_features_jit(self.n_features_, self.max_features)

            #scaled_py_gini = numpy.array([py_gini[i] * pnode[i] for i in range(len(X))])
            scaled_py_gini = numpy.array([py_gini[:,0] * pnode[:] , py_gini[:,1] * pnode[:]]).T

            current_score, normalization, class_p_arr = _gini_init(scaled_py_gini)

            gain, best_gain, best_attribute, best_attribute_value, best_left, best_right = get_best_split(X, scaled_py_gini,
                                                                                                          y, current_score,
                                                                                                      features_chosen_indices)
        else:
            best_gain = 0


        # Caclculate split probabilities for each object
        p_split_right = split_probability_all(X[:,best_attribute], dX[:,best_attribute], best_attribute_value)
        p_split_left = 1 - p_split_right

        pnode_right, pnode_left, best_right, best_left, is_max_right, is_max_left = get_split_objects(pnode, p_split_right, p_split_left, is_max, n_objects_node, self.keep_proba)

        pnode_right, _ = pull_values(pnode_right, best_right, best_left)
        _, pnode_left  = pull_values(pnode_left,  best_right, best_left)
        is_max_right, _ = pull_values(is_max_right, best_right, best_left)
        _, is_max_left  = pull_values(is_max_left,  best_right, best_left)

        # Check if the best split is valid (that is not a useless 0-everything split). If yes continue growing tree, if no we
        # have a leaf
        if (best_gain > 0) and (numpy.sum(pnode_right) >= 1) and  (numpy.sum(pnode_left) >= 1):

            # add the impurity of the best split into the feature importance value
            p = len(y) / self.n_samples_
            self.feature_importances_[best_attribute] += p * best_gain

            # Split all the arrays according to the indicies we have for the object in each side of the split
            X_right, X_left = pull_values(X, best_right, best_left)
            y_right, y_left = pull_values(y, best_right, best_left)
            dX_right, dX_left = pull_values(dX, best_right, best_left)
            py_right, py_left = pull_values(py_gini, best_right, best_left)
            py_leafs_right, py_leafs_left = pull_values(py_leafs, best_right, best_left)


            # go to the next steps of the recursive process
            depth = depth + 1
            right_branch = self._growDecisionTreeFrom(X_right, y_right, dX_right, py_right, py_leafs_right, pnode_right, depth, is_max_right)
            left_branch  = self._growDecisionTreeFrom(X_left,  y_left,  dX_left,  py_left,  py_leafs_left , pnode_left, depth, is_max_left)

            return Tree(feature_index=best_attribute, feature_threshold=best_attribute_value, true_branch=right_branch, false_branch=left_branch)
        else:
            class_probas = return_class_probas(pnode, py_leafs)

            return Tree(results= class_probas)#Tree(results=self._uniqueCounts(py))

    def _classifyWithoutMissingData(self, x, tree):
        """
        function classifies a single object accrding to the trained tree
        """
        if type(tree.results) != type(None):  # leaf
            return tree.results
        else:
            v = x[tree.feature_index]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v > tree.feature_threshold: branch = tree.true_branch
                else: branch = tree.false_branch
            else:
                if v == tree.feature_value: branch = tree.true_branch
                else: branch = tree.false_branch
        return self._classifyWithoutMissingData(x, branch)

    def _PclassifyWithoutMissingData(self, x, dx, tree, p_tree = 1):
        """
        function classifies a single object accrding to the trained tree
        """
        if type(tree.results) != type(None):  # leaf
            return tree.results, p_tree
        else:
            val = x[tree.feature_index]
            delta = dx[tree.feature_index]
            branch = None
            if val > tree.feature_threshold:
                p_split =     split_probability(val, delta, tree.feature_threshold)
                p_tree *= p_split
                branch = tree.true_branch
            else:
                p_split = 1 - split_probability(val, delta, tree.feature_threshold)
                p_tree *= p_split
                branch = tree.false_branch
        return self._PclassifyWithoutMissingData(x, dx, branch, p_tree)

    def _predict_all_leafs(self, x, dx, tree, p_tree = 1):
        """
        function classifies a single object accrding to the trained tree
        """
        summed_prediction = numpy.zeros(self.n_classes_)
        if type(tree.results) != type(None):
            return tree.results * p_tree
        else:
            val = x[tree.feature_index]
            delta = dx[tree.feature_index]
            p_split =     split_probability(val, delta, tree.feature_threshold)
            p_true = p_tree * p_split
            summed_prediction += self._predict_all_leafs(x, dx, tree.true_branch, p_true)

            p_false = p_tree * (1 - p_split)
            summed_prediction += self._predict_all_leafs(x, dx, tree.false_branch, p_false)

        return summed_prediction

    def __predict_all_leafs_trimmed(self, x, dx, tree, p_tree = 1, is_max = True):
        """
        function classifies a single object accrding to the trained tree
        """
        summed_prediction = numpy.zeros(self.n_classes_)
        if type(tree.results) != type(None):
            return tree.results * p_tree
        else:
            if is_max:

                val = x[tree.feature_index]
                delta = dx[tree.feature_index]
                p_split =     split_probability(val, delta, tree.feature_threshold)

                is_max_true = True
                is_max_false = False
                if p_split <= 0.5:
                    is_max_true = False
                    is_max_false = True

                p_true = p_tree * p_split
                summed_prediction += self._predict_all_leafs_trimmed(x, dx, tree.true_branch, p_true, is_max_true)

                p_false = p_tree * (1 - p_split)
                summed_prediction += self._predict_all_leafs_trimmed(x, dx, tree.false_branch, p_false, is_max_false)
            else:
                is_max_true = False
                is_max_false = False
                val = x[tree.feature_index]
                delta = dx[tree.feature_index]
                p_split =     split_probability(val, delta, tree.feature_threshold)
                p_true = p_tree * p_split
                if p_true > self.keep_proba:
                    summed_prediction += self._predict_all_leafs_trimmed(x, dx, tree.true_branch, p_true, is_max_true)
                p_false = p_tree * (1 - p_split)
                if p_false > self.keep_proba:
                    summed_prediction += self._predict_all_leafs_trimmed(x, dx, tree.false_branch, p_false, is_max_false)


        return summed_prediction

    def _predict_all_leafs_trimmed(self, x, dx, p_tree = 1, is_max = True):
        """
        function classifies a single object accrding to the trained tree
        """
        keep_proba = self.keep_proba
        curr_node = 0

        summed_prediction = numba_predict(self.node_tree_results, self.node_feature_idx, self.node_feature_th, self.node_true_branch, self.node_false_branch, x, dx, curr_node, keep_proba, p_tree = 1, is_max = True)

        return summed_prediction

    def get_nodes(self):


        node_list = []
        node_list = self.tree_.get_node_list(node_list, self.tree_, 0)[1]
        node_idx = numpy.zeros(len(node_list), dtype = int)
        for i,node in enumerate(node_list):
            node_idx[i] = node[0]
            node_list_sort = []
        new_order = numpy.argsort(node_idx)
        for new_idx, idx in enumerate(new_order):
            node_list_sort += [node_list[idx]]


        return node_list_sort

    def node_arr_init(self):

        node_list = self.get_nodes()

        node_tree_results = numpy.ones([len(node_list),self.n_classes_] )*(-1)
        node_feature_idx = numpy.ones(len(node_list), dtype = int)*(-1)
        node_feature_th = numpy.zeros(len(node_list))
        node_true_branch = numpy.ones(len(node_list), dtype = int)*(-1)
        node_false_branch = numpy.ones(len(node_list), dtype = int)*(-1)

        for idx in range(len(node_list)):

            n = node_list[idx]
            if not n[3] is None:
                node_feature_idx[idx] = n[1]
                node_feature_th[idx] = n[2]
                node_true_branch[idx] = n[3]
                node_false_branch[idx] = n[4]

            else:
                node_tree_results[idx] = n[5]

        self.node_feature_idx = node_feature_idx
        self.node_feature_th = node_feature_th
        self.node_true_branch = node_true_branch
        self.node_false_branch = node_false_branch
        self.node_tree_results = node_tree_results

        return

    def fit(self, X, y, pX, py):
        """
        the DecisionTreeClassifier.fit() function with a similar appearance to that of sklearn
        """
        self.n_classes_ = len(numpy.unique(y))
        self.classes_ = numpy.sort(numpy.unique(y))
        self.n_features_ = len(X[0])
        self.n_samples_ = len(X)
        self.feature_importances_ = [0] * self.n_features_

        pnode = numpy.ones(self.n_samples_)
        is_max = numpy.ones(self.n_samples_, dtype = int)

        py_flat = py.copy()
        py_flat[py < 0.5] = 0
        py_flat[py > 0.5] = 1

        if self.use_py_gini:
            py_gini = py
        else:
            py_gini = py_flat

        if self.use_py_leafs:
            py_leafs = py
        else:
            py_leafs = py_flat
        depth = 0
        self.tree_ = self._growDecisionTreeFrom(X, y, pX, py_gini, py_leafs, pnode, depth, is_max)

    def predict(X):
        """
        The DecisionTreeClassifier.predict() function with a similar appearance to that of sklearn
        """
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            result = self._classifyWithoutMissingData(x, self.tree_)
            best_class_index = numpy.argmax(result)
            y_pred.append(best_class_index)
        return y_pred # TODO: the output is a list and not a numpy.array(), this should be changed in the future

    def predict_proba(self, X, dX):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        """
        keep_proba = self.keep_proba

        summed_prediction = numba_predict_all_objects(self.node_tree_results, self.node_feature_idx, self.node_feature_th, self.node_true_branch, self.node_false_branch, X, dX, keep_proba)

        return summed_prediction

    def predict_log_proba(self, X):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        """
        return numpy.log10(self.predict_proba(X))

    def predict_proba_one_object(self, x):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        the function works for a single object
        """
        proba = self._classifyWithoutMissingData(x, self.tree_)
        return proba


    def ppredict_proba_one_object(self, x, dx):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        the function works for a single object
        """
        #proba_leaf, proba_object_in_leaf = self._PclassifyWithoutMissingData(x, dx, self.tree_)
        #return proba_leaf*proba_object_in_leaf
        return self._predict_all_leafs_trimmed(x, dx)


############################################################
############################################################
################ RandomForestClassifier Class  #############
############################################################
############################################################

class RandomForestClassifier:
    def __init__(self, n_estimators=10, criterion='gini', max_features='auto', use_py_gini = True, use_py_leafs = True, max_depth = None, keep_proba = 0.05):
        self.n_estimators_ = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.estimators_ = []
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba

    def _choose_objects(self, X, y, pX, py):
        """
        function builds a sample of the same size as the input data, but chooses the objects with replacement
        according to the given probability matrix
        """
        objects_indices = numpy.arange(len(y))
        objects_chosen = numpy.random.choice(objects_indices, size=len(y), replace=True)

        X_chosen = X[objects_chosen, :]
        y_chosen = y[objects_chosen]
        pX_chosen = pX[objects_chosen, :]
        py_chosen = py[objects_chosen, :]

        return X_chosen, y_chosen, pX_chosen, py_chosen



    def _fit_single_tree(self, X, y, pX, py):

        X_chosen, y_chosen, pX_chosen, py_chosen = self._choose_objects(X, y, pX, py)
        tree = DecisionTreeClassifier(criterion=self.criterion,
                                      max_features=self.max_features_num,
                                      use_py_gini = self.use_py_gini,
                                      use_py_leafs = self.use_py_leafs,
                                      max_depth = self.max_depth,
                                      keep_proba = self.keep_proba)


        tree.fit(X_chosen, y_chosen, pX_chosen, py_chosen)

        return tree


    def fit(self, X, y, dX, py):
        """
        The RandomForestClassifier.fit() function with a similar appearance to that of sklearn
        """
        self.n_classes_ = len(numpy.unique(y))
        self.classes_ = list(numpy.sort(numpy.unique(y)))
        self.n_features_ = len(X[0])
        self.feature_importances_ = numpy.zeros(self.n_features_)

        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features_num = int(numpy.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            self.max_features_num = int(numpy.log2(self.n_features_))
        elif type(self.max_features) == int:
            self.max_features_num = self.max_features
        elif type(self.max_features) == float:
            self.max_features_num = int(self.max_features * self.n_features_)
        else:
            self.max_features_num = self.n_features_


        tree_list = [self._fit_single_tree(X, y, dX, py) for i in range(self.n_estimators_)]
        #tree_list = Parallel(n_jobs=-1, verbose = 10)(delayed(self._fit_single_tree)
        #                                          (X, y, dX, py)                   for i in range(self.n_estimators_))

        for tree in tree_list:
            self.estimators_.append(tree)
            self.feature_importances_ += numpy.array(tree.feature_importances_)

        self.feature_importances_ /= self.n_estimators_



    def predict_single_object(self, row):

        # let all the trees vote
        summed_probabilites = numpy.zeros(self.n_classes_)

        for tree_index in range(0, self.n_estimators_):
            y_proba_tree = self.estimators_[tree_index].predict_proba_one_object(row)
            summed_probabilites += y_proba_tree


        return numpy.argmax(summed_probabilites)


    def pick_best(self, class_proba):


        new_class_proba = numpy.zeros(class_proba.shape)
        votes = numpy.zeros(class_proba.shape, dtype=numpy.float64)

        best_idx = numpy.argmax(class_proba)
        new_class_proba[best_idx] = class_proba[best_idx]
        votes[best_idx] = 1
        #print(class_proba)

        return new_class_proba

    def predict_single_tree(self, predict, X, dX, out, lock):

        # let all the trees vote - the function pick_best find the best class from each tree, and gives zero probability to all the others
        #prediction = numpy.vstack([self.pick_best(predict(x, dx)) for x, dx in zip(X,dX)])

        #prediction = numpy.vstack([predict(x, dx) for x, dx in zip(X,dX)])
        prediction = predict(X,dX)

        #print(prediction[:10])

        with lock:
            if len(out) == 1:
                out[0] += prediction
                #out2[0] += prediction[1]
            else:
                for i in range(len(out)):
                    out[i] += prediction[i]
                    #out2[i] += prediction[i][1]

    def predict(self, X, dX):
        """
        The RandomForestClassifier.predict() function with a similar appearance to that of sklearn
        """
        #vote_count = numpy.zeros((X.shape[0], self.n_classes_), dtype=numpy.float64)
        all_proba = numpy.zeros((X.shape[0], self.n_classes_), dtype=numpy.float64)
        lock = threading.Lock()
        #Parallel(n_jobs=-1, verbose = 10, backend="threading")(delayed(self.predict_single_tree)
        #                       (tree.ppredict_proba_one_object, X, dX, all_proba,  lock) for tree in self.estimators_)

        for i, tree in enumerate(self.estimators_):
            tree.node_arr_init()
            self.predict_single_tree(tree.predict_proba, X, dX, all_proba,  lock)

        all_proba /= self.n_estimators_
        y_pred = numpy.argmax(all_proba, axis = 1)

        return y_pred, all_proba




