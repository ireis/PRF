import numpy
from numba import jit

from . import best_split
from . import misc_functions as m

#from importlib import reload
#reload(m)
#reload(best_split)

cache = False

class _tree:
    """
    This is the recursive binary tree implementation.
    """
    def __init__(self, feature_index=-1, feature_threshold=None, true_branch=None, false_branch=None, p_right=None, results=None):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.p_right = p_right
        self.results = results # None for nodes, not None for leaves # TODO: decide what you want to do with this.

    def get_node_list(self, node_list, this_node, node_idx):

        node_idx_right = node_idx + 1
        last_node_left_branch = node_idx
        if type(this_node.true_branch) != type(None):
            last_node_right_branch, node_list = self.get_node_list(node_list, this_node.true_branch, node_idx_right)
            node_idx_left = last_node_right_branch + 1
            last_node_left_branch, node_list = self.get_node_list(node_list, this_node.false_branch, node_idx_left)


        if type(this_node.results) != type(None):
            node_list.append([node_idx, this_node.feature_index, this_node.feature_threshold, None,None, this_node.results, None])
        else:
            node_list.append([node_idx, this_node.feature_index, this_node.feature_threshold, node_idx_right, node_idx_left, None, this_node.p_right])

        return last_node_left_branch, node_list

############################################################
############################################################
############################################################
############################################################
############           UNSUPERVISED             ############
############################################################
############################################################
############################################################
############################################################



#@jit(cache=True, nopython=True)
def default_synthetic_data(X):
    """
    Synthetic data with same marginal distribution for each feature
    """
    synthetic_X = numpy.zeros(X.shape)

    nof_features = X.shape[1]
    nof_objects = X.shape[0]

    for f in range(nof_features):
        feature_values = X[:, f]
        synthetic_X[:, f] += numpy.random.choice(feature_values, nof_objects)
    return synthetic_X

#@jit(cache=True, nopython=True)
def get_synthetic_data(X, dX, py, py_remove, pnode, is_max):

    #if (len(numpy.unique(y)) == 1):
    #    y= numpy.zeros(len(y), dtype = int)

    real_inds = numpy.where(py[:,1] == 0)[0]
    X_real = X[real_inds]
    dX_real = dX[real_inds]
    py_real = py[real_inds]
    pnode_real = pnode[real_inds]
    is_max_real = is_max[real_inds]
    n_real = X_real.shape[0]
    if n_real < 50:
        return X, dX, py, py_remove, pnode, is_max

    X_syn  = default_synthetic_data(X_real)
    dX_syn = default_synthetic_data(dX_real)

    X_new = numpy.vstack([X_real,X_syn])
    dX_new = numpy.vstack([dX_real,dX_syn])

    py_new = numpy.zeros([X_new.shape[0],2])
    py_new[:n_real,0] = py_real[:,0] #Class 'real'
    py_new[n_real:,1] = py_real[:,0]

    pnode_new = numpy.zeros([X_new.shape[0]])
    pnode_new[:n_real] = pnode_real
    pnode_new[n_real:] = pnode_real

    is_max_new = numpy.concatenate([is_max_real, is_max_real])


    return X_new, dX_new, py_new, py_new, pnode_new, is_max_new




############################################################
############################################################
############################################################
############################################################
############               TRAIN                ############
############################################################
############################################################
############################################################
############################################################



def fit_tree(X, dX, py_gini, py_leafs, pnode, depth, is_max, tree_max_depth, max_features, feature_importances,
             tree_n_samples, keep_proba, unsupervised=False, new_syn_data_frac=0, min_py_sum_leaf=1):
    """
    function grows a recursive disicion tree according to the objects X and their classifications y
    """

    if len(X) == 0:
        print('Warning: empty node')
        return _tree()

    n_features = X.shape[1]
    n_objects_node = X.shape[0]

    if unsupervised:
        new_syn_data = False
        if depth == 0:
            new_syn_data = True
        elif (n_objects_node > 50):
            if (numpy.random.rand() < new_syn_data_frac):
                new_syn_data = True

        if new_syn_data:
            #print('before:', X.shape, dX.shape, py_gini.shape, py_leafs.shape, pnode.shape, is_max.shape)
            X, dX, py_gini, py_leafs, pnode, is_max = get_synthetic_data(X, dX, py_gini, py_leafs, pnode, is_max)
            #print('after:', X.shape, dX.shape, py_gini.shape, py_leafs.shape, pnode.shape, is_max.shape)
            n_objects_node = X.shape[0]



    max_depth = depth + 1
    if tree_max_depth:
        max_depth = tree_max_depth

    if depth < max_depth:
        scaled_py_gini = numpy.multiply(py_gini, pnode[:,numpy.newaxis])

        current_score, normalization, class_p_arr = best_split._gini_init(scaled_py_gini)
        features_chosen_indices = m.choose_features(n_features, max_features)
        best_gain, best_attribute, best_attribute_value = best_split.get_best_split(X, scaled_py_gini,  current_score, features_chosen_indices, max_features)

        # Caclculate split probabilities for each object
        if best_gain > 0:
            p_split_right = m.split_probability_all(X[:,best_attribute], dX[:,best_attribute], best_attribute_value)
            p_split_left = 1 - p_split_right
            pnode_right, pnode_left, best_right, best_left, is_max_right, is_max_left, pnode_right_tot = m.get_split_objects(pnode, p_split_right, p_split_left, is_max, n_objects_node, keep_proba)

            # Check if the best split is valid (that is not a useless 0-everything split)
            th = min_py_sum_leaf
            if (numpy.sum(pnode_right) >= th) and (numpy.sum(pnode_left) >= th):
                # add the impurity of the best split into the feature importance value
                p = scaled_py_gini.sum() / tree_n_samples
                feature_importances[best_attribute] += p * best_gain

                # Split all the arrays according to the indicies we have for the object in each side of the split
                X_right, X_left = m.pull_values(X, best_right, best_left)
                dX_right, dX_left = m.pull_values(dX, best_right, best_left)
                py_right, py_left = m.pull_values(py_gini, best_right, best_left)
                py_leafs_right, py_leafs_left = m.pull_values(py_leafs, best_right, best_left)

                # go to the next steps of the recursive process
                depth = depth + 1
                right_branch = fit_tree(X_right, dX_right, py_right, py_leafs_right, pnode_right, depth, is_max_right, tree_max_depth, max_features, feature_importances, tree_n_samples, keep_proba, unsupervised, new_syn_data_frac, min_py_sum_leaf)
                left_branch  = fit_tree(X_left,  dX_left,  py_left,  py_leafs_left , pnode_left, depth, is_max_left, tree_max_depth, max_features, feature_importances, tree_n_samples, keep_proba, unsupervised, new_syn_data_frac, min_py_sum_leaf)

                return _tree(feature_index=best_attribute, feature_threshold=best_attribute_value, true_branch=right_branch, false_branch=left_branch, p_right=pnode_right_tot)



    class_probas = m.return_class_probas(pnode, py_leafs)
    #if len(pnode) > 2500:
    #    print(len(pnode), best_gain, len(pnode_right), len(pnode_left), numpy.mean(p_split_right), numpy.mean(dX[:,best_attribute]), numpy.nanmin(X[:,best_attribute]), numpy.nanmax(X[:,best_attribute]), best_attribute_value )
    return _tree(results= class_probas)



############################################################
############################################################
############################################################
############################################################
############               PREDICT              ############
############################################################
############################################################
############################################################
############################################################




@jit(cache=cache, nopython=True)
def predict_all(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, X, dX, keep_proba, return_leafs):

    nof_objects = X.shape[0]
    nof_classes = len(node_tree_results[0])
    result = numpy.zeros((nof_objects, nof_classes))
    curr_node = 0
    for i in range(nof_objects):
        result[i] = predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, X[i], dX[i], curr_node, keep_proba, p_tree = 1.0, is_max = True, return_leafs=return_leafs)
    return result

@jit(cache=cache, nopython=True)
def predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, x, dx, curr_node, keep_proba, p_tree = 1.0, is_max = True, return_leafs=False):
        """
        function classifies a single object according to the trained tree
        """
        node = curr_node
        tree_results = node_tree_results[curr_node]
        tree_feature_index = node_feature_idx[curr_node]
        tree_feature_th = node_feature_th[curr_node]
        true_branch_node = node_true_branch[curr_node]
        false_branch_node = node_false_branch[curr_node]
        p_right_node = node_p_right[curr_node]

        nof_classes = len(tree_results)

        if (tree_results[0] >= 0):
            if return_leafs:
                summed_prediction = tree_results*0 + node
            else:
                summed_prediction = tree_results * p_tree
        else:
            summed_prediction = numpy.zeros(nof_classes)
            if is_max:
                val = x[tree_feature_index]
                delta = dx[tree_feature_index]
                p_split =     m.split_probability(val, delta, tree_feature_th)
                if numpy.isnan(p_split):
                    p_split = p_right_node

                p_true = p_tree * p_split
                p_false = p_tree * (1 - p_split)

                is_max_true = True
                is_max_false = False
                if p_split <= 0.5:
                    is_max_true = False
                    is_max_false = True

                if ((p_true > keep_proba) or is_max_true):
                    summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, x, dx, true_branch_node, keep_proba, p_true, is_max_true, return_leafs)

                if ((p_false > keep_proba) or is_max_false):
                    summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, x, dx, false_branch_node, keep_proba, p_false, is_max_false, return_leafs)

            else:
                is_max_true = False
                is_max_false = False
                val = x[tree_feature_index]
                delta = dx[tree_feature_index]
                p_split =     m.split_probability(val, delta, tree_feature_th)

                if numpy.isnan(p_split):
                    p_split = p_right_node

                p_true = p_tree * p_split
                p_false = p_tree * (1 - p_split)

                if p_true > keep_proba:
                    summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, x, dx, true_branch_node, keep_proba, p_true, is_max_true, return_leafs)

                if p_false > keep_proba:
                    summed_prediction += predict_single(node_tree_results, node_feature_idx, node_feature_th, node_true_branch, node_false_branch, node_p_right, x, dx, false_branch_node, keep_proba, p_false, is_max_false, return_leafs)

        return summed_prediction
