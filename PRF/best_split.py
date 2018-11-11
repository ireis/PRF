import numpy
from numba import jit, jitclass

from .  import misc_functions as m

#from importlib import reload
#reload(m)

############################################################
############################################################
################ Find best split functions  ################
############################################################
############################################################


@jit(cache=True, nopython=True)
def _gini_init(py):

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
def get_best_split(X, py, current_score, features_chosen_indices, max_features):

    n_features = len(features_chosen_indices)
    nof_objects = py.shape[0]

    # Initialize values in case no best split is found
    best_gain = 0
    gain = current_score
    best_attribute = 0
    best_attribute_value = 0

    n_visited = 0
    found_split = False
    while (n_visited < max_features) or ((found_split == False) and (n_visited < n_features)):
        feature_index = features_chosen_indices[n_visited]
        n_visited = n_visited + 1
    #for feature_index in features_chosen_indices:

        # We first sort the feature values for the selected feature. This allows us to avoid spliting the sample in each
        # iteration. Instead we can just move one object
        feature_values = X[:,feature_index].copy()
        # skip objects with nan values
        nan_values = numpy.isnan(feature_values)
        nof_objects_skip = nan_values.sum()
        if nof_objects_skip == nof_objects:
            continue

        feature_values[nan_values] = numpy.nanmax(feature_values) + 1

        x_asort = numpy.argsort(feature_values)

        # We calculate the impurity when all the objects are on the right node.
        # in each iteration of loop over possible splits, we just update this value by moving one object to the other side of the
        # split (using the _gini_update function).
        impurity_right, normalization_right, class_p_right = _gini_init(py)
        impurity_left, normalization_left, class_p_left = 0, 0, 0*class_p_right

        nof_objects_itr = nof_objects - nof_objects_skip
        nof_objects_right = nof_objects_itr
        nof_objects_left = 0

        # In each iteration of this loop we move object by object from the left to the right side of the loop.
        for i in range(nof_objects_itr):

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
                    found_split = True

                    # Save the values of the best split so far
                    best_gain = gain
                    best_attribute = feature_index
                    best_attribute_value = feature_values[x_asort[i]]

    return  best_gain, best_attribute, best_attribute_value
