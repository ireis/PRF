import numpy
from numba import jit, jitclass
from scipy.stats import norm

cache = True


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
        if (threshold - value) >= 0:
        #    split_proba = 0.5
        #elif (threshold - value) > 0:
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
            is_max_right.append(1)
        elif pnode_right[i] > keep_proba:
            best_right.append(i)
            is_max_right.append(0)

        if (p_split_left[i] > 0.5 and is_max[i] == 1):
            best_left.append(i)
            is_max_left.append(1)
        elif pnode_left[i] > keep_proba:
            best_left.append(i)
            is_max_left.append(0)

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
    #features_chosen = numpy.random.choice(features_indices, size=max_features, replace = True)
    features_chosen = numpy.random.choice(features_indices, size=nof_features, replace = False)
    
    #print(features_chosen)
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

def get_pY(pY_true, y_fake):
    """
    Recieves a vector with the probability to be true (pY_true)
    returns a matrix with the probability to be in each class
    
    we put pY_true as the probability of the true class
    and (1-pY_true)/(nof_lables-1) for all other classes
    """
    nof_objects = len(pY_true)
    
    all_labels = numpy.unique(y_fake)
    label_dict = {i:a for i,a in enumerate(all_labels) }
    nof_labels = len(all_labels)
    
    pY = numpy.zeros([nof_objects, nof_labels])
    
    for o in range(nof_objects):
        for c_idx, c in enumerate(all_labels):
            if y_fake[o] == c:
                pY[o,c_idx] = pY_true[o]
            else:
                pY[o,c_idx] = float(1 - pY_true[o])/(nof_labels - 1)
                    
    return pY, label_dict

