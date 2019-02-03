import numpy
from joblib import Parallel, delayed
from numba import jit
from collections import Counter

def is_good_vec(tree, X):
    """
    """
    is_good = (tree.predict_proba(X, numpy.zeros(X.shape))[:, 0] > 0.5)
    return  is_good


def is_good_matrix_get(forest, X):
    #is_good_matrix = Parallel(n_jobs=-1, verbose=0)(delayed(is_good_vec)
    #                                                (tree, X) for tree in forest.estimators_)
    is_good_matrix = [is_good_vec(tree,X) for tree in forest.estimators_]
    is_good_matrix = numpy.vstack(is_good_matrix)
    is_good_matrix = is_good_matrix.T

    return is_good_matrix



def run_random_forest(X, n_trees, nof_objects, **kwargs):
    """
    Runs random forest on X
    """
    X_total, Y_total = create_synthetic_data(X, nof_objects, **kwargs)

    max_features = kwargs.get('max_features', 'auto')
    max_depth = kwargs.get('max_depth', None)

    rforest = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, max_features = max_features,  max_depth = max_depth)
    rforest.fit(X_total, Y_total)

    return rforest




def rf_distance_matrix(X, n_trees, **kwargs):

    try:
        objnum = X.shape[0]
        Xs = X
    except:

        Xs = X.copy()
        X = numpy.hstack(Xs)
        objnum = X.shape[0]

    csize = 10
    start = numpy.arange(1 + int(objnum / csize)) * csize
    end = start + csize
    fe = numpy.vstack([start, end]).T
    fe.shape
    fe[-1][1] = objnum

    only_nn = kwargs.get('only_nearest_neighbors', False)

    rf = run_random_forest(X=Xs, n_trees=n_trees, nof_objects=objnum, **kwargs)
    rf_leafs = rf.apply(X)

    return rf_leafs

    if only_nn:
        nof_nn = kwargs.get('n_nearest_neighbors', 5)
        nn, nn_hash_counts = get_hash_nn(rf_leafs, nof_nn)
        return nn, nn_hash_counts
    else:
        is_good = is_good_matrix_get(rf, X)

        distance_matrix = Parallel(n_jobs=-1)(delayed(build_distance_matrix_slow)
                                              (rf_leafs, is_good, se)          for se in fe)
        distance_matrix = numpy.vstack(distance_matrix)

        distance_matrix = distance_mat_fill(distance_matrix)

        return distance_matrix

############################################################
############################################################
############################################################
############################################################
##########       FULL DISTANCE MATRIX             ##########
############################################################
############################################################
############################################################
############################################################



@jit
def build_distance_matrix_slow(leafs, is_good, fe):

    start = fe[0]
    end = fe[1]

    obs_num = leafs.shape[0]
    tree_num = leafs.shape[1]
    dis_mat = numpy.ones((end - start,obs_num),dtype = 'f2')

    for i in range(start,end):
        jstart = i
        for j in range(jstart, obs_num):
            same_leaf = 0
            good_trees = 0
            for k in range(tree_num):
                if (is_good[i,k]  == 1 ) and (is_good[j,k] == 1):
                    good_trees = good_trees + 1
                    if (leafs[i,k] == leafs[j,k]):
                        same_leaf = same_leaf + 1
            if good_trees == 0:
                dis = 1
            else:
                dis = 1 - float(same_leaf) / good_trees

            dis_mat[i - start][j] = dis

    return dis_mat

@jit
def distance_mat_fill(dis_mat):


    for i in range(len(dis_mat)):
        jend = i
        for j in range(0,jend):

            dis_mat[i][j] = dis_mat[j][i]

    return dis_mat


############################################################
############################################################
############################################################
############################################################
##########          NEAREST NEAIGHBORS            ##########
############################################################
############################################################
############################################################
############################################################





#@jit(cache=True, nopython=True)
def get_hash_tbls(leafs,is_good_matrix):

    nof_objects = leafs.shape[0]
    nof_leafs = leafs.shape[1]
    hash_tabels = []
    for leafs_idx in range(nof_leafs):
        leafs_ = leafs[:,leafs_idx]
        is_good_ = is_good_matrix[:,leafs_idx]
        unique_leafs = numpy.unique(leafs_)
        h = {}
        for l in unique_leafs:
            leaf_objects = list(numpy.where( (leafs_ == l) & (is_good_ == 1) )[0])
            #if len(leaf_objects) < 1000:
            h[l] = leaf_objects
        hash_tabels += [h]

    return hash_tabels




#@jit(cache=True)
def get_nn_single_object(leafs,hash_tabels,nof_nn):

    neighbors = []
    for i,h in enumerate(hash_tabels):
        l = leafs[i]
        neighbors += h[l]

    counts = Counter(neighbors)
    objects_ = numpy.array(list(counts.keys()))
    counts_ = numpy.array(list(counts.values()))

    nn_inds = numpy.argsort(counts_)[::-1][:nof_nn]

    objects = objects_[nn_inds]
    counts = counts_[nn_inds]

    return objects, counts

@jit(cache=True, nopython=True)
def count_nn(h_list,nof_nn, nof_hits):

    for h in h_list:
        h_arr = numpy.array(h)
        nof_hits[h_arr] = nof_hits[h_arr] + 1

    nn_inds = numpy.argsort(nof_hits)[::-1][:nof_nn]

    objects = nn_inds
    counts = nof_hits[nn_inds]

    return objects, counts


def get_nn_single_object_jit(leafs,hash_tabels,nof_nn,nof_objects):

    h_list = []
    for i,h in enumerate(hash_tabels):
        l = leafs[i]
        h_list += [h[l]]

    return count_nn(h_list, nof_nn, numpy.zeros(nof_objects))



def get_nn_batch(leafs,hash_tabels,nof_nn, se):

    nn_arr = numpy.zeros([se[1] - se[0], nof_nn], dtype = int)
    counts_arr = numpy.zeros([se[1] - se[0], nof_nn], dtype = int)
    for o in range(se[0],se[1]):
        objects, counts = get_nn_single_object(leafs[o],hash_tabels,nof_nn)
        nn_arr[o-se[0]] = objects
        counts_arr[o-se[0]] = counts

    return numpy.hstack([nn_arr, counts_arr])

def get_nn(leafs, hash_tabels, nof_nn):


    nof_objects = leafs.shape[0]
    nof_leafs = leafs.shape[1]


    if False:
        csize = 2500
        start = numpy.arange(1 + int(nof_objects / csize)) * csize
        end = start + csize
        fe = numpy.vstack([start, end]).T
        fe.shape
        fe[-1][1] = nof_objects

        res = Parallel(n_jobs=-1, verbose = 10)(delayed(get_nn_batch)
                                                  (leafs, hash_tabels, nof_nn, se) for se in fe)

        res = numpy.vstack(res)
        nn_arr, counts_arr = numpy.split(res,2,axis=1)

    else:
        nn_arr = numpy.zeros([nof_objects, nof_nn], dtype = int)
        counts_arr = numpy.zeros([nof_objects, nof_nn], dtype = int)
        for o in range(nof_objects):
            neighbors = []
            for i,h in enumerate(hash_tabels):
                l = leafs[o,i]
                neighbors += h[l]

            counts = Counter(neighbors)
            objects_ = numpy.array(list(counts.keys()))
            ### FIXME: what if len(objects_) < nof_nn?
            counts_ = numpy.array(list(counts.values()))

            nn_inds = numpy.argsort(counts_)[::-1][:nof_nn]

            objects = objects_[nn_inds]
            nn_arr[o,:len(objects)] = objects
            counts = counts_[nn_inds]
            counts_arr[o,:len(objects)] = counts

    return nn_arr, counts_arr

def nn_classification_score(nn, y):
    n_objects = len(y)
    true_count = 0
    pred = numpy.zeros(n_objects, dtype=int)
    for i in range(n_objects):
        c = Counter(y[nn[i,1:10]])
        pred_ = c.most_common(n=1)[0][0]
        if y[i] == pred_:
            true_count += 1
        pred[i] = pred_
    return float(true_count)/n_objects, pred

def predict_urf(forest, X, y):
    leafs = forest.apply(X)
    is_good = is_good_matrix_get(forest, X)
    hash_tbls = get_hash_tbls(leafs, is_good)
    nn_list, counts_list = get_nn(leafs, hash_tbls, 100)
    score, pred = nn_classification_score(nn_list, y)
    print('Score:', score)
    return pred
