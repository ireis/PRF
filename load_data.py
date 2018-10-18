import pickle, numpy


def cv_vals_get(X_all, y_all, dX_all):
    """
    Draw a subsample of the data for the train set, and another subsample for a test set.
    """
    objnum_all = X_all.shape[0]
    feature_num = X_all.shape[1]
    objnum = int(objnum_all / 2)

    cv_inds = numpy.random.choice(range(objnum_all), objnum_all, replace = False)

    train_inds = cv_inds[:objnum]
    test_inds = cv_inds[objnum:]
        
    
    y_train = y_all[train_inds]
    X_train = X_all[train_inds]
    dX_train = dX_all[train_inds]
    #sigma_mat_train = sigma_mat_old[train_inds]
    
    #dview.push(dict(y_train=y_train), block = True)

    y_test = y_all[test_inds]
    X_test = X_all[test_inds]
    dX_test = dX_all[test_inds]

    #sigma_mat_test = sigma_mat_old[test_inds] #Works
    
    
    
    return X_train, y_train, dX_train, X_test, y_test, dX_test

def load_CV_data():
    
    print('Loading data from data/training_data.pkl, data/test_data.pkl')
    path = "data/training_data.pkl"
    with open(path, 'rb') as f:
        [X_train, y_train, relative_unc_train ,proba_train, uni_train] = pickle.load(f, encoding='latin1') 
    print(X_train.shape)

    path = "data/test_data.pkl"
    with open(path, 'rb') as f:
        [X_test, y_test, relative_unc_test ,proba_test, uni_test] = pickle.load(f, encoding='latin1')
    print(X_test.shape)
    
    
    print('Merging to a single set (in order to be able to cross validate later)')
    X_all = numpy.concatenate([X_train, X_test])
    y_all = numpy.concatenate([y_train, y_test])
    relative_unc_all = numpy.concatenate([relative_unc_train, relative_unc_test])
    proba_all = numpy.concatenate([proba_train, proba_test])

    relative_unc_all[numpy.isnan(relative_unc_all)] = numpy.median(relative_unc_all[~numpy.isnan(relative_unc_all)])
    relative_unc_all = 1/relative_unc_all
    dX_all = abs(relative_unc_all * X_all)
    
    print('drawing train and test samples')
    X_train, y_train, dX_train, X_test, y_test, dX_test = cv_vals_get(X_all, y_all, dX_all)
    print('done loading data')
    print('X_train, y_train, dX_train, X_test, y_test, dX_test')
    print(X_train.shape, y_train.shape, dX_train.shape, X_test.shape, y_test.shape, dX_test.shape)
    
    return X_train, X_test, y_train, y_test, dX_train, dX_test



def add_noise_to_lables(y_clean, noise_level = 0.7):
    """
    This function add noise to the labels.
    For each object we obtain a 
    -probability its label will be switched- == SLP (Switch Label Probability)
    by 1. drawing a number between 0 and 1
       2. multiplying the number by the noise_level parameter
    if noise_level = 0, no objects gets switched
    if noise_level = 1, aroung half the objects will get swithced
    
    Next, for each object, we draw switch_label, which is 0 or 1 with probability SLP to be 1.
    if switch_label == 1, we switch the label
    """
    
    nof_objects = len(y_clean)
    
    #draw a probability between 0 and 0.5 to switch the label (to a wrong label) for each object
    switch_label_proba = numpy.random.random(nof_objects) * noise_level
    
    y_w_fake_errors = y_clean.copy()
    
    for o_idx in range(nof_objects):
        
        slp = switch_label_proba[o_idx]
    
        # calculate new lables according to the probability to be wrong
        # binary value, 1 means switch label
        switch_label = numpy.random.choice([False,True],size=1,p = [1-slp,slp])
        #print(slp,switch_label)
        if switch_label:
            y_w_fake_errors[o_idx] = -y_w_fake_errors[o_idx]+1

    return y_w_fake_errors, 1-switch_label_proba

def add_noise_to_lables_(y_clean, noise_level = 1):
    """
    This function add noise to the labels.
    For each object we obtain a 
    -probability its label will be switched- == SLP (Switch Label Probability)
    by 1. drawing a number between 0 and 1
       2. multiplying the number by the noise_level parameter
    if noise_level = 0, no objects gets switched
    if noise_level = 1, aroung half the objects will get swithced
    
    Next, for each object, we draw switch_label, which is 0 or 1 with probability SLP to be 1.
    if switch_label == 1, we switch the label
    """
    
    nof_objects = len(y_clean)
    
    #draw a probability between 0 and 0.5 to switch the label (to a wrong label) for each object
    if noise_level == 0:
        switch_label_proba = numpy.zeros(nof_objects)
    else:
        switch_label_proba = (numpy.random.random(nof_objects) ** (1/noise_level)) * 0.5
    #switch_label_proba = switch_label_proba * 0.5 / numpy.mean(switch_label_proba)
    
    y_w_fake_errors = y_clean.copy()
    
    for o_idx in range(nof_objects):
        
        slp = switch_label_proba[o_idx]
    
        # calculate new lables according to the probability to be wrong
        # binary value, 1 means switch label
        switch_label = numpy.random.choice([False,True],size=1,p = [1-slp,slp])
        #print(slp,switch_label)
        if switch_label:
            y_w_fake_errors[o_idx] = -y_w_fake_errors[o_idx]+1

    return y_w_fake_errors, 1-switch_label_proba

def add_noise_to_lables__(y_clean, noise_level = 1):
    """
    This function add noise to the labels.
    For each object we obtain a 
    -probability its label will be switched- == SLP (Switch Label Probability)
    by 1. drawing a number between 0 and 1
       2. multiplying the number by the noise_level parameter
    if noise_level = 0, no objects gets switched
    if noise_level = 1, aroung half the objects will get swithced
    
    Next, for each object, we draw switch_label, which is 0 or 1 with probability SLP to be 1.
    if switch_label == 1, we switch the label
    """
    
    nof_objects = len(y_clean)
    
    #draw a probability between 0 and 0.5 to switch the label (to a wrong label) for each object
    if noise_level == 0:
        switch_label_proba = numpy.zeros(nof_objects)
    else:
        switch_label_proba = (numpy.random.random(nof_objects) ** (1/noise_level)) * 0.5
        
    
    
    nof_good_objs = int(nof_objects/10)
    good_objs_inds = numpy.random.choice(numpy.arange(nof_objects),  nof_good_objs, replace = False)
    noise_level_good_objs = 0.1
    if noise_level_good_objs == 0:
        switch_label_proba[good_objs_inds] = numpy.zeros(nof_good_objs)
    else:
        switch_label_proba[good_objs_inds] = (numpy.random.random(nof_good_objs) ** (1/noise_level_good_objs)) * 0.5
    #switch_label_proba = switch_label_proba * 0.5 / numpy.mean(switch_label_proba)
    
    y_w_fake_errors = y_clean.copy()
    
    for o_idx in range(nof_objects):
        
        slp = switch_label_proba[o_idx]
    
        # calculate new lables according to the probability to be wrong
        # binary value, 1 means switch label
        switch_label = numpy.random.choice([False,True],size=1,p = [1-slp,slp])
        #print(slp,switch_label)
        if switch_label:
            y_w_fake_errors[o_idx] = -y_w_fake_errors[o_idx]+1

    return y_w_fake_errors, 1-switch_label_proba

def add_noise_to_lables_normal(y_clean, object_errs ,noise_level = 1):
    """
    This function add noise to the labels.
    
    """
    
    nof_objects = len(y_clean)
    
    #draw a probability between 0 and 0.5 to switch the label (to a wrong label) for each object
    if noise_level == 0:
        switch_label_proba = numpy.zeros(nof_objects)
    else:
        switch_label_proba = (numpy.random.normal(noise_level, 0.25, nof_objects)) 
    switch_label_proba[switch_label_proba > 0.5] = 0.5
    switch_label_proba[switch_label_proba < 0] = 0
        
    
    
    nof_good_objs = int(nof_objects/10)
    good_objs_inds = numpy.argsort(object_errs)[:nof_good_objs]#numpy.random.choice(numpy.arange(nof_objects),  nof_good_objs, replace = False)
    switch_label_proba[good_objs_inds] = numpy.zeros(nof_good_objs)
    
    #noise_level_good_objs = 0.05
    #if noise_level_good_objs == 0:
    #    switch_label_proba[good_objs_inds] = numpy.zeros(nof_good_objs)
    #else:
    #    switch_label_proba[good_objs_inds] = (numpy.random.normal(noise_level_good_objs, 0.05, nof_good_objs))  
    
    #for good_obj_idx in good_objs_inds:
    #    if switch_label_proba[good_obj_idx] < 0:
    #           switch_label_proba[good_obj_idx] = 0 
    #switch_label_proba = switch_label_proba * 0.5 / numpy.mean(switch_label_proba)
    
    y_w_fake_errors = y_clean.copy()
    
    for o_idx in range(nof_objects):
        
        slp = switch_label_proba[o_idx]
    
        # calculate new lables according to the probability to be wrong
        # binary value, 1 means switch label
        switch_label = numpy.random.choice([False,True],size=1,p = [1-slp,slp])
        #print(slp,switch_label)
        if switch_label:
            y_w_fake_errors[o_idx] = -y_w_fake_errors[o_idx]+1

    return y_w_fake_errors, 1-switch_label_proba

def add_noise_to_lables_normal_feature_errs(y_clean, object_errs, noise_level_for_bad_objects):
    """
    This function add noise to the labels.
    
    """
    nof_objects = len(y_clean)
    
    switch_label_proba = numpy.zeros(nof_objects)
    
    mean_err = numpy.mean(object_errs)
    
    #Zero noise level for the good objects - the ones with low feature uncertainty
    good_objs_inds = numpy.where(object_errs < mean_err)[0]
    nof_good_objs = len(good_objs_inds)
    switch_label_proba[good_objs_inds] = numpy.zeros(nof_good_objs)
    
    #High noise level for the bad objects - the ones with high feature uncertainty
    bad_objs_inds = numpy.where(object_errs >= mean_err)[0]
    nof_bad_objs = len(bad_objs_inds)
    switch_label_proba[bad_objs_inds] = noise_level_for_bad_objects#numpy.random.normal(0.75, 0.25, nof_bad_objs)
    
    #print(mean_err, nof_good_objs, nof_bad_objs)
    
    switch_label_proba[switch_label_proba > 0.5] = 0.5
    switch_label_proba[switch_label_proba < 0] = 0
        

    y_w_fake_errors = y_clean.copy()
    
    for o_idx in range(nof_objects):
        
        slp = switch_label_proba[o_idx]
    
        # calculate new lables according to the probability to be wrong
        # binary value, 1 means switch label
        switch_label = numpy.random.choice([False,True],size=1,p = [1-slp,slp])
        #print(slp,switch_label)
        if switch_label:
            y_w_fake_errors[o_idx] = -y_w_fake_errors[o_idx]+1

    return y_w_fake_errors, 1-switch_label_proba


def get_pY(pY_true, y_fake):
    """
    Recieves a vector with the probability to be true (pY_true)
    returns a matrix with the probability to be in each class
    
    we put pY_true as the probability of the true class
    and (1-pY_true)/(nof_lables-1) for all other classes
    """
    nof_objects = len(pY_true)
    
    all_lables = numpy.unique(y_fake)
    nof_lables = len(all_lables)
    
    pY = numpy.zeros([nof_objects, nof_lables])
    
    for o in range(nof_objects):
        for c_idx, c in enumerate(all_lables):
            if y_fake[o] == c:
                pY[o,c_idx] = pY_true[o]
            else:
                pY[o,c_idx] = float(1 - pY_true[o])/(nof_lables - 1)
                
    #print('created pY', pY.shape)
    
    return pY


def get_noised_lables(y_clean, noise_level):

    y_w_fake_errors, pY_true = add_noise_to_lables_normal(y_clean, noise_level)
    
    pY = get_pY(pY_true, y_w_fake_errors)
    
    fraction_of_good_objects = (y_clean == y_w_fake_errors).sum()/len(y_clean)
    #print('fraction  of objects with good lables =',fraction_of_good_objects)
    
    return y_w_fake_errors, pY, fraction_of_good_objects



##################################################################
##################################################################
########################## FEATURES ##############################
##################################################################
##################################################################

def get_base_noise_factors(objnum, feature_num):
    """
    Draws the base noise factor of each feature and each object
    Return a matrix containing the multiplication of these two
    The final noise level (for each feature value of each object) will be 
    
   this matrix times 
   an overall noise level
   (times the feature value? the width of the feature value distribution?)
    """
    sigma_mat = numpy.zeros([objnum,feature_num])
    obj_err = numpy.ones(objnum)
    
    for obj in range(objnum):
        obj_err[obj] = numpy.random.random()
                
    for fe in range(feature_num):
        feature_err = numpy.random.random()*0.9 + 0.1
        sigma_mat[:,fe] = feature_err
        for obj in range(objnum):
            sigma_mat[obj,fe] = sigma_mat[obj,fe]*obj_err[obj]
            
    return sigma_mat

def get_noised_features(X, sigma_mat, alpha):
    """
    Creates a fake dX matrix containing uncertainties for each measurement.
    Creates new X_w_errs by adding noise to X according to dX.
    """
    
    objnum = X.shape[0]
    feature_num = X.shape[1]
    
    X_w_errs = X.copy()
    dX = sigma_mat * alpha
  
    for fe in range(feature_num):
        feature_vals_width = abs(X[:,fe]).std()
        dX[:,fe] = dX[:,fe] * feature_vals_width
        for obj in range(objnum):
            if dX[obj,fe] != 0:
                X_w_errs[obj,fe] = X_w_errs[obj,fe] + numpy.random.normal(loc = 0, scale = dX[obj,fe])
                #delta = numpy.random.exponential(scale = dX[obj,fe])
                #print(X_w_errs[obj,fe], delta)
                #X_w_errs[obj,fe] = X_w_errs[obj,fe] + delta
               
                
    
    return X_w_errs, dX

def get_noised_features_lognormal(X, sigma_mat, alpha):
    """
    Creates a fake dX matrix containing uncertainties for each measurement.
    Creates new X_w_errs by adding noise to X according to dX.
    """
    
    objnum = X.shape[0]
    feature_num = X.shape[1]
    
    X_w_errs = X.copy()
    dX = sigma_mat * alpha
  
    for fe in range(feature_num):
        feature_vals_width = abs(X[:,fe]).std()
        dX[:,fe] = dX[:,fe] * feature_vals_width
        for obj in range(objnum):
            if dX[obj,fe] != 0:
                X_w_errs[obj,fe] = X_w_errs[obj,fe] + numpy.random.lognormal(mean = 0, sigma = dX[obj,fe])
                    
    return X_w_errs, dX

def get_noised_features_poisson(X, sigma_mat, alpha):
    """
    Creates a fake dX matrix containing uncertainties for each measurement.
    Creates new X_w_errs by adding noise to X according to dX.
    """
    
    objnum = X.shape[0]
    feature_num = X.shape[1]
    
    X_w_errs = X.copy()
    dX = sigma_mat * alpha
  
    for fe in range(feature_num):
        feature_vals_width = abs(X[:,fe]).std()
        dX[:,fe] = dX[:,fe] * feature_vals_width
        for obj in range(objnum):
            if dX[obj,fe] != 0:
                X_w_errs[obj,fe] = X_w_errs[obj,fe] + numpy.random.poisson(lam = dX[obj,fe]) - dX[obj,fe]
                    
    return X_w_errs, dX
 




def get_noised_data(X_clean, dX_orig, y_clean, base_noise_mat_features, noise_level_lables, noise_level_features, dist = 'normal'):
    
    if dist == 'normal':
        X_w_errs, dX = get_noised_features(X_clean, base_noise_mat_features, noise_level_features)
    elif dist == 'lognormal':
        X_w_errs, dX = get_noised_features_lognormal(X_clean, base_noise_mat_features, noise_level_features)
    elif dist == 'poisson':
        X_w_errs, dX = get_noised_features_poisson(X_clean, base_noise_mat_features, noise_level_features)
    else:
        print('distribution not supported')
        return
    #print(dX.shape, dX_orig.shape)
    dX = numpy.sqrt(dX_orig**2 +  dX**2)
    
    o_errs = numpy.sum(base_noise_mat_features, axis = 1)
    #print(o_errs)
    
    y_w_errs, pY_true = add_noise_to_lables_normal(y_clean, o_errs, noise_level_lables)
    
    
    pY = get_pY(pY_true, y_w_errs)
    
    fraction_of_good_objects = (y_clean == y_w_errs).sum()/len(y_clean)
    #print('fraction  of objects with good lables =',fraction_of_good_objects)
    
    return X_w_errs, dX, y_w_errs, pY, fraction_of_good_objects



