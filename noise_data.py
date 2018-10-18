import numpy



def get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature):
    
    sigma_mat = numpy.ones([nof_objects,nof_features])
    obj_err = numpy.ones(nof_objects)
    #print(noise_level_per_object)

    
    for obj in range(nof_objects):
        obj_err[obj] = numpy.random.choice(noise_level_per_object)
                
    for fe in range(nof_features):
        sigma_mat[:,fe] = sigma_mat[:,fe] * numpy.random.choice(noise_level_per_feature)
        for obj in range(nof_objects):
            sigma_mat[obj,fe] = sigma_mat[obj,fe]*obj_err[obj]
            
    return sigma_mat

# 1
def random_random_noise(nof_objects, nof_features):
    
    noise_level_per_object = numpy.random.random(1000)
    noise_level_per_feature = numpy.random.random(1000)
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

def snp_noise(nof_objects, nof_features):
       
    sigma_mat = numpy.random.random((nof_objects, nof_features))
            
    return sigma_mat


# 2
def many_bad_features_noise(nof_objects, nof_features):
    
    noise_level_per_object = [1]#numpy.random.random(1000) * 0.1 + 0.1 # smaller scatter per object
    noise_level_per_feature = [0]*7 + [1]*3
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

# 3
def few_bad_features_noise(nof_objects, nof_features):
    
    noise_level_per_object = numpy.random.random(1000) #* 0.1 + 0.1 # smaller scatter per object
    noise_level_per_feature = numpy.random.random(1000) #[0.1]*8 + [1]*2
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

# 4
def bad_features_per_object_group_noise(nof_objects, nof_features):
    """
    Different objects have different features noise levels
    """
    
    nof_groups = 500
    nof_objects_in_group = int(float(nof_objects)/nof_groups)
    nof_leftover_objects = nof_objects - nof_objects_in_group*nof_groups
    
    noise_level_per_object = [0]*2 + [1]*9 #[0] #numpy.random.random(1000) * 0.1 + 0.1 # smaller scatter per object
    noise_level_per_feature = [0]*2 + [1]*8
    
    sigma_mat = numpy.zeros([nof_objects, nof_features])
    
    for i in range(nof_groups):
        start = nof_objects_in_group * (i  )
        end   = nof_objects_in_group * (i+1)
        sigma_mat[start:end] = get_base_noise_factors(nof_objects_in_group, nof_features, noise_level_per_object, noise_level_per_feature)
     
    leftover_objects = numpy.arange(end, nof_objects)
        
    sigma_mat[leftover_objects,:] = get_base_noise_factors(nof_leftover_objects, nof_features, noise_level_per_object, noise_level_per_feature)
    
    return sigma_mat


# 22
def many_bad_objects_noise(nof_objects, nof_features):
    
    noise_level_per_object = [0]*1 + [1]*175 # smaller scatter per object
    noise_level_per_feature = numpy.random.random(1000) * 0.1 + 0.1
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

# 33
def few_bad_objects_noise(nof_objects, nof_features):
    
    noise_level_per_object = [0]*2 + [1]*8 # smaller scatter per object
    noise_level_per_feature = [1]#numpy.random.random(1000) * 0.1 + 0.1
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

def frac_bad_objects_noise(nof_objects, nof_features, f_good, f_bad):
    
    noise_level_per_object = [0]*f_good + [1]*f_bad # smaller scatter per object
    noise_level_per_feature = [1]#numpy.random.random(1000) * 0.1 + 0.1
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

def frac_bad_objects_noise__(nof_objects, nof_features, f_good, f_bad):
    
    noise_level_per_object = [0]*f_good + [1]*f_bad # smaller scatter per object
    noise_level_per_feature = numpy.random.random(1000)
    
    sigma_mat = get_base_noise_factors(nof_objects, nof_features, noise_level_per_object, noise_level_per_feature)
            
    return sigma_mat

def corr_w_class_noise(nof_objects, nof_features, y, f_good, f_bad):
    
    classes = numpy.unique(y)
    
    noise_level_per_object = [0]*f_good + [1]*f_bad#numpy.random.random(1000) 
    
    noise_level_per_feature = {}
    for c in classes:
        noise_level_per_feature[c] = numpy.random.choice(numpy.random.random(1000),nof_features)
    
    sigma_mat = numpy.ones([nof_objects,nof_features])
    obj_err = numpy.ones(nof_objects)
    #print(noise_level_per_object)

    
    for obj in range(nof_objects):
        obj_err[obj] = numpy.random.choice(noise_level_per_object)
        
    for fe in range(nof_features):
        for obj in range(nof_objects):
            sigma_mat[obj,fe] = sigma_mat[obj,fe] * noise_level_per_feature[y[obj]][fe]
            sigma_mat[obj,fe] = sigma_mat[obj,fe] * obj_err[obj]
            
    return sigma_mat


def corr_w_class_noise_v2(nof_objects, nof_features, y, f_good, f_bad, noise_per_class):
    
    classes = numpy.unique(y)
    
    noise_level_per_object = [0]*f_good + [1]*f_bad#numpy.random.random(1000) 
    
    noise_level_per_feature = {}
    for c in classes:
        noise_level_per_feature[c] = numpy.random.choice(numpy.random.random(1000),nof_features)*noise_per_class[c]
    
    sigma_mat = numpy.ones([nof_objects,nof_features])
    obj_err = numpy.ones(nof_objects)
    #print(noise_level_per_object)

    
    for obj in range(nof_objects):
        obj_err[obj] = numpy.random.choice(noise_level_per_object)
        
    for fe in range(nof_features):
        for obj in range(nof_objects):
            sigma_mat[obj,fe] = sigma_mat[obj,fe] * noise_level_per_feature[y[obj]][fe]
            sigma_mat[obj,fe] = sigma_mat[obj,fe] * obj_err[obj]
            
    return sigma_mat





