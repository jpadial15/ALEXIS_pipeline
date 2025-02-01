import cvxpy as cp
import numpy as np
import pandas as pd


def norm_RMSE(data):
    
    a = ((np.array(data)))
    
#     b = np.array([1]*a.shape[0])
    
    return(a)



def norm_pears(data):
    
    a = ((np.array(data)))
    
    b = np.array([1]*a.shape[0])
    
    normed_pears = np.abs(a-b)/2
    
    return(normed_pears)



def norm_etot(data):
    
    a = ((np.array(data)))
    
#     b = np.array([1]*a.shape[0])


    def custom_function(x):
        
        
        if x > 0 and x <1: # overestimating the vector ==> from 1 (worst) to zero (best)
            return(1-x)
        
        if x < 0 and x > -1: # opposite coef ==> capped at 1
            return(1)
        
        if x > 1 and x <2: # underestimating the vector ==> from 0 (best) to 1 (worst)
            return(x-1)
        
        if x > 2:
            return(1)
        
        if x < -1:
            return(1+x-x)

    
    y_values = [custom_function(a) for a in data]
    
    return(y_values)

def norm_vtot(data):
    
    a = ((np.array(data)))
#     b = np.array([1]*a.shape[0])

    def custom_function(x):
        
        
        if x > 0 and x <1:
            return(np.sqrt(np.log(2-x))) # from zero to .69
        
        if x < 0 and x > -1:
            return(1)
        
        if x > 1 and x <2:
            return((np.sqrt(1-np.exp(1-x)))) # from zero to .63
        
        if x > 2:
            return(1)
        
        if x < -1:
            return(1+x-x)

    
    y_values = [custom_function(a) for a in data]

    return(y_values)


def test_norm_vtot_etot(data):
    
    a = ((np.array(data)))
    b = np.array([1]*a.shape[0])

    def custom_function(x):
        
        
        if x > 0 and x < 1:
            return(1-x)

        if x < 0 and x > -1:
            return(1-x)
        
        if x > 1 and x <2:
#             
            return(x-1)
        
        if x > 2:
            return(x-1)
        
        if x < -1:

            return(1-x)

    
    y_values = [custom_function(a) for a in data]

    return(y_values)


def best_fit(this_df):

    distance_from_pears = (np.sqrt(np.square(1 - this_df.pears_corr.iloc[0])))

    distance_from_pval = (np.sqrt(np.square(0 - this_df.p_val.iloc[0])))

    distance_from_RMSE = (np.sqrt(np.square(0 - this_df.RMSE.iloc[0])))

    distance_from_MSE = (np.sqrt(np.square(0 - this_df.MSE.iloc[0])))

    distance_from_E_tot = (np.sqrt(np.square(1 - this_df.E_tot.iloc[0])))

    distance_from_vector_fit = (np.sqrt(np.square(1 - this_df.vector_fit.iloc[0])))

    distance_from_origin = distance_from_vector_fit + distance_from_E_tot + distance_from_MSE + distance_from_RMSE + distance_from_pval + distance_from_pears

    return(distance_from_origin)





def fit_least_sq_single_vector(vector, xray_vector):

    variables_E_tot = cp.Variable(1)

    obj_model_E_tot = cp.Minimize(cp.sum_squares(vector*variables_E_tot - xray_vector ))

    prob_model_E_tot = cp.Problem(obj_model_E_tot)

    result_model_E_tot = prob_model_E_tot.solve()
    
    try:
        return(variables_E_tot.value[0])
    except:
        return(10000)

def fit_least_sq_E_tot(E_tot, E_tot_xray):

    variables_E_tot = cp.Variable(1)

    obj_model_E_tot = cp.Minimize(cp.sum_squares(E_tot*variables_E_tot - E_tot_xray[0] ))

    prob_model_E_tot = cp.Problem(obj_model_E_tot)

    result_model_E_tot = prob_model_E_tot.solve()

    try:
        return(variables_E_tot.value[0])
    except:
        return(10000)