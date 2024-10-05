import cvxpy as cp
import numpy as np
import pandas as pd





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