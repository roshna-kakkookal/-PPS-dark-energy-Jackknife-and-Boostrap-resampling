#!/opt/anaconda/bin/python

########################################
# Functions for computing m

'''
We will not fix value of M. Apparently degeneracy between H0 and M can be broken if we use Sh0es data.
See arXiv: Brout 2022.
'''
#M = -19.253 #Riess et al (2022), see 2303.06974
def f(z, omega_m, omega_l, w0, wa):
#    with np.errstate(invalid='warn'):    
    if model == 'flat LCDM':
        #    omega_l = 0.
        #    w = 0.
        #    wa = 0.
        inv_H =  1./np.sqrt(omega_m*(1.+z)**3+(1.-omega_m))
    elif model == 'LCDM':
        #    w = 0.
        #    wa = 0.
        inv_H =  1./np.sqrt(omega_m*(1.+z)**3 + omega_l + (1.- omega_m - omega_l)*(1. + z)**2)
    elif model == 'flat wCDM':
        #    omega_l = 0.
        #    wa = 0.
        inv_H = 1./np.sqrt(omega_m*(1.+z)**3  + (1.- omega_m)*(1. + z)**(3.*(1. + w0)))
    elif model == 'wCDM':
        # wa = 0.
        inv_H = 1./np.sqrt(omega_m*(1.+z)**3  + omega_l*(1. + z)**(3.*(1. + w0))+ (1.- omega_m - omega_l)*(1. + z)**2)
    elif model == 'flat w0-wa CDM':
        #    omega_l = 0.
        inv_H = 1./np.sqrt(omega_m*(1.+z)**3  + (1.- omega_m)*np.exp(-3.*wa*z/(1.+z))*(1. + z)**(3.*(1. + w0 + wa)))
    return inv_H


def construct_integration_interpolating_function(omega_m, omega_l, w0, wa, a=0., b=3., n=300000):
    from scipy.interpolate import interp1d
    x_arr = np.linspace(a, b, n+1)
    step = (b - a)/n
    f_arr = f(x_arr, omega_m, omega_l, w0, wa)
    intg_list=[]
    x_interp_list = []
    d = 100
    intg = 0.
    for i in range(int(n/d)):
        intg = intg + f_arr[i*d]/2.
        intg = intg + np.sum(f_arr[i*d+1:(i+1)*d])
        intg = intg + f_arr[(i+1)*d]/2.
        intg_list.append(intg)
        x_interp_list.append(x_arr[(i+1)*d])    
    intg_arr = np.array(intg_list)*step
    x_interp_arr = np.array(x_interp_list)
    integration_interpolation_function = interp1d(x_interp_arr,intg_arr)
    return integration_interpolation_function



def D_L(z, interp_integration, omega_k, h):
    if omega_k == 0: 
        D=(3000/h)*(1+z)*interp_integration(z)
    elif omega_k > 0:
        D = (3000/h)*(1+z)*(1/np.sqrt(np.abs(omega_k)))*np.sinh(np.sqrt(np.abs(omega_k))*interp_integration(z))
    elif omega_k < 0:
        D = (3000/h)*(1+z)*(1/np.sqrt(np.abs(omega_k)))*np.sin(np.sqrt(np.abs(omega_k))*interp_integration(z))
    return(D)


#######################################
# Calculation of chi square


def chi_square(params):
#    print("Estimating MLE of", model, "...")
    if model == 'flat LCDM':
        omega_m, h, M = params
        omega_k = 0.
        omega_l = 0.
        w0 = 0.
        wa = 0.
    elif model == 'LCDM':
        omega_m, omega_l, h, M = params
        omega_k = 1. - omega_m - omega_l
        w0 = 0.
        wa = 0.
        E_z = (omega_m*(1.+z)**3 + omega_l + (1.- omega_m - omega_l)*(1. + z)**2)
        if np.any(E_z < 0):
            chi_square = np.inf
            return np.inf
    elif model == 'flat wCDM':
        omega_m, w0, h, M = params
        omega_k = 0.
        omega_l = 0.
        wa = 0.
    elif model == 'wCDM':
        omega_m, omega_l, w0, h, M = params
        omega_k = 1 - omega_m - omega_l
        wa = 0.
        E_z = (omega_m*(1.+z)**3  + omega_l*(1. + z)**(3.*(1. + w0))+ (1.- omega_m - omega_l)*(1. + z)**2) 
        if np.any(E_z < 0):
            chi_square = np.inf
            return np.inf
    elif model == 'flat w0-wa CDM':
        omega_m, w0, wa, h, M = params
        omega_k = 0.
        omega_l = 0

    int_func = construct_integration_interpolating_function(omega_m, omega_l, w0, wa, 0., 3., 300000)

    m_model = (ceph_dist+M)*is_calib
    m_model = m_model+(5*np.log10(D_L(z,int_func,omega_k,h)) + 25 + M)*np.logical_not(is_calib)

    delta = m_given - m_model
    chi_square = np.dot(delta, np.matmul(curvmat, delta))
    if not np.isfinite(chi_square):
        return np.inf
    return (chi_square)


# compute_central_difference

def compute_second_derivative(index_i, index_j, params, delta):
   iminus = np.copy(params)
   iplus = np.copy(params)
   iminus[index_i]-=params[index_i]*delta
   iplus[index_i]+=params[index_i]*delta
   if index_i==index_j:
       diff = (chi_square(iplus) - 2.*chi_square(params) + chi_square(iminus))/delta**2/params[index_i]**2
   else:
       iplus_jplus = np.copy(iplus)
       iplus_jplus[index_j] +=params[index_j]*delta
       iplus_jminus = np.copy(iplus)
       iplus_jminus[index_j] -=params[index_j]*delta
       iminus_jplus = np.copy(iminus)
       iminus_jplus[index_j] +=params[index_j]*delta
       iminus_jminus = np.copy(iminus)
       iminus_jminus[index_j] -=params[index_j]*delta
       diff = (chi_square(iplus_jplus) - chi_square(iplus_jminus) - chi_square(iminus_jplus) + chi_square(iminus_jminus))/(4.*delta**2*params[index_i]*params[index_j])
   return diff


def plot_contours_gls(bestfit, curvature):
    import matplotlib.pyplot as plt
    covariance = np.linalg.inv(curvature)
    sigma = np.sqrt(covariance.diagonal())
    #choose two parameters
    p1 = bestfit[0]
    p2 = bestfit[-2]
    p1_lower = p1 - 3*sigma[0]
    p1_upper = p1 + 3*sigma[0]
    p2_lower = p2 - 3*sigma[-2]
    p2_upper = p2 + 3*sigma[-2]
    curvprime = curvature[[0,-2],:][:,[0,-2]]
    
    def deltachisquare(theta1, theta2, p1, p2, curvprime):
        delta = np.array([theta1-p1,theta2-p2])
        return np.matmul(np.matmul(delta.T,curvprime),delta)
    
    custom_levels = [2.279,5.991]
    
    X,Y = np.meshgrid(np.linspace(p1_lower, p1_upper,100),np.linspace(p2_lower, p2_upper,100))
    print(X.shape, Y.shape)
    print(X)
    print(Y)
    Z_list = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_list.append(deltachisquare(X[i,j],Y[i,j],p1,p2,curvprime))
    Z_arr =np.array(Z_list)
    Z = Z_arr.reshape(X.shape)
    print(Z.shape)
    print(Z)
    plt.figure(figsize=(8,6))
    plt.contour(X,Y,Z,levels=custom_levels, colors='red', linestyles='--', linewidths=2)
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$h$")
    plt.savefig("gls_contour_flcdm.pdf")


# Compute Covariance matrix

def compute_curvature(param_values):
    params = param_values
    n_params = len(params)
    count = 0
    max_iter = 100
    delta = 0.5
    A_ij = np.zeros((n_params, n_params))
    err = 1000.*np.ones((n_params, n_params))
    while(np.any(err > 1.e-4) ):
        for i in range(n_params):
            for j in range(i, n_params):
                ddchisq = compute_second_derivative(i, j, params, delta)
                if count > 0 and count < max_iter:
                    err[i,j] = np.abs((A_ij[i,j] - ddchisq)/A_ij[i,j])
                    err[j, i] = err[i, j]
                elif count >= max_iter:
                    print ("maximum iteration of 100 reached while trying to find second derivative")
                A_ij[i, j] = ddchisq
                A_ij[j,i] = A_ij[i, j]
        count +=1
        delta = delta/2
    return A_ij/2.

##########################################################################

# For emcee


def log_prior(params):
    if np.all((np.array(lower)<np.array(params))) and np.all((np.array(params)<np.array(upper))):
        return 0.0
    else: 
        return -np.inf

def log_likelihood(params):
    chi2 = chi_square(params)
    return -0.5 * chi2 


def log_posterior(params):
    if log_prior(params) == -np.inf:
        return log_prior(params)
    return log_prior(params) + log_likelihood(params)


##########################################################################

# For polychord

def polychord_likelihood(params):
    logL = log_likelihood(params)
    return logL, []

def polychord_prior(cube):
    params = np.zeros_like(cube)
    for i in range(len(cube)):
        params[i] = lower[i] + cube[i] * (upper[i] - lower[i])
    return params


########################################################################


# For jackknife

def chi_square_jack(params):
    if model == 'flat LCDM':
        omega_m, h, M = params
        omega_k = 0.
        omega_l = 0.
        w0 = 0.
        wa = 0.
    elif model == 'LCDM':
        omega_m, omega_l, h, M = params
        omega_k = 1 - omega_m - omega_l
        w0 = 0.
        wa = 0.
    elif model == 'flat wCDM':
        omega_m, w0, h, M = params
        omega_k = 0.
        omega_l = 0.
        wa = 0.
    elif model == 'wCDM':
        omega_m, omega_l, w0, h, M = params
        omega_k = 1 - omega_m - omega_l
        wa = 0.
    elif model == 'flat w0-wa CDM':
        omega_m, w0, wa, h, M = params
        omega_k = 0.
        omega_l = 0.

    int_func = construct_integration_interpolating_function(omega_m, omega_l, w0, wa, 0., 3., 300000)

    m_model = (ceph_jack+M)*is_calib_jack
    m_model = m_model+(5*np.log10(D_L(z_jack,int_func,omega_k,h)) + 25 + M)*np.logical_not(is_calib_jack)
    delta = m_jack - m_model
    chi_square_jack = np.dot(delta, np.matmul(cov_jack, delta))
    return (chi_square_jack)



##########################################################################

# For Bootstrap

def chi_square_boot(params):
    if model == 'flat LCDM':
        omega_m, h, M = params
        omega_k = 0.
        omega_l = 0.
        w0 = 0.
        wa = 0.
    elif model == 'LCDM':
        omega_m, omega_l, h, M = params
        omega_k = 1 - omega_m - omega_l
        w0 = 0.
        wa = 0.
    elif model == 'flat wCDM':
        omega_m, w0, h, M = params
        omega_k = 0.
        omega_l = 0.
        wa = 0.
    elif model == 'wCDM':
        omega_m, omega_l, w0, h, M = params
        omega_k = 1 - omega_m - omega_l
        wa = 0.
    elif model == 'flat w0-wa CDM':
        omega_m, w0, wa, h, M = params
        omega_k = 0.
        omega_l = 0.

    int_func = construct_integration_interpolating_function(omega_m, omega_l, w0, wa, 0., 3., 300000)

    m_model = (ceph_bs+M)*is_calib_bs
    m_model = m_model+(5*np.log10(D_L(z_bs,int_func,omega_k, h)) + 25 + M)*np.logical_not(is_calib_bs)
    delta = m_bs - m_model
    chi_square_bs = np.dot(delta, np.matmul(cov_bs, delta))
    return (chi_square_bs)


##########################################################################

def run_chi_square_minimization():

    from getdist import plots, MCSamples

    import matplotlib.pyplot as plt


    print("Running chi square minimization routine for ", model, "...")

    result = pybobyqa.solve(chi_square, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000)

    print ("Model fitted:", model)
    print ("Model parameters:", model_params)
    print("chi_square corresponding to values obtained by Brout et al (2022):",chi_square_brout)


    print (result)

    values = result.x
    curv = compute_curvature(values)
    print("Curvature matrix at MLE:\n", curv)
    cov = np.linalg.inv(curv)
    print("Covariance matrix at MLE :\n", cov)

    print("Matrix of standard deviations:\n",np.sqrt(np.abs(cov)))


#######################################################

def run_emcee():
    import time
    import emcee
    from multiprocessing import Pool
    print("Running emcee routine for ", model, "...")
    width_factor = 0.6
    lower_val = np.array(initial_guess) - width_factor*np.array(initial_guess)#np.array([0.2, 0.6, -20])
    lower_val = np.maximum(np.abs(lower_val),np.abs(lower))*np.sign(initial_guess)
    spread = 2.*( np.array(initial_guess) - lower_val)#np.array([0.2, 0.2, 1.5])
    n_walkers = 16
    n_dim = len(initial_guess)
    n_steps = 40000
    nburn = (int)(0.3*n_steps)

    pos = lower_val + spread * np.random.rand(n_walkers, n_dim)
 
    num_procs = 48
    with Pool(processes=num_procs) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, n_steps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))


    print("computing samples...")
    
    from getdist import plots, MCSamples
    
    # Get the chains from emcee
    # emcee chains have shape (nsteps, nwalkers, ndim)
    chain = sampler.get_chain(discard=nburn)  # Shape: (nsteps, nwalkers, ndim)
    log_prob = sampler.get_log_prob(discard=nburn)  # Shape: (nsteps, nwalkers)
    # Convert to MCSamples using multiple chains
    # Each emcee walker should be treated as a separate chain
    # Convert to list of chains (each walker becomes a chain)
    chain_list = [chain[:, i, :] for i in range(chain.shape[1])]  # List of (nsteps, ndim)
    logprob_list = [log_prob[:, i] for i in range(log_prob.shape[1])]  # List of (nsteps,)


    samples = MCSamples(samples=chain_list,  # List of arrays, each walker as separate chain
            loglikes=[-lp for lp in logprob_list],  # List of -log(posterior) arrays
            names=model_params, labels=model_params, label=model)

    f=plots.getSubplotPlotter(width_inch=12)

    f.triangle_plot(samples,filled=True,legend_loc='upper right', 
            line_args=[{'lw':1, 'color':'darkblue'}], contour_colors=['darkgreen'])

    output_root = 'emcee_'+model.replace(" ","")
    f.export(output_root+'.pdf')
    output_header = ['#', 'like']+ model_params
    for i in range(n_walkers):
        file_name = 'chains/'+output_root+f"{i:02d}"+'.dat'
        with open(file_name, 'w') as f:
            np.savetxt(f, [output_header], fmt="%s",delimiter="\t")
            np.savetxt(f, ["\n"], fmt="%s")
            np.savetxt(f, np.column_stack((log_prob[:, i],chain[:, i, :])))

    marg_stats = samples.getMargeStats()
    covmat = samples.getCovMat()
    likestats = samples.getLikeStats()
    GR_criteria = samples.getGelmanRubin()

    print("An emcee analysis of", model, "is complete.")
    print("Margestats:\n", marg_stats)
    print("Covariance matrix: \n", covmat.matrix)
    print("Likestats: ", likestats)
    print("Gellman-Rubin convergence: ", GR_criteria)

#################################################################

def run_polychord():
    import time
    import pypolychord
    from pypolychord.settings import PolyChordSettings
    print("Running polychord routine for ", model, "...")

    n_dim = len(initial_guess)
    nDerived = 0
    
    def dumper(live, dead, logweights, logZ, logZerr):
        print("Last dead point:", dead[-1])

    settings = PolyChordSettings(n_dim, nDerived)
    settings.base_dir = "chains"
    settings.file_root = "polychord_brout_try"+ model.replace(" ","")
    settings.nlive = 250 #200
    settings.do_clustering = True
    settings.read_resume = True
    settings.precision_criterion= 0.001

    output = pypolychord.run_polychord(polychord_likelihood, n_dim, nDerived, settings, polychord_prior, dumper)


    print(output)
    print("finished running pypoychord.")

    paramnames = [('p%i' % i, model_params[i]) for i in range(n_dim)]
    output.make_paramnames_files(paramnames)


    import getdist.plots
    posterior = output.posterior
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(posterior, filled=True)
    g.export('polychord_wcdm_brout.pdf')

    print("polychord run finished")

#############################################

def run_jackknife():
    print("Running jackknife routine for ", model, "...")
    param_bestfit_list = []
    global z_jack, m_jack, is_calib_jack, ceph_jack, cov_jack
    for i in range(num_sn):
        z_jack = np.delete(z, i)
        m_jack = np.delete(m_given, i)
        is_calib_jack = np.delete(is_calib, i)
        ceph_jack = np.delete(ceph_dist, i)
        cov_jack = np.delete(np.delete(curvmat, i, axis=0), i, axis=1)
        result_jackknife = pybobyqa.solve(chi_square_jack, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000)
        values_jackknife = result_jackknife.x
        param_bestfit_list.append(values_jackknife)
    param_bestfit_arr = np.array(param_bestfit_list)
    np.savetxt(model.replace(" ", "") + 'jackknife_param_list.txt',param_bestfit_arr)
    jack_estimate = np.mean(param_bestfit_arr, axis=0)
    jack_var = ((num_sn - 1) / num_sn) * np.sum((param_bestfit_arr- jack_estimate)**2, axis=0)
    jack_err = np.sqrt(jack_var)
    full_estimate = pybobyqa.solve(chi_square, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000).x
    jack_bias = (num_sn - 1)*(jack_estimate - full_estimate)
    
    covariance =  np.zeros((jack_estimate.size, jack_estimate.size))
    print("Shape of param_bestfit_arr:",param_bestfit_arr.shape)
    print("Shape of param_bestfit_arr[1,:]",param_bestfit_arr[1,:].shape)
    print("Shape of jack_estimate",jack_estimate.shape)
    print("Shape of (param_bestfit_arr[i,:] - jack_estimate)",(param_bestfit_arr[1,:] - jack_estimate).shape)
    print("Shape of covariance",covariance.shape)
    for i in range(num_sn):
        covariance += np.outer((param_bestfit_arr[i,:] - jack_estimate),(param_bestfit_arr[i,:] - jack_estimate))
    covariance *= ((num_sn-1)/num_sn)

    print("A jackknife analysis of", model, "is complete.")
    print("Estimate using all data: \n", full_estimate)
    print("Jackknife estimate: \n", jack_estimate)
    print("Error in Jacknife estimate: \n", jack_err)
    print("Bias : \n", jack_bias)
    print("Covariance matrix :\n", covariance)
    print("Curvature matrix:\n", np.linalg.inv(covariance))
    print("Jackknife corrected estimate:\n", full_estimate - jack_bias)

###############################################


'''
# use for d-delete jackknife only


def run_jackknife():
    print("Running jackknife routine for ", model, "...")
    param_bestfit_list = []
    global z_jack, m_jack, is_calib_jack, ceph_jack, cov_jack
    numiter = 500
    for i in range(numiter):
        # Initialize the generator
        rng = np.random.default_rng()
        # Select d unique numbers from range 0 to N-1
        # N = range limit, d = number of samples
        d = 170 # 41
        indices = rng.choice(num_sn, size=num_sn-d, replace=False)
        z_jack = z[indices]#np.delete(z, i)
        m_jack = m_given[indices]#np.delete(m_given, i)
        is_calib_jack = is_calib[indices]#np.delete(is_calib, i)
        ceph_jack = ceph_dist[indices]#np.delete(ceph_dist, i)
        cov_jack = curvmat[:,indices][indices,:]#np.delete(np.delete(curvmat, i, axis=0), i, axis=1)
        result_jackknife = pybobyqa.solve(chi_square_jack, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000)
        values_jackknife = result_jackknife.x
#        print(values_jackknife)
        param_bestfit_list.append(values_jackknife)
    param_bestfit_arr = np.array(param_bestfit_list)
    np.savetxt("lcdm_mock_exactredshift"+model.replace(" ", "") + '170-delete-numiter500-jackknife_param_list.txt',param_bestfit_arr)
    jack_estimate = np.mean(param_bestfit_arr, axis=0)
#    print((param_bestfit_arr - jack_estimate)**2)
    jack_var = ((num_sn - d) / d/numiter) * np.sum((param_bestfit_arr- jack_estimate)**2, axis=0)
    jack_err = np.sqrt(jack_var)
    full_estimate = pybobyqa.solve(chi_square, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000).x
    jack_bias = (num_sn - d)/d*(jack_estimate - full_estimate)

    covariance =  np.zeros((jack_estimate.size, jack_estimate.size))
    print("Shape of param_bestfit_arr:",param_bestfit_arr.shape)
    print("Shape of param_bestfit_arr[1,:]",param_bestfit_arr[1,:].shape)
    print("Shape of jack_estimate",jack_estimate.shape)
    print("Shape of (param_bestfit_arr[i,:] - jack_estimate)",(param_bestfit_arr[1,:] - jack_estimate).shape)
    print("Shape of covariance",covariance.shape)
    for i in range(num_sn):
        covariance += np.outer((param_bestfit_arr[i,:] - jack_estimate),(param_bestfit_arr[i,:] - jack_estimate))
    covariance *= ((num_sn-1)/num_sn)

    print("A jackknife analysis of", model, "is complete.")
    print("Estimate using all data: \n", full_estimate)
    print("Jackknife estimate: \n", jack_estimate)
    print("Error in Jacknife estimate: \n", jack_err)
    print("Bias : \n", jack_bias)
    print("Covariance matrix :\n", covariance)
    print("Curvature matrix:\n", np.linalg.inv(covariance))
    print("Jackknife corrected estimate:\n", full_estimate - jack_bias)


'''
###########################################################

def run_bootstrap():
    global z_bs, m_bs, is_calib_bs, ceph_bs, cov_bs
    print("Running bootstrap routine for ", model, "...")
    n_bs_samples = 1000
    param_bs_list = []
    for i in range(n_bs_samples):
        indices = np.random.choice(num_sn, size=num_sn, replace=True)
        z_bs = z[indices]
        m_bs = m_given[indices]
        is_calib_bs = is_calib[indices]
        ceph_bs = ceph_dist[indices]
        cov_bs = curvmat[indices][:, indices]
        result_bs = pybobyqa.solve(chi_square_boot, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000)
        values_bs = result_bs.x
        param_bs_list.append(values_bs)
    param_bs_arr = np.array(param_bs_list)
    np.savetxt(model.replace(" ", "") + 'bootstrap_param_list.txt',param_bs_arr)
    bs_estimate = np.mean(param_bs_arr, axis=0)
    full_estimate = pybobyqa.solve(chi_square, initial_guess, bounds= (lower, upper), scaling_within_bounds=True,maxfun=10000).x
    bs_var =(1/n_bs_samples)*np.sum((param_bs_arr - full_estimate)**2, axis = 0)
    bs_err = np.sqrt(bs_var)
    bs_bias = (bs_estimate - full_estimate)
    param_bs_arr_corrected = full_estimate - bs_bias
    print("A bootstrap analysis of", model, "is complete.")
    print("MLE estimate:", full_estimate)
    print("Bootstrap estimate:", bs_estimate)
    print("Error in Bootstrap estimate:", bs_err)
    print("Bias :", bs_bias)
    print("Bias corrected estimate:\n", param_bs_arr_corrected)


##########################################################################

# Main code

import numpy as np
import pandas as pd
import pybobyqa


#####################################
# Read data and covariance matrix


data_file = "../Pantheon+SH0ES.dat"
covmat_file = "../Pantheon+SH0ES_STAT+SYS.cov"

#print(f"Loading data from {data_file}")
data = pd.read_csv(data_file, sep=r'\s+')
origlen = len(data)

'''
Read redshift: We use zHD following 2303.06974.
'''
zHD = data['zHD'].to_numpy()


# Read in m(zi) = mu_i + M
m_obs = data['m_b_corr'].to_numpy()

# Does the galaxy contain Cepheid?
is_calib = data['IS_CALIBRATOR'].to_numpy()

# Cepheid distance: to be used instead of mu_th
ceph_dist = data['CEPH_DIST'].to_numpy()


z = zHD
m_given = m_obs



# Read in covariance matrix. 
covmat = np.loadtxt(covmat_file)#a list of values, single column
num_sn = int(covmat[0]) #read in number of supernovae
curvmat = np.linalg.inv(covmat[1:].reshape(num_sn,num_sn))#make it nxn form



model = 'flat LCDM'
method = 'GLS'
#method = 'emcee'
#method = 'jackknife'
#method = 'bootstrap'
#method ='polychord'


print("*********************************************\n \n \n")

# Set starting value and prior

if model == 'flat LCDM':
    model_params =[ "\Omega_m", "h", "M"]
    initial_guess = [0.33, 0.72, -19.]
    bounds = [(0.01, 0.9), (0.55, 0.91), (-20.,-18.)]
    lower = [0.01, 0.55, -20.]
    upper = [0.9, 0.91, -18]
    chi_square_brout = chi_square([0.334,0.736,-19.248])

elif model == 'LCDM':
    model_params =["\Omega_m", "\Omega_{DE}", "h", "M"]
    initial_guess = [0.33, 0.65, 0.72, -19.]
    bounds = [(0.01, 0.9), (-2, 2), (0.55, 0.91), (-20.,-18.)]
    lower = [0.01, -2, 0.55, -20.]
    upper = [0.9, 2, 0.91, -18.]
    chi_square_brout = chi_square([0.306, 0.625, 0.734, -19.248])
  
elif model == 'flat wCDM':
    model_params = ["\Omega_m", "w_{0}", "h", "M"]
    initial_guess = [0.33, -1.0, 0.72, -19.]
    bounds = [(0.01, 0.9), (-4., 4), (0.55, 0.91), (-20.,-18.)]
    lower = [0.01, -4, 0.55, -20.]
    upper = [0.9, 4, 0.91, -18.]
    chi_square_brout = chi_square([0.309, -0.90, 0.735, -19.248])
elif model == 'wCDM':
    model_params = ["\Omega_m", "\Omega_{DE}", "w_{0}", "h", "M"]
    initial_guess = [0.33, 0.65, -1., 0.72, -19.]
    bounds = [(0.01, 0.9), (-2, 2), (-4, 4), (0.55, 0.91), (-20.,-18.)]
    lower = [0.01, -2, -4, 0.55, -20.]
    upper = [0.9, 2, 4, 0.91, -18.]
    chi_square_brout = ' Not analysed by Brout et al.' 
elif model == 'flat w0-wa CDM':
    model_params = ["\Omega_m", "w_{0}", "w_{a}", "h", "M"]
    initial_guess = [0.33, -1., -0.5, 0.72, -19.]
    bounds = [(0.01, 0.9), (-4, 4), (-4., 4.), (0.55, 0.91), (-20.,-18.)]
    lower = [0.01, -4, -4., 0.55, -20.]
    upper = [0.9, 4, 4., 0.91, -18.]
    chi_square_brout = chi_square([0.403, -0.93, -0.1, 0.733, -19.248])


if method == 'GLS':
    run_chi_square_minimization()
elif method == 'emcee':
    run_emcee()
elif method == 'jackknife':
    run_jackknife()
elif method == 'bootstrap':
    run_bootstrap()
elif method == 'polychord':
    run_polychord()



