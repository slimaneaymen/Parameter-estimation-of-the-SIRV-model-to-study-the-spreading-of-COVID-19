# Network architencture
layers = 5
hidden_units =48
activation = 'Sigmoid'
# input_dim = 9 #size of the input layer
input_dim = 7 #size of the input layer
# output_dim = 5 #size of the output layer
output_dim = 4 #size of the output layer


# Fixed parameters
alpha_2 = 1e-3
beta_2 = 0.
lam = 0.95
eps = 1/180
zeta = 1e-3
eta = 5e-4
alpha_1 = 1/180
# Model parameters bundle
# a_0_set = [0.1, 0.3]
i_0_set = [0.07, 0.09]
v_0_set = [0., 0.]
r_0_set = [0.20, 0.23]
# alpha_1s = [1/180, 1/181]
beta_1s = [0.1, 0.25]
gammas = [0.07, 0.12]
deltas = [1e-3, 5e-3]

# set the initial conditions bundle
initial_conditions_set = [i_0_set, v_0_set, r_0_set]

# set the parameters bundle
# parameters_bundle = [alpha_1s, beta_1s, gammas, deltas]
parameters_bundle = [beta_1s, gammas, deltas]

# set the fixed parameters dictionary
# parameters_fixed = { 'alpha_2' : alpha_2,
#                         'beta_2' : beta_2, 
#                         'lam' : lam, 
#                         'eps' : eps, 
#                         'zeta' : zeta, 
#                         'eta' : eta }
parameters_fixed = { 'alpha_1' : alpha_1,
                        'lam' : lam, 
                        'eps' : eps, 
                        'zeta' : zeta}

#model_name = 'restart.pt' # for doing annealing from a solution w. different parameters                        
# model_name = 'Unsupervised_a_0={}_i_0={}_v_0={}_r_0={}_beta_1s={}_gammas={}_alpha_1s={}.pt'.format(i_0_set, v_0_set, r_0_set, beta_1s, gammas, alpha_1s)                        
model_name = 'Unsupervised_i_0={}_v_0={}_r_0={}_beta_1s={}_gammas={}.pth'.format(i_0_set, v_0_set, r_0_set, beta_1s, gammas)

                        
                        
