import numpy as np
import torch
from scipy.integrate import odeint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from numpy.random import uniform


# def SAIVR_derivs(u, t, alpha_1, beta_1, gamma, delta, parameters_fixed):
def SAIVR_derivs(u, t, beta_1, gamma, delta, parameters_fixed):

    s, i, v, r = u  # unpack current values of u
    N = s + i + v + r
    
    args_fixed = parameters_fixed.values()

   #  alpha_2, beta_2, lam, eps, zeta, eta = [ val for val in args_fixed ]  
   #  lam, eps, zeta = [ val for val in args_fixed ]
   #  alpha_1, lam, eps, zeta = [ val for val in args_fixed ]
    alpha_1, lam, eps, zeta = [ val for val in args_fixed ]
 
    beta = beta_1
   #  alpha = alpha_1 + alpha_2
    
    derivs = [ (beta * i * s) / N + delta * s / N - (1 - lam) * eps * v - alpha_1*r,
     - (beta * i * s) / N - (zeta * v * i) / N + gamma * i,
    - delta * s / N + (zeta * v * i) / N + eps * v,
    - gamma * i - lam * eps * v + alpha_1*r]

    derivs = [ -i for i in derivs ]

    return derivs

# solve the SAIVR model with integrators
# def SAIVR_solution(t, s_0, i_0, v_0, r_0, alpha_1, beta_1, gamma, delta, parameters_fixed):
def SAIVR_solution(t, s_0, i_0, v_0, r_0, beta_1, gamma, delta, parameters_fixed):

    # Scipy Solver
    u_0 = [s_0, i_0, v_0, r_0]

    # Call the ODE solver
   #  sol_sir = odeint(SAIVR_derivs, u_0, t, args=(alpha_1, beta_1, gamma, delta, parameters_fixed))
    sol_sir = odeint(SAIVR_derivs, u_0, t, args=(beta_1, gamma, delta, parameters_fixed))

    s = sol_sir[:, 0]
   #  a = sol_sir[:, 1]
    i = sol_sir[:, 1]
    v = sol_sir[:, 2]
    r = sol_sir[:, 3]

    return s, i, v, r


def perturbPoints(grid, v0, vf, sig=0.5):
    #stochastic perturbation of the evaluation points    
    #force t[0]=t0
    delta_v = grid[1] - grid[0]  
    noise = delta_v * torch.randn_like(grid)*sig
    v = grid + noise
    v.data[2] = torch.ones(1, 1)*(-1)
    v.data[v<v0]=v0 - v.data[v<v0]
    v.data[v>vf]=2*vf - v.data[v>vf]
    return v


def generate_dataloader( t_0, t_final, train_size, batch_size, perturb=True, shuffle=True):
    # Generate a dataloader with perturbed points starting from a grid_explorer
    grid = torch.linspace(t_0, t_final, train_size).reshape(-1, 1)
    
    if perturb:
        grid = perturbPoints(grid, t_0, t_final, sig=0.3 * t_final)
    grid.requires_grad = True

    t_dl = DataLoader(dataset=grid, batch_size=batch_size, shuffle=shuffle)
    return t_dl


def SIR(u, t, beta, gamma):
    s, i, r = u  # unpack current values of u
    N = s + i + r
    derivs = [-(beta * i * s) / N, (beta * i * s) / N - gamma * i, gamma * i]  # list of dy/dt=f functions
    return derivs

def SIR_solution(t, s_0, i_0, r_0, beta, gamma):
    # Scipy Solver
    u_0 = [s_0, i_0, r_0]

    # Call the ODE solver
    sol_sir = odeint(SIR, u_0, t, args=(beta, gamma))
    s = sol_sir[:, 0]
    i = sol_sir[:, 1]
    r = sol_sir[:, 2]

    return s, i, r

# generate synthetic data of lenght size       
def generate_synthetic_data(model, t_0, t_final, initial_conditions_set, parameters_bundle, size):  
   time_sequence = np.linspace(t_0, t_final, size)
   
   # alpha_1s, beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:], parameters_bundle[3][:]   
   beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:]  
   
   # a_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
   i_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
   v_0 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=1)
   r_0 = uniform(initial_conditions_set[2][0], initial_conditions_set[2][1], size=1)
   # alpha_1 = uniform(alpha_1s[0], alpha_1s[1], size=1)
   beta_1 = uniform(beta_1s[0], beta_1s[1], size=1)
   gamma = uniform(gammas[0], gammas[1], size=1)
   delta = uniform(deltas[0], deltas[1], size=1)
   
   # a_0 = torch.Tensor([a_0]).reshape((-1, 1))
   i_0 = torch.Tensor([i_0]).reshape((-1, 1))
   v_0 = torch.Tensor([v_0]).reshape((-1, 1))
   r_0 = torch.Tensor([r_0]).reshape((-1, 1))
   # alpha_1 = torch.Tensor([alpha_1]).reshape((-1, 1))
   beta_1 = torch.Tensor([beta_1]).reshape((-1, 1))
   gamma = torch.Tensor([gamma]).reshape((-1, 1))
   delta = torch.Tensor([delta]).reshape((-1, 1))        
           
   # s_0 = 1 - (a_0 + i_0 + v_0 + r_0)
   s_0 = 1 - (i_0 + v_0 + r_0)
   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]   
   # param_bundle = [alpha_1, beta_1, gamma, delta]
   param_bundle = [beta_1, gamma, delta]

   synthetic_data = {}

   # print('\n Synthetic data parameters \n' 'S0 = {:.2f}, A0 = {:.2e}, I0 = {:.2e}, V0 = {:.2e}, R0 = {:.2f} \n'
   #        'Alpha_1$ = {:.2f}, Beta_1 = {:.2f}, Gamma = {:.2f}, Delta = {:.2e} \n'.format(s_0.item(), a_0.item(), i_0.item(), v_0.item(), r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))
   print('\n Synthetic data parameters \n' 'S0 = {:.2f}, I0 = {:.2e}, V0 = {:.2e}, R0 = {:.2f} \n'
         #  'Alpha_1$ = {:.2f}, Beta_1 = {:.2f}, Gamma = {:.2f}, Delta = {:.2e} \n'.format(s_0.item(), i_0.item(), v_0.item(), r_0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))
          'Beta_1 = {:.2f}, Gamma = {:.2f}, Delta = {:.2e} \n'.format(s_0.item(), i_0.item(), v_0.item(), r_0.item(), beta_1.item(), gamma.item(), delta.item()))

   for t in time_sequence:
       t = torch.Tensor([t]).reshape(-1,1)
      #  s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, param_bundle)
       s, i, v, r = model.parametric_solution(t, t_0, initial_conditions, param_bundle)       
      #  synthetic_data[t.item()] = [s.item(), a.item(), i.item(), v.item(), r.item()]
       synthetic_data[t.item()] = [s.item(), i.item(), v.item(), r.item()]       

   return synthetic_data


# test the model with a given set of parameters and intial conditions
def test_fitmodel(model, data_type, time_series_dict, optimized_params, average = 'D', lineW = 3):
   time_sequence = list(time_series_dict.keys())
   t = torch.Tensor(time_sequence).reshape(-1,1)
   t_0 = time_sequence[0]
   n_test = len(t)
   # s0, a0, i0, v0, r0, alpha1, beta1, gamma0, delta0 = optimized_params
   # s0, i0, v0, r0, alpha1, beta1, gamma0, delta0 = optimized_params
   s0, i0, v0, r0, beta1, gamma0, delta0 = optimized_params
   
   # a_0 = a0*torch.ones(n_test)
   i_0 = i0*torch.ones(n_test)
   v_0 = v0*torch.ones(n_test)
   r_0 = r0*torch.ones(n_test)
   # alpha_1 = alpha1*torch.ones(n_test)
   beta_1 = beta1*torch.ones(n_test)
   gamma = gamma0*torch.ones(n_test)
   delta = delta0*torch.ones(n_test)
   # a_0 = torch.Tensor(a_0).reshape((-1, 1))
   i_0 = torch.Tensor(i_0).reshape((-1, 1))
   v_0 = torch.Tensor(v_0).reshape((-1, 1))
   r_0 = torch.Tensor(r_0).reshape((-1, 1))
   # alpha_1 = torch.Tensor(alpha_1).reshape((-1, 1))
   beta_1 = torch.Tensor(beta_1).reshape((-1, 1))
   gamma = torch.Tensor(gamma).reshape((-1, 1))
   delta = torch.Tensor(delta).reshape((-1, 1))
   # s_0 = 1 - a_0 - i_0 - v_0 - r_0
   s_0 = 1 - i_0 - v_0 - r_0
   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]
   # params_bundle = [alpha_1, beta_1, gamma, delta]
   params_bundle = [beta_1, gamma, delta]
   
   # s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)
   s, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)

   t = t.detach().numpy()
   s = s.detach().numpy()
   # a = a.detach().numpy()
   i = i.detach().numpy() 
   v = v.detach().numpy()
   r = r.detach().numpy() 
  
   true_values = list(time_series_dict.values())

   if data_type == 'synthetic' :
      # s_true, a_true, i_true, v_true, r_true = map(list, zip(*true_values))
      s_true, i_true, v_true, r_true = map(list, zip(*true_values))                                                        

      plt.plot(t, s_true,'--g', label=' S Data', linewidth=lineW);
      plt.plot(t, s ,'-g', label='S fit',linewidth=lineW, alpha=.5);
      plt.plot(t, v_true,'--y', label=' V Data', linewidth=lineW);
      plt.plot(t, v ,'-y', label='V fit',linewidth=lineW, alpha=.5);
      plt.plot(t, i_true,'--r', label=' I Data', linewidth=lineW);
      plt.plot(t, i ,'-r', label='I fit',linewidth=lineW, alpha=.5);
      plt.plot(t, r_true,'--b', label=' R Data', linewidth=lineW);
      plt.plot(t, r ,'-b', label='R fit',linewidth=lineW, alpha=.5);
      
   else:
      s_true, i_true, v_true, r_true = map(list, zip(*true_values))
      plt.figure(figsize=(15,15))
      plt.subplot(2, 2, 1)
      plt.plot(t, s_true,'--r', label=' S Data', linewidth=lineW);
      plt.plot(t, s ,'-r', label='S fit',linewidth=lineW, alpha=.5);
      plt.ylabel('Population percentage')  
      plt.xlabel('Days') 
      plt.legend()

      plt.subplot(2, 2, 2)
      plt.plot(t, i_true,'--b', label=' I Data', linewidth=lineW);
      plt.plot(t, i ,'-b', label='I fit',linewidth=lineW, alpha=.5);
      plt.ylabel('Population percentage')  
      plt.xlabel('Days') 
      plt.legend()

      plt.subplot(2, 2, 3)
      plt.plot(t, v_true,'--b', label=' V Data', linewidth=lineW);
      plt.plot(t, v ,'-b', label='V fit',linewidth=lineW, alpha=.5);
      plt.ylabel('Population percentage')  
      plt.xlabel('Days') 
      plt.legend()

      plt.subplot(2, 2, 4)
      plt.plot(t, r_true,'--b', label=' R Data', linewidth=lineW);
      plt.plot(t, r ,'-b', label='R fit',linewidth=lineW, alpha=.5);
      plt.ylabel('Population percentage')  
      plt.xlabel('Days') 
      plt.legend()

   # if average == 'D':
   #    plt.xlabel('Days') 
   # else: 
   #    plt.xlabel('Weeks') 
       
   # plt.ylabel('Population percentage')  
   # plt.title(' $a_0$= {:.2f}, $i_0$={:.2f}, $r_0$={:.2f}, alpha_1={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(a0,2),
   # plt.title(' $i_0$={:.2f}, $r_0$={:.2f}, alpha_1={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(i0,2),
               # round(r0,2), round(alpha1,2), round(beta1,2), round(gamma0,2), round(delta0,2)))     
   plt.title(' $i_0$={:.2f}, $r_0$={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(i0,2),
               round(r0,2), round(beta1,2), round(gamma0,2), round(delta0,5)))     
 
   plt.legend()
   
   plt.savefig('plots/{}_fit_vs_data.png'.format(data_type))
   plt.tight_layout()
   plt.close()

   plt.plot(t, s_true,'--r', label=' S Data', linewidth=lineW);
   plt.plot(t, s ,'-r', label='S fit',linewidth=lineW, alpha=.5);
   plt.plot(t, i_true,'--b', label=' I Data', linewidth=lineW);
   plt.plot(t, i ,'-b', label='I fit',linewidth=lineW, alpha=.5);
   plt.plot(t, v_true,'--b', label=' V Data', linewidth=lineW);
   plt.plot(t, v ,'-b', label='V fit',linewidth=lineW, alpha=.5);   
   plt.plot(t, r_true,'--b', label=' R Data', linewidth=lineW);
   plt.plot(t, r ,'-b', label='R fit',linewidth=lineW, alpha=.5);
   plt.ylabel('Population percentage')  
   plt.xlabel('Days') 
   plt.legend()

   plt.title(' $i_0$={:.2f}, $r_0$={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(i0,2),
               round(r0,2), round(beta1,2), round(gamma0,2), round(delta0,5)))     
 
   plt.legend()
   
   plt.savefig('plots/{}_fit_vs_data_all.png'.format(data_type))
   plt.tight_layout()
   plt.close()
# test the fitted model with a given set of parameters and intial conditions 
def test_fitmodel_groudtruth(model, data_type, time_series_dict, optimized_params, params_fixed, average = 'D', lineW = 3):

   time_sequence = list(time_series_dict.keys())   
   t = torch.Tensor(time_sequence).reshape(-1,1)
   t_0 = time_sequence[0]
   n_test = len(t)
   # s0, a0, i0, v0, r0, alpha1, beta1, gamma0, delta0 = optimized_params
   # s0, i0, v0, r0, alpha1, beta1, gamma0, delta0 = optimized_params
   s0, i0, v0, r0, beta1, gamma0, delta0 = optimized_params

   # a_0 = a0*torch.ones(n_test)
   i_0 = i0*torch.ones(n_test)
   v_0 = v0*torch.ones(n_test)
   r_0 = r0*torch.ones(n_test)
   # alpha_1 = alpha1*torch.ones(n_test)
   beta_1 = beta1*torch.ones(n_test)
   gamma = gamma0*torch.ones(n_test)
   delta = delta0*torch.ones(n_test)
   
   # a_0 = torch.Tensor(a_0).reshape((-1, 1))
   i_0 = torch.Tensor(i_0).reshape((-1, 1))
   v_0 = torch.Tensor(v_0).reshape((-1, 1))
   r_0 = torch.Tensor(r_0).reshape((-1, 1))
   # alpha_1 = torch.Tensor(alpha_1).reshape((-1, 1))
   beta_1 = torch.Tensor(beta_1).reshape((-1, 1))
   gamma = torch.Tensor(gamma).reshape((-1, 1))
   delta = torch.Tensor(delta).reshape((-1, 1))
   
   # s_0 = 1 - a_0 - i_0 - v_0 - r_0
   s_0 = 1 - i_0 - v_0 - r_0

   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]   
   # params_bundle = [alpha_1, beta_1, gamma, delta]
   params_bundle = [beta_1, gamma, delta]
   
   # s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)
   s, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)   
   
   t = t.detach().numpy()
   s = s.detach().numpy()
   # a = a.detach().numpy()
   i = i.detach().numpy() 
   v = v.detach().numpy()
   r = r.detach().numpy() 
   t = t.reshape((n_test,))   
   
   true_values = list(time_series_dict.values())
   
   # s_exact, a_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, a0, i0, v0, r0, alpha1, beta1, gamma0, delta0, params_fixed)
   # s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, i0, v0, r0, alpha1, beta1, gamma0, delta0, params_fixed)
   s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, i0, v0, r0, beta1, gamma0, delta0, params_fixed)

   _, i_true, v_true, _ = map(list, zip(*true_values))
                                                        
   plt.plot(t, i_exact,':r', label=' I exact', linewidth=lineW);
   plt.plot(t, i_true,'--r', label=' I Data', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I fit',linewidth=lineW, alpha=.5);
   
   plt.xlabel('Days') 
   plt.ylabel('Population percentage') 
    
   # plt.title(' $a_0$= {:.2e}, $i_0$={:.2e}, $r_0$={:.2f}, alpha_1={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(a0,2), 
   # plt.title(' $i_0$={:.2e}, $r_0$={:.2f}, alpha_1={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(i0,2), 
   #            round(r0,2), round(alpha1,2), round(beta1,2), round(gamma0,2), round(delta0,2)))     
   plt.title(' $i_0$={:.2e}, $r_0$={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(i0,2), 
              round(r0,2),  round(beta1,2), round(gamma0,2), round(delta0,2)))     

   plt.legend()
   
   plt.savefig('plots/{}_fit_vs_data_vs_groudtruth.png'.format(data_type))
   plt.tight_layout()
   plt.close()


  
# test the solution using one set of random parameters in the bundle   
# def test_snippet(model, epoch, loss, t_0, t_final, params_fixed, i0, v0, r0, alpha_1=0.15, beta_1=0.1,
def test_snippet(model, epoch, loss, t_0, t_final, params_fixed, i0, v0, r0, beta_1=0.1,
                gamma=0.1, delta = 1e-3, n_test = 100, lineW = 3):

   t_test = torch.linspace(t_0, t_final, n_test).reshape(-1,1)
   t_test.requires_grad = True
   
   # s0 = 1 - a0 - i0 - v0 -r0
   s0 = 1 - i0 - v0 -r0   
   # a_0 = a0*torch.ones(n_test).reshape(-1,1)
   i_0 = i0*torch.ones(n_test).reshape(-1,1)
   v_0 = v0*torch.ones(n_test).reshape(-1,1)
   r_0 = r0*torch.ones(n_test).reshape(-1,1)
   # alpha_1s = alpha_1*torch.ones(n_test).reshape(-1,1)
   beta_1s = beta_1*torch.ones(n_test).reshape(-1,1)
   gammas = gamma*torch.ones(n_test).reshape(-1,1)
   deltas = delta*torch.ones(n_test).reshape(-1,1)
   # s_0 = 1 - a_0 - i_0 - v_0 -r_0
   s_0 = 1 - i_0 - v_0 -r_0
   
   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]

   # params_bundle = [alpha_1s, beta_1s, gammas, deltas]
   params_bundle = [beta_1s, gammas, deltas]

   # s, a, i, v, r = model.parametric_solution(t_test, t_0, initial_conditions, params_bundle)
   s, i, v, r = model.parametric_solution(t_test, t_0, initial_conditions, params_bundle)

   t_test = t_test.detach().numpy()
   s = s.detach().numpy()
   # a = a.detach().numpy()
   i = i.detach().numpy() 
   v = v.detach().numpy()
   r = r.detach().numpy()
   t_test = t_test.reshape((n_test,))   
   
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')   
     
   plt.plot(t_test, s ,'-g', label='S Network',linewidth=lineW, alpha=.5);
   # plt.plot(t_test, a ,'-y', label='A Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, i ,'-r', label='I Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, v ,'-b', label='V Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, r ,'-k', label='R Network',linewidth=lineW, alpha=.5);
   plt.legend()  
   # plt.title(' i0={:.2f}, r0={:.2f}, alpha1={:.2f}, beta1={:.2f}, gamma={:.2f}, delta={:.2e}'.format(i0.item(),r0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item())) 
   plt.title(' i0={:.2f}, r0={:.2f}, beta1={:.2f}, gamma={:.2f}, delta={:.2e}'.format(i0.item(),r0.item(), beta_1.item(), gamma.item(), delta.item())) 

   plt.savefig('plots/Checkpoint/Training_Checkpoint_Epoch={:.2e}_Loss={:.2e}.png'.format(epoch, loss))
   plt.close()
   
   plt.xlabel('Days')
   plt.ylabel('Population percentage')
     
     
   # s_exact, a_exact, i_exact, v_exact, r_exact = SAIVR_solution(t_test, s0, a0, i0, v0, r0, alpha_1, beta_1, gamma, delta, params_fixed)
   # s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t_test, s0, i0, v0, r0, alpha_1, beta_1, gamma, delta, params_fixed)
   s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t_test, s0, i0, v0, r0, beta_1, gamma, delta, params_fixed)
     
  # plt.plot(t_test, a ,'-y', label='A Network',linewidth=lineW, alpha=.5);
  # plt.plot(t_test, a_exact ,'--y', label='A GroudTruth',linewidth=lineW, alpha=.5);
   plt.figure(figsize=(15,15))
   plt.subplot(2, 2, 1)
   plt.plot(t_test, s ,'-r', label='S Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, s_exact ,'--r', label='S GroudTruth',linewidth=lineW, alpha=.5);
   plt.legend()
   plt.subplot(2, 2, 2)
   plt.plot(t_test, i ,'-r', label='I Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, i_exact ,'--r', label='I GroudTruth',linewidth=lineW, alpha=.5);
   plt.legend()
   plt.subplot(2, 2, 3)
   plt.plot(t_test, v ,'-r', label='V Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, v_exact ,'--r', label='V GroudTruth',linewidth=lineW, alpha=.5);
   plt.legend()
   plt.subplot(2, 2, 4)
   plt.plot(t_test, r ,'-r', label='R Network',linewidth=lineW, alpha=.5);
   plt.plot(t_test, r_exact ,'--r', label='R GroudTruth',linewidth=lineW, alpha=.5);
   plt.legend()

   # plt.title('a0={:.2e}, i0={:.2e}, r0={:.2f}, alpha1={:.2f}, beta1={:.2f}, gamma={:.2f}, delta={:.2e}'.format(a0.item(),i0.item(),r0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item())) 
   # plt.title('i0={:.2e}, r0={:.2f}, alpha1={:.2f}, beta1={:.2f}, gamma={:.2f}, delta={:.2e}'.format(i0.item(),r0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item())) 
   plt.title('i0={:.2e}, r0={:.2f}, beta1={:.2f}, gamma={:.2f}, delta={:.4e}'.format(i0.item(),r0.item(), beta_1.item(), gamma.item(), delta.item())) 

   plt.savefig('plots/Checkpoint/AI_Zoom={:.2f}_Loss={:.2e}.png'.format(epoch, loss))
   plt.close()



def test_fitmodel_checkpoint(model, epoch, time_series_dict, optimized_params, params_fixed, average = 'D', lineW = 3):
   # test the model with a given set of parameters and intial conditions 
   time_sequence = list(time_series_dict.keys())   
   t = torch.Tensor(time_sequence).reshape(-1,1)
   t_0 = time_sequence[0]
   n_test = len(t)
   # s0, a0, i0, v0, r0, alpha1, beta1, gamma0, delta0 = optimized_params
   # s0, i0, v0, r0, alpha1, beta1, gamma0, delta0 = optimized_params
   s0, i0, v0, r0, beta1, gamma0, delta0 = optimized_params

   # a_0 = a0*torch.ones(n_test)
   i_0 = i0*torch.ones(n_test)
   v_0 = v0*torch.ones(n_test)
   r_0 = r0*torch.ones(n_test)
   # alpha_1 = alpha1*torch.ones(n_test)
   beta_1 = beta1*torch.ones(n_test)
   gamma = gamma0*torch.ones(n_test)
   delta = delta0*torch.ones(n_test)
   
   # a_0 = torch.Tensor(a_0).reshape((-1, 1))
   i_0 = torch.Tensor(i_0).reshape((-1, 1))
   v_0 = torch.Tensor(v_0).reshape((-1, 1))
   r_0 = torch.Tensor(r_0).reshape((-1, 1))
   # alpha_1 = torch.Tensor(alpha_1).reshape((-1, 1))
   beta_1 = torch.Tensor(beta_1).reshape((-1, 1))
   gamma = torch.Tensor(gamma).reshape((-1, 1))
   delta = torch.Tensor(delta).reshape((-1, 1))
   # s_0 = 1 - a_0 - i_0 - v_0 - r_0
   s_0 = 1 - i_0 - v_0 - r_0

   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]   
   # params_bundle = [alpha_1, beta_1, gamma, delta]
   params_bundle = [beta_1, gamma, delta]
   
   # s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)
   s, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)

   t = t.detach().numpy()
   s = s.detach().numpy()
   # a = a.detach().numpy()
   i = i.detach().numpy() 
   v = v.detach().numpy()
   r = r.detach().numpy() 
   t = t.reshape((n_test,))   
   
   true_values = list(time_series_dict.values())
   
   # s_exact, a_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, a0, i0, v0, r0, alpha1, beta1, gamma0, delta0, params_fixed)
   # s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, i0, v0, r0, alpha1, beta1, gamma0, delta0, params_fixed)
   s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, i0, v0, r0, beta1, gamma0, delta0, params_fixed)

   s_true, i_true, v_true, r_true = map(list, zip(*true_values))

   plt.figure(figsize=(15,15))
   plt.subplot(2, 2, 1)
   plt.plot(t, i_exact,':r', label=' I exact', linewidth=lineW);
   plt.plot(t, i_true,'--r', label=' I Data', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()

   plt.subplot(2, 2, 2)
   plt.plot(t, v_exact,':g', label=' V exact', linewidth=lineW);
   plt.plot(t, v_true,'--g', label=' V Data', linewidth=lineW);
   plt.plot(t, v ,'-g', label='V fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()

   plt.subplot(2, 2, 3)
   plt.plot(t, s_exact,':b', label=' S exact', linewidth=lineW);
   plt.plot(t, s_true,'--b', label=' S Data', linewidth=lineW);
   plt.plot(t, s ,'-b', label='S fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()
   
   plt.subplot(2, 2, 4)
   plt.plot(t, r_exact,':y', label=' R exact', linewidth=lineW);
   plt.plot(t, r_true,'--y', label=' R Data', linewidth=lineW);
   plt.plot(t, r ,'-y', label='R fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()
   

   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   # plt.title(' $a_0$= {:.2e}, $i_0$={:.2e}, $r_0$={:.2f}, Alpha_1={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(a0,2), round(i0,2), round(r0,2), round(alpha1,2), round(beta1,2), round(gamma0,2), round(delta0,2)))     

   plt.legend()
   # plt.title(' $s_0$={:.2e}, $i_0$={:.2e}, $v_0$={:.2e}, $r_0$={:.2f}, Alpha_1={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(s0,2),round(i0,2),round(v0,2), round(r0,2), round(alpha1,2), round(beta1,2), round(gamma0,2), round(delta0,2)), loc='center', y=-0.3)     
   plt.title(' $s_0$={:.2e}, $i_0$={:.2e}, $v_0$={:.2e}, $r_0$={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(s0,2),round(i0,2),round(v0,2), round(r0,2), round(beta1,2), round(gamma0,2), round(delta0,2)), loc='center', y=-0.3)     
   
   plt.savefig('plots/Checkpoint/{}_fit_vs_data_vs_groudtruth_vs_odeint.png'.format(epoch))
   plt.tight_layout()
   plt.close()

   plt.figure(figsize=(15,15))
   plt.subplot(2, 2, 1)
   # plt.plot(t, i_exact,':r', label=' I exact', linewidth=lineW);
   plt.plot(t, i_true,'--r', label=' I Data', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()

   plt.subplot(2, 2, 2)
   # plt.plot(t, v_exact,':g', label=' V exact', linewidth=lineW);
   plt.plot(t, v_true,'--g', label=' V Data', linewidth=lineW);
   plt.plot(t, v ,'-g', label='V fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()

   plt.subplot(2, 2, 3)
   # plt.plot(t, s_exact,':b', label=' S exact', linewidth=lineW);
   plt.plot(t, s_true,'--b', label=' S Data', linewidth=lineW);
   plt.plot(t, s ,'-b', label='S fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()
   
   plt.subplot(2, 2, 4)
   # plt.plot(t, r_exact,':y', label=' R exact', linewidth=lineW);
   plt.plot(t, r_true,'--y', label=' R Data', linewidth=lineW);
   plt.plot(t, r ,'-y', label='R fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()
   

   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   # plt.title(' $a_0$= {:.2e}, $i_0$={:.2e}, $r_0$={:.2f}, Alpha_1={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(a0,2), round(i0,2), round(r0,2), round(alpha1,2), round(beta1,2), round(gamma0,2), round(delta0,2)))     

   plt.legend()
   # plt.title(' $s_0$={:.2e}, $i_0$={:.2e}, $v_0$={:.2e}, $r_0$={:.2f}, Alpha_1={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(s0,2),round(i0,2),round(v0,2), round(r0,2), round(alpha1,2), round(beta1,2), round(gamma0,2), round(delta0,2)), loc='center', y=-0.3)     
   plt.title(' $s_0$={:.2e}, $i_0$={:.2e}, $v_0$={:.2e}, $r_0$={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(s0,2),round(i0,2),round(v0,2), round(r0,2), round(beta1,2), round(gamma0,2), round(delta0,2)), loc='center', y=-0.3)     
   
   plt.savefig('plots/Checkpoint/{}_fit_vs_data_vs_groudtruth.png'.format(epoch))
   plt.tight_layout()
   plt.close()

   plt.figure(figsize=(11,11))
   plt.plot(t, i_true,'--r', label=' I Data', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I fit',linewidth=lineW, alpha=.5);
   plt.plot(t, v_true,'--g', label=' V Data', linewidth=lineW);
   plt.plot(t, v ,'-g', label='V fit',linewidth=lineW, alpha=.5);
   plt.plot(t, s_true,'--b', label=' S Data', linewidth=lineW);
   plt.plot(t, s ,'-b', label='S fit',linewidth=lineW, alpha=.5);
   plt.plot(t, r_true,'--y', label=' R Data', linewidth=lineW);
   plt.plot(t, r ,'-y', label='R fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()
   plt.title(' $s_0$={:.2e}, $i_0$={:.2e}, $v_0$={:.2e}, $r_0$={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(s0,2),round(i0,2),round(v0,2), round(r0,2), round(beta1,2), round(gamma0,2), round(delta0,2)), loc='center', y=-0.3)     
   
   plt.savefig('plots/Checkpoint/{}_fit_vs_data_vs_groudtruth_NN.png'.format(epoch))
   plt.tight_layout()
   plt.close()

   plt.figure(figsize=(11,11))
   plt.plot(t, i_exact,'--r', label=' I Data', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I fit',linewidth=lineW, alpha=.5);
   plt.plot(t, v_exact,'--g', label=' V Data', linewidth=lineW);
   plt.plot(t, v ,'-g', label='V fit',linewidth=lineW, alpha=.5);
   plt.plot(t, s_exact,'--b', label=' S Data', linewidth=lineW);
   plt.plot(t, s ,'-b', label='S fit',linewidth=lineW, alpha=.5);
   plt.plot(t, r_exact,'--y', label=' R Data', linewidth=lineW);
   plt.plot(t, r ,'-y', label='R fit',linewidth=lineW, alpha=.5);
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')     
   plt.legend()
   plt.title(' $s_0$={:.2e}, $i_0$={:.2e}, $v_0$={:.2e}, $r_0$={:.2f}, Beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}'.format(round(s0,2),round(i0,2),round(v0,2), round(r0,2), round(beta1,2), round(gamma0,2), round(delta0,2)), loc='center', y=-0.3)     
   
   plt.savefig('plots/Checkpoint/{}_fit_vs_data_vs_groudtruth_odeint.png'.format(epoch))
   plt.tight_layout()
   plt.close()
    
def test_model(model, initial_conditions_set, parameters_bundle, params_fixed, t_0, t_final, lineW = 3, n_test = 1000):
   # test the model with a given set of parameters and intial conditions      
   t = torch.linspace(t_0, t_final, n_test).reshape(-1,1)
   t.requires_grad = True
   
   # alpha_1s, beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:], parameters_bundle[3][:]   
   beta_1s, gammas, deltas = parameters_bundle[0][:], parameters_bundle[1][:], parameters_bundle[2][:]  
   
   # a00 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
   i00 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=1)
   v00 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=1)
   r00 = uniform(initial_conditions_set[2][0], initial_conditions_set[2][1], size=1)
   # alpha_10 = uniform(alpha_1s[0], alpha_1s[1], size=1)
   beta_10 = uniform(beta_1s[0], beta_1s[1], size=1)
   gamma0 = uniform(gammas[0], gammas[1], size=1)
   delta0 = uniform(deltas[0], deltas[1], size=1)
   s00 = 1. - i00 - v00 - r00
   
   # a0 = torch.Tensor([a00])
   i0 = torch.Tensor([i00])
   v0 = torch.Tensor([v00])
   r0 = torch.Tensor([r00])
   # alpha_1 = torch.Tensor([alpha_10])
   beta_1 = torch.Tensor([beta_10])
   gamma = torch.Tensor([gamma0])
   delta = torch.Tensor([delta0])    
   
   s0 = 1 - i0 - v0 -r0
   # a_0 = a0*torch.ones(n_test).reshape(-1,1)
   i_0 = i0*torch.ones(n_test).reshape(-1,1)
   v_0 = v0*torch.ones(n_test).reshape(-1,1)
   r_0 = r0*torch.ones(n_test).reshape(-1,1)
   # alpha_1s = alpha_1*torch.ones(n_test).reshape(-1,1)
   beta_1s = beta_1*torch.ones(n_test).reshape(-1,1)
   gammas = gamma*torch.ones(n_test).reshape(-1,1)
   deltas = delta*torch.ones(n_test).reshape(-1,1)
   s_0 = 1 - i_0 - v_0 -r_0
   
   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]
   # params_bundle = [alpha_1s, beta_1s, gammas, deltas]
   params_bundle = [beta_1s, gammas, deltas]

   # s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)
   s, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)

   t = t.detach().numpy()
   s = s.detach().numpy()
   # a = a.detach().numpy()
   i = i.detach().numpy() 
   v = v.detach().numpy()
   r = r.detach().numpy()
   t = t.reshape((n_test,))   
   
   # s_exact, a_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, float(s00), float(a00), float(i00), float(v00), float(r00), float(alpha_10), float(beta_10), float(gamma0), float(delta0), params_fixed)
   # s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, float(s00), float(i00), float(v00), float(r00), float(alpha_10), float(beta_10), float(gamma0), float(delta0), params_fixed)
   s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, float(s00), float(i00), float(v00), float(r00), float(beta_10), float(gamma0), float(delta0), params_fixed)
                                                     
   plt.plot(t, s_exact,'--g', label=' S Ground Truth', linewidth=lineW);
   plt.plot(t, s ,'-g', label='S Network',linewidth=lineW, alpha=.5);
   # plt.plot(t, a_exact,'--y', label=' A Ground Truth', linewidth=lineW);
   # plt.plot(t, a ,'-y', label='A Network',linewidth=lineW, alpha=.5);
   plt.plot(t, i_exact,'--r', label=' I Ground Truth', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I Network',linewidth=lineW, alpha=.5);
   plt.plot(t, v_exact,'--b', label=' V Ground Truth', linewidth=lineW);
   plt.plot(t, v ,'-b', label='V Network',linewidth=lineW, alpha=.5);
   plt.plot(t, r_exact,'--k', label=' R Ground Truth', linewidth=lineW);
   plt.plot(t, r ,'-k', label='R Network',linewidth=lineW, alpha=.5);
   plt.legend()
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')
   
   # plt.savefig('plots/Unsupervised_vs_GroundTruth_$i_0$={:.2e} $a_0$={:.2e}, alpha_1={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}.png'.format(i0.item(), a0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))
   plt.savefig(r'C:/Users/user1/Documents/2021_2022/Covid19_SIR/NN_approach/MLCD-Machine-Learning-Covid-Dynamics/plots/train/Unsupervised_vs_GroundTruth_i_0={:.2e}, beta_1={:.2f}, gamma={:.2f}, delta={:.2e}.png'.format(i0.item(), beta_1.item(), gamma.item(), delta.item()))
   plt.tight_layout()
   plt.close()

   # plt.plot(t, a_exact,'--y', label=' A Ground Truth', linewidth=lineW);
   # plt.plot(t, a ,'-y', label='A Network',linewidth=lineW, alpha=.5);
   plt.plot(t, i_exact,'--r', label=' I Ground Truth', linewidth=lineW);
   plt.plot(t, i ,'-r', label='I Network',linewidth=lineW, alpha=.5);
   plt.legend()
   plt.xlabel('Days') 
   plt.ylabel('Population percentage')  
   
   # plt.savefig('plots/AI_Unsupervised_vs_GroundTruth_$i_0$={:.2e}, $a_0$={:.2e}, alpha_1={:.2f}, beta_1={:.2f}, $\gamma$={:.2f}, $\delta$={:.2e}.png'.format(i0.item(), a0.item(), alpha_1.item(), beta_1.item(), gamma.item(), delta.item()))
   plt.savefig(r'C:/Users/user1/Documents/2021_2022/Covid19_SIR/NN_approach/MLCD-Machine-Learning-Covid-Dynamics/plots/train/AI_Unsupervised_vs_GroundTruth_i_0={:.2e}, beta_1={:.2f}, gamma={:.2f}, delta={:.2e}.png'.format(i0.item(), beta_1.item(), gamma.item(), delta.item()))   
   plt.tight_layout()
   plt.close()

    
# def test_ground_truth(model, params_fixed, t_0, t_final, a0, i0, v0, r0, alpha_1, beta_1, gamma, delta = 1e-3, n_test = 1000):  
def test_ground_truth(model, params_fixed, t_0, t_final, i0, v0, r0, beta_1, gamma, delta = 1e-3, n_test = 1000):  

   # test the model with a given set of parameters and intial conditions      
   t = torch.linspace(t_0, t_final, n_test).reshape(-1,1)
   t.requires_grad = True
   
   # s0 = 1 - a0 - i0 - v0 -r0
   s0 = 1 - i0 - v0 -r0   
   # a_0 = a0*torch.ones(n_test).reshape(-1,1)
   i_0 = i0*torch.ones(n_test).reshape(-1,1)
   v_0 = v0*torch.ones(n_test).reshape(-1,1)
   r_0 = r0*torch.ones(n_test).reshape(-1,1)
   # alpha_1s = alpha_1*torch.ones(n_test).reshape(-1,1)
   beta_1s = beta_1*torch.ones(n_test).reshape(-1,1)
   gammas = gamma*torch.ones(n_test).reshape(-1,1)
   deltas = delta*torch.ones(n_test).reshape(-1,1)
   # s_0 = 1 - a_0 - i_0 - v_0 -r_0
   s_0 = 1 - i_0 - v_0 -r_0   
   
   # initial_conditions = [s_0, a_0, i_0, v_0, r_0]
   initial_conditions = [s_0, i_0, v_0, r_0]   
   # params_bundle = [alpha_1s, beta_1s, gammas]
   params_bundle = [beta_1s, gammas, deltas]

   # s, a, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)
   s, i, v, r = model.parametric_solution(t, t_0, initial_conditions, params_bundle)   

   t = t.detach().numpy()
   s = s.detach().numpy()
   # a = a.detach().numpy()
   i = i.detach().numpy() 
   v = v.detach().numpy()
   r = r.detach().numpy()
   t = t.reshape((n_test,))   
     
   # s_exact, a_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, a0, i0, v0, r0, alpha_1, beta_1, gamma, params_fixed)
   # s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, i0, v0, r0, alpha_1, beta_1, gamma, params_fixed)
   s_exact, i_exact, v_exact, r_exact = SAIVR_solution(t, s0, i0, v0, r0, beta_1, gamma, params_fixed)

   lineW = 3
   plt.plot(t, s_exact,'--g', label=' S Ground Truth', linewidth=lineW);
   # plt.plot(t, a_exact,'--y', label=' A Ground Truth', linewidth=lineW);
   plt.plot(t, i_exact,'--r', label=' I Ground Truth', linewidth=lineW);
   plt.plot(t, v_exact,'--b', label=' V Ground Truth', linewidth=lineW);
   plt.plot(t, r_exact,'--k', label=' R Ground Truth', linewidth=lineW);
   plt.legend()
   
   plt.savefig('plots/GroundTruth_i0={:.2f},r0={:.2f},beta_1={:.2f},gamma={:.2f}.png'.format(i0, r0, beta_1, gamma))
   plt.tight_layout()
   plt.close()

     


 

