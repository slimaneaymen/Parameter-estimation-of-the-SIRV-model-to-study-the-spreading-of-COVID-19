import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader


class saivrNet(torch.nn.Module):
    def __init__(self, input_dim=8, layers=10, hidden=48, output=4, activation=None):
       
        super(saivrNet, self).__init__()
        #activation function
        if activation is None:
            self.actF = torch.nn.Sigmoid()
        elif activation == 'sin':
            self.actF = mySin()
        else:
            self.actF = getattr(torch.nn, activation)()

        self.fca = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            self.actF
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            *[self.fca for _ in range(layers)],
            torch.nn.Linear(hidden, output)
        )

    def forward(self, x):
        x = self.ffn(x)
        s_N = (x[:, 0]).reshape(-1, 1)
        # a_N = (x[:, 1]).reshape(-1, 1)
        i_N = (x[:, 1]).reshape(-1, 1)
        v_N = (x[:, 2]).reshape(-1, 1)
        r_N = (x[:, 3]).reshape(-1, 1)
        # return s_N, a_N, i_N, v_N, r_N
        return s_N, i_N, v_N, r_N

    def parametric_solution(self, t_tensor, t_0, initial_conditions, param_bundle):
        # s_0, a_0, i_0, v_0, r_0 = initial_conditions[0][:], initial_conditions[1][:], initial_conditions[2][:], initial_conditions[3][:] ,initial_conditions[4][:] 
        s_0, i_0, v_0, r_0 = initial_conditions[0][:], initial_conditions[1][:], initial_conditions[2][:], initial_conditions[3][:] 
 
        # alpha_1, beta_1, gamma, delta = param_bundle[0][:], param_bundle[1][:], param_bundle[2][:], param_bundle[3][:]
        # alpha, beta, gamma, delta = param_bundle[0][:], param_bundle[1][:], param_bundle[2][:], param_bundle[3][:]        
        beta, gamma, delta = param_bundle[0][:], param_bundle[1][:], param_bundle[2][:]  

        dt = t_tensor - t_0
        f = (1 - torch.exp(-0.01*dt)) 
        t_bundle = torch.cat([t_tensor, i_0, v_0, r_0, beta, gamma, delta], dim=1)        
        N = self.forward(t_bundle)
        N0, N1, N2, N3 = N
        # concatenate to go into a softmax layer
        to_softmax = torch.cat([N0, N1, N2, N3], dim=1)
        softmax_output = softmax(to_softmax, dim=1)
        N0, N1, N2, N3 = softmax_output[:, 0], softmax_output[:, 1], softmax_output[:, 2], softmax_output[:, 3]
        N0, N1, N2, N3 = N0.reshape(-1, 1), N1.reshape(-1, 1), N2.reshape(-1, 1), N3.reshape(-1, 1)
        s_hat = (s_0 + 1.3*f * (N0 - 1.5*s_0))
        # a_hat = (a_0 + f * (N1 - a_0))
        i_hat = (i_0 + 1.2 *f * (N1 - i_0))
        # v_hat = (v_0 + f * (N2 - v_0))
        v_hat = (v_0 + 1.3*f * ((4*N2 - v_0)))
        r_hat = (r_0 + 1.3*f * (N3 - r_0))

        # return s_hat, a_hat, i_hat, v_hat, r_hat
        return s_hat, i_hat, v_hat, r_hat


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

