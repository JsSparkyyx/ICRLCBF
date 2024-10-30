import torch
from torch.distributions import Normal
from torch.distributions.independent import Independent
from utils import build_mlp

class CBF(torch.nn.Module):
    def __init__(self,state_dim=72,
                      hidden_dim=64,
                      dt=0.1,
                      num_layer=3):
        super(CBF, self).__init__()
        hidden_dims = [state_dim]
        for _ in range(num_layer-1):
            hidden_dims.append(hidden_dim)
        hidden_dims.append(1)
        self.cbf = build_mlp(hidden_dims)
        self.activation = torch.nn.Tanh()
        self.dt = dt
    
    def forward(self,state):
        return self.activation(self.cbf(state))
    
    def compute_h_dot(self,state,next_state):
        return (self.forward(next_state)-self.forward(state))/self.dt

class NeuralDynamics(torch.nn.Module):
    def __init__(self,num_action=2,
                      state_dim=72,
                      hidden_dim=64,
                      dt=0.1,
                      num_layer=3):
        super(NeuralDynamics, self).__init__()
        hidden_dims = [state_dim]
        for _ in range(num_layer-1):
            hidden_dims.append(hidden_dim)
        hidden_dims.append(state_dim)
        self.f = build_mlp(hidden_dims)
        hidden_dims[-1] = state_dim*num_action
        self.g = build_mlp(hidden_dims)
        self.num_action = num_action
        self.state_dim = state_dim
        self.dt = dt
    
    def forward(self,state):
        return self.f(state), self.g(state)

    def forward_prop(self,state,action):
        f,g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs',g.view(g.shape[0],self.state_dim,self.num_action),action)
        # gu = torch.einsum('btsa,bta->bts',g.view(g.shape[0],g.shape[1],self.state_dim,self.num_action),action)
        state_dot = f + gu
        state_nom = state + state_dot*self.dt
        return state_nom

class NominalDynamics(torch.nn.Module):
    def __init__(self,num_action=2,
                      state_dim=72,
                      hidden_dim=64,
                      dt=0.1,
                      num_layer=3):
        super(NominalDynamics, self).__init__()
        hidden_dims = [state_dim]
        for _ in range(num_layer-1):
            hidden_dims.append(hidden_dim)
        hidden_dims.append(state_dim)
        self.f = build_mlp(hidden_dims)
        hidden_dims[-1] = state_dim*num_action
        self.g = build_mlp(hidden_dims)
        self.num_action = num_action
        self.state_dim = state_dim
        self.dt = dt
    
    def forward(self,state):
        return self.f(state), self.g(state)

    def forward_prop(self,state,action,next_state):
        f,g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs',g.view(g.shape[0],self.state_dim,self.num_action),action)
        # gu = torch.einsum('btsa,bta->bts',g.view(g.shape[0],g.shape[1],self.state_dim,self.num_action),action)
        state_dot = f + gu
        state_nom = state + state_dot*self.dt
        return state_nom + (next_state-state_nom).detach()

class GaussianMLP(torch.nn.Module):
    def __init__(self,
                 action_dim=2,
                 state_shape=72,
                 hidden_dim=64,
                #  units=(8, 8),
                 hidden_nonlinearity=torch.nn.Tanh,
                 w_init=torch.nn.init.xavier_normal_,
                 b_init=torch.nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1.e-6,
                 max_std=None,
                 num_layer=3,
                 std_parameterization='exp',
                 normal_distribution_cls=Normal):
        super(GaussianMLP, self).__init__()

        self._state_shape = state_shape
        self._action_dim = action_dim
        self._std_parameterization = std_parameterization
        self._norm_dist_class = normal_distribution_cls

        self._mean_module = torch.nn.Sequential()
        units = [hidden_dim for _ in range(num_layer-1)]
        in_dim = state_shape
        for idx, unit in enumerate(units):
            linear_layer = torch.nn.Linear(in_dim, unit)
            w_init(linear_layer.weight)
            b_init(linear_layer.bias)

            self._mean_module.add_module(f'linear_{idx}', linear_layer)
            if hidden_nonlinearity:
                self._mean_module.add_module('non_linear_{idx}',
                                             hidden_nonlinearity())

            in_dim = unit

        linear_layer = torch.nn.Linear(in_dim, action_dim)
        w_init(linear_layer.weight)
        b_init(linear_layer.bias)
        self._mean_module.add_module('out', linear_layer)

        init_std_param = torch.Tensor([init_std]).log()
        if learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()

    def _get_mean_and_log_std(self, inputs):
        # assert len(inputs) == 1
        mean = self._mean_module(inputs)

        broadcast_shape = list(inputs[0].shape[:-1]) + [self._action_dim]
        uncentered_log_std = torch.zeros(broadcast_shape).to(inputs.device) + self._init_std

        return mean, uncentered_log_std

    def forward(self, inputs):
        mean, log_std_uncentered = self._get_mean_and_log_std(inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._norm_dist_class(mean, std)

        # Makes it so that a sample from the distribution is treated as a
        # single sample and not dist.batch_shape samples.
        dist = Independent(dist, 1)
        return dist.sample()
