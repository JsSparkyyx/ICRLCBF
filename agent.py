import torch
from torch.autograd import Variable
from qpth.qp import QPFunction

class Agent(torch.nn.Module):
    def __init__(self,
                 env,
                 ref_controller,
                 cbf,dynamics,
                 alpha=torch.nn.Identity(),
                 num_action=2):
        self.env = env
        self.cbf = cbf
        self.dynamics = dynamics
        self.controller = ref_controller
        self.num_action = num_action
        self.alpha = alpha

    def forward(self,state,detach_cbf=False,detach_controller=False):
        B, T, S = state.shape
        Q = Variable(torch.eye(self.num_action))
        Q = Q.unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1).to(state.device)
        u_ref = -2*self.ref_controller(state)
        h = self.cbf(state)
        delta_h = torch.autograd.grad(h,state,retain_graph=True)[0]
        f,g = self.dynamics(state)
        left = -torch.einsum("bts,btsa->bta",delta_h,g.view(g.shape[0],g.shape[1],self.state_dim,self.num_action))
        right = self.alpha(h)+torch.einsum("bts,bts->bt",delta_h,f)
        if detach_controller:
            u_ref = u_ref.detach()
        e = Variable(torch.Tensor())
        u = QPFunction(verbose=False)(Q, u_ref, left, right, e, e)
        return u


