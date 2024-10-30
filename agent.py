import torch
from torch.autograd import Variable
from qpth.qp import QPFunction

class Agent(torch.nn.Module):
    def __init__(self,
                 ref_controller,
                 cbf,
                 dynamics,
                 alpha=torch.nn.Identity(),
                 num_action=2):
        super(Agent, self).__init__()
        self.cbf = cbf
        self.dynamics = dynamics
        self.ref_controller = ref_controller
        self.num_action = num_action
        self.alpha = alpha

    def forward(self,state,detach_cbf=False,detach_controller=False):
        B, S = state.shape
        # B, T, S = state.shape
        Q = Variable(torch.eye(self.num_action))
        Q = Q.unsqueeze(0).unsqueeze(0).expand(B,-1,-1).to(state.device)
        # Q = Q.unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1).to(state.device)
        u_ref = -2*self.ref_controller(state)
        state.requires_grad = True
        h = self.cbf(state)
        delta_h = torch.autograd.grad(h.mean(),state,retain_graph=True)[0]
        f,g = self.dynamics(state)
        left = -torch.einsum("bs,bsa->ba",delta_h,g.view(g.shape[0],g.shape[1],S,self.num_action))
        right = self.alpha(h).squeeze(-1)+torch.einsum("bs,bs->b",delta_h,f)
        # left = -torch.einsum("bts,btsa->bta",delta_h,g.view(g.shape[0],g.shape[1],S,self.num_action))
        # right = self.alpha(h).squeeze(-1)+torch.einsum("bts,bts->bt",delta_h,f)
        if detach_controller:
            u_ref = u_ref.detach()
        e = Variable(torch.Tensor())
        print(Q.shape)
        print(u_ref.shape)
        print(left.shape)
        print(right.shape)
        u = QPFunction(verbose=False)(Q, u_ref, left, right, e, e)
        return u


