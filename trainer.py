import lightning as L
import torch.nn.functional as F
import torch

class ICRLCBFTrainer(L.LightningModule):
    def __init__(self,
                 agent,
                 alpha=torch.nn.Identity(),
                 epoch_dynamics=10,
                 epoch_cbf=10,
                 eps=1e-5,
                 regularizer_coeff=0.1,
                 lr=1e-3,
                 weight_decay=5e-4,
                 **kwargs):
        super(ICRLCBFTrainer, self).__init__()
        self.epoch_dynamics = epoch_dynamics
        self.epoch_cbf = epoch_cbf
        self.agent = agent
        self.alpha = alpha
        self.eps = eps
        self.lr = lr
        self.weight_decay = weight_decay
        self.regularizer_coeff = regularizer_coeff
        self.train_dynamics_only = True
        self.train_controller = False
        self.save_hyperparameters(ignore='agent')

    def on_train_epoch_start(self):
        if self.current_epoch == self.epoch_dynamics:
            for n in self.agent.dynamics.parameters():
                n.require_grads = False
            self.train_dynamics_only = False
        if (self.current_epoch - self.epoch_dynamics) % 2 == 0:
            self.train_controller = False
        else:
            self.train_controller = True

    def compute_value(self,state,action):
        next_state = self.agent.dynamics.forward_prop(state,action)
        h = self.agent.cbf(state)
        h_dot = self.agent.cbf.compute_h_dot(state,next_state)
        return h + self.alpha(h_dot)

    def training_step(self, batch, batch_idx):
        state, next_state, action = batch

        if self.train_dynamics_only:
            pred_state = self.agent.dynamics.forward_prop(state,action)
            dynamics_loss = F.mse_loss(pred_state,next_state)
            self.log_dict({"train_dynamics_loss": dynamics_loss})
            return dynamics_loss
        if self.train_controller:
            pred_action = self.agent(state,detach_cbf=True)
            controller_loss = F.mse_loss(pred_action,action)
            self.log_dict({"train_controller_loss": controller_loss})
            return controller_loss
        else:
            pred_action = self.agent(state,detach_controller=True)
            expert_value = self.compute_value(state,action)
            nominal_value = self.compute_value(state,pred_action)
            expert_loss = torch.log(expert_value + self.eps).mean()
            nominal_loss = torch.log(nominal_value + self.eps).mean()
            barrier_loss = -expert_loss + nominal_loss
            mean_expert_value = torch.mean(1-expert_value)
            mean_nominal_value = torch.mean(1-nominal_value)
            regularizer_loss = self.regularizer_coeff * (mean_expert_value + mean_nominal_value)
            loss = barrier_loss + regularizer_loss
            self.log_dict({"train_barrier_loss": barrier_loss})
            self.log_dict({"train_regularizer_loss": regularizer_loss})
            self.log_dict({"train_expert_value": mean_expert_value})
            self.log_dict({"train_nominal_value": mean_nominal_value})
            self.log_dict({"train_loss": loss})
            return loss
        
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        state, next_state, action = batch

        if self.current_epoch < self.epoch_dynamics:
            pred_state = self.agent.dynamics.forward_prop(state,action)
            dynamics_loss = F.mse_loss(pred_state,next_state)
            self.log_dict({"val_dynamics_loss": dynamics_loss})
        else:
            pred_action = self.agent(state,detach_cbf=True,detach_controller=True)
            controller_loss = F.mse_loss(pred_action,action)
            expert_value = self.compute_value(state,action)
            nominal_value = self.compute_value(state,pred_action)
            expert_loss = torch.log(expert_value + self.eps).mean()
            nominal_loss = torch.log(nominal_value + self.eps).mean()
            barrier_loss = -expert_loss + nominal_loss
            mean_expert_value = torch.mean(1-expert_value)
            mean_nominal_value = torch.mean(1-nominal_value)
            regularizer_loss = self.regularizer_coeff * (mean_expert_value + mean_nominal_value)
            loss = barrier_loss + regularizer_loss
            self.log_dict({"val_controller_loss": controller_loss})
            self.log_dict({"val_barrier_loss": barrier_loss})
            self.log_dict({"val_regularizer_loss": regularizer_loss})
            self.log_dict({"val_expert_value": mean_expert_value})
            self.log_dict({"val_nominal_value": mean_nominal_value})
            self.log_dict({"val_loss": loss})
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.agent.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer