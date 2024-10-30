from networks import CBF,NeuralDynamics,GaussianMLP
from agent import Agent
from trainer import ICRLCBFTrainer
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BatchSizeFinder
from init_parameters import init_parameters
from data import SafeGymnasiumDataset
import wandb
import os

def main(args):
    wandb.login(key="7893bf6676aaa0213e6da2edbc8f4b42fa816084")
    logger = WandbLogger(project=args["name"])
    data = SafeGymnasiumDataset(**args, pin_memory=True)
    cbf = CBF(hidden_dim=args['hidden_dim'])
    dynamics = NeuralDynamics(hidden_dim=args['hidden_dim'])
    ref_controller = GaussianMLP(hidden_dim=args['hidden_dim'])
    agent = Agent(ref_controller,cbf,dynamics)
    trainer = ICRLCBFTrainer(agent,**args)
    runner = Trainer(logger=logger,
                 callbacks=[
                     ModelCheckpoint(save_top_k=0, 
                                     dirpath =os.path.join(args["save_path"], "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                     BatchSizeFinder(mode="binsearch")
                 ],
                 max_epochs=args['epoch_cbf']+args['epoch_dynamics'])
    runner.fit(trainer, datamodule=data)

if __name__ == '__main__':
    args = init_parameters()
    seed_everything(args['seed'])
    main(args)