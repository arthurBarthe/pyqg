import numpy as np
from matplotlib import pyplot as plt
import pyqg
import torch
import logging
import mlflow
import sys
sys.path.append('/home/ag7531/code/')
from subgrid.models.utils import load_model_cls
from subgrid.utils import select_experiment, select_run

class Parameterization:
    """Defines a parameterization of subgrid momentum forcing bases on a
    trained neural network. To be used within an object of type
    WaterModelWithDLParameterization."""
    def __init__(self, nn, device, mult_factor: float = 1.,
                 every: int = 4, every_noise: int = 4, force_zero_sum: bool =
                 False):
        self.nn = nn.to(device=device)
        self.device = device
        self.means = dict(s_x=None, s_y=None)
        self.betas = dict(s_x=None, s_y=None)
        self.mult_factor = mult_factor
        self.every = every
        self.every_noise = every_noise
        self.force_zero_sum = force_zero_sum
        self.counter_0 = 0
        self.counter_1 = 0

    def __call__(self, u, v):
        """Return the two components of the forcing given the coarse
        velocities. The velocities are expected so sit on the same grid
        points. The returned forcing also sits on those grid points."""
        # Scaling required by the nn
        u *= 10
        v *= 10
        if self.counter_0 == 0:
            # Update calculated mean and std of conditional forcing
            with torch.no_grad():
                # Convert to tensor, puts on selected device
                u = torch.tensor(u, device=self.device).unsqueeze(dim=0).float()
                v = torch.tensor(v, device=self.device).unsqueeze(dim=0).float()
                input_tensor = torch.stack((u, v), dim=1)
                output_tensor = self.nn.forward(input_tensor)
                mean_sx, mean_sy, beta_sx, beta_sy = torch.split(output_tensor,
                                                                 1, dim=1)
                mean_sx = mean_sx.cpu().numpy().squeeze()
                mean_sy = mean_sy.cpu().numpy().squeeze()
                beta_sx = beta_sx.cpu().numpy().squeeze()
                beta_sy = beta_sy.cpu().numpy().squeeze()
                self.apply_mult_factor(mean_sx, mean_sy, beta_sx, beta_sy)
                self.means['s_x'] = mean_sx
                self.means['s_y'] = mean_sy
                self.betas['s_x'] = beta_sx
                self.betas['s_y'] = beta_sy
        else:
            # Use previously computed values
            mean_sx = self.means['s_x']
            mean_sy = self.means['s_y']
            beta_sx = self.betas['s_x']
            beta_sy = self.betas['s_y']
        if self.counter_1 == 0:
            # Update noise
            self.epsilon_x = np.random.randn(*mean_sx.shape)
            self.epsilon_y = np.random.randn(*mean_sy.shape)
        self.s_x = mean_sx + self.epsilon_x
        self.s_y = mean_sy + self.epsilon_y
        if self.force_zero_sum:
            self.s_x = self.force_zero_sum(self.s_x, mean_sx, 1 / beta_sx)
            self.s_y = self.force_zero_sum(self.s_y, mean_sy, 1 / beta_sy)
        # Scaling required by nn
        self.s_x *= 1e-7
        self.s_y *= 1e-7
        # Update the two counters
        self.counter_0 += 1
        self.counter_1 += 1
        self.counter_0 %= self.every
        self.counter_1 %= self.every_noise
        # Return forcing
        return self.s_x, self.s_y

    @staticmethod
    def force_zero_sum(data, mean, std):
        sum_ = np.sum(data)
        sum_std = np.sum(std)
        data = data - sum_ * std / sum_std
        return data

    def apply_mult_factor(self, *args):
        for a in args:
            a *= self.mult_factor

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the neural network used as parameterization
# Prompts the user to select a trained model to be used as parameterization
models_experiment_id, _ = select_experiment()
cols = ['metrics.test loss', 'start_time', 'params.time_indices',
        'params.model_cls_name', 'params.source.run_id', 'params.submodel']
model_run = select_run(sort_by='start_time', cols=cols,
                       experiment_ids=[models_experiment_id, ])
model_module_name = model_run['params.model_module_name']
model_cls_name = model_run['params.model_cls_name']
logging.info('Creating the neural network model')
model_cls = load_model_cls(model_module_name, model_cls_name)

# Load the model's file
client = mlflow.tracking.MlflowClient()
model_file = client.download_artifacts(model_run.run_id,
                                       'models/trained_model.pth')
net = model_cls(2, 4, padding='same')

# Load parameters of pre-trained model
logging.info('Loading the neural net parameters')
net.cpu()
net.load_state_dict(torch.load(model_file))
print('*******************')
print(net)
print('*******************')

parameterization = Parameterization(net, device)

m = pyqg.QGModel(tavestart=0,  dt=8000, nx=64//4, ny=64//4)
m.parameterization = parameterization

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=1000*m.dt):
    plt.clf()
    plt.imshow(m.q[0] + m.Qy1 * m.y)
    plt.clim([0,  m.Qy1 * m.W])
    plt.pause(0.01)
    plt.draw()
    
# now the model is done
