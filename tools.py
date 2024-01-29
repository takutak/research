import torch
from torch import nn
import numpy as np

criteria = nn.MSELoss()

class Forward_Burgers:
    def __init__(self, u):
        u = self.u
    
    X_bc = np.array([-1.0, 1.0])
    bc_sample_size = 200
    t_bc = np.linspace(0, 1.0, bc_sample_size)
    
    # X = -1.0
    X_bc1 = np.zeros((bc_sample_size,2))
    X_bc1[:, 0] = -1.0
    X_bc1[:, 1] = t_bc
    #-> [[-1,0],[-1,0.1],[-1,0.2],...]的な感じになる

    # X = 1.0
    X_bc2 = np.zeros((bc_sample_size,2))
    X_bc2[:, 0] = 1.0
    X_bc2[:, 1] = t_bc

    #境界条件を一つのスタックにしてまとめる。
    X_bc_stack = np.vstack((X_bc1, X_bc2))
    u_bc_stack = np.zeros(X_bc_stack.shape[0])

    X_bc_t = torch.tensor(X_bc_stack, requires_grad=True).float()
    u_bc_t = torch.tensor(u_bc_stack, requires_grad=True).float().unsqueeze(dim=1)

    #ここからは、コロケーションポイント
    x_col = np.linspace(-1, 1, 100)
    t_col = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_col, t_col, indexing='ij')
    x_flat = X.flatten()
    t_flat = T.flatten()

    #サンプリングサイズ
    sampling_size = 5000
    random_idx = np.random.choice(np.arange(x_flat.shape[0]), size= sampling_size)

    #サンプリング
    x_sampled = x_flat[random_idx]
    t_sampled = t_flat[random_idx]
    X_sampled = np.zeros((sampling_size, 2))
    X_sampled[:, 0] = x_sampled
    X_sampled[:, 1] = t_sampled
    
    X_sample_tensor = torch.tensor(X_sampled, requires_grad=True).float()


    def physics_informed_loss(self, x , t, net):
        u = net(x,t)
        u_t = torch.autograd(
            u, t,
            grad_outputs = torch.ones_like(u),
            retain_graph = True,
            create_graph=True,
            allow_unused=True
            )[0]
        u_x = torch.autograd(
            u, x,
            grad_outputs = torch.ones_like(u),
            retain_graph = True,
            create_graph=True,
            allow_unused=True
            )[0]
        u_xx = torch.autograd(
            u_x, x,
            grad_outputs = torch.ones_like(u),
            retain_graph = True,
            create_graph=True,
            allow_unused=True
            )[0]
        
        pinn_loss = u_t + u * u_x - (0.01/np.pi) * u_xx
        zeros_t = torch.zeros(pinn_loss.size)
        pinn_loss_ = criteria(pinn_loss, zeros_t)
        return pinn_loss_
    
    def initial_condition_loss(self, x, t, net, u_ini):
        u = net(x, t)
        ini_condition_loss = criteria(u, u_ini)
        return ini_condition_loss
    
    def boundary_condition_loss(self, x, t, net, u_bc):
        u = net(x, t)
        ini_condition_loss = criteria(u, u_bc)
        return ini_condition_loss
    
    def train_loop(self, num_epochs, u):
        
        for epoch in range(num_epochs):



