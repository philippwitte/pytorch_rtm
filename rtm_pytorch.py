import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from SeisModel import ricker_wavelet, absorbing_boundaries, stencil

class Net(nn.Module):

    def __init__(self, nx, nz, dt, nt, m, xrec=None, zrec=None, order=4):
        super(Net, self).__init__()

        # Laplacian
        self.laplace = nn.Conv2d(1, 1, (order+1, order+1), padding=(int(order/2), int(order/2)), bias=False)
        self.laplace.requires_grad_(False)
        self.nx = nx
        self.nz = nz
        self.dt = dt
        self.nt = nt
        self.m = m
        if xrec is not None and zrec is not None:
            self.xrec = xrec
            self.zrec = zrec
            self.nrec = len(self.xrec)

    def forward(self, q, sx, sz):

        # Define wavefields
        p_prev = torch.zeros(1, 1, self.nx, self.nz, requires_grad=False, dtype=torch.float32)
        p_curr = torch.zeros(1, 1, self.nx, self.nz, requires_grad=False, dtype=torch.float32)
        q_curr = torch.zeros(1, 1, self.nx, self.nz, requires_grad=False, dtype=torch.float32)
        damp = absorbing_boundaries(nx, nz)
        if hasattr(self, 'xrec'):
            d_pred = torch.zeros(1, 1, self.nt, self.nrec, dtype=torch.float32)

        for j in range(self.nt):

            # Inject source
            q_curr[0, 0, sx, sz] = q[j]

            # Propagate
            p_new = damp*(self.dt**2/self.m*self.laplace(p_curr) + 2.0*p_curr - damp*p_prev + q_curr)
            p_prev = p_curr
            p_curr = p_new

            # Sample shot record if receiver coordinates were passed to network
            if hasattr(self, 'xrec'):
                d_pred[0, 0, j, :] = p_curr[0, 0, self.xrec, self.zrec]

        if hasattr(self, 'xrec'):
            return d_pred, p_curr
        else:
            return p_curr


#########################################################################################

# Velocity model
nx = 100
nz = 100
d = 10.0    # grid spacing
v = torch.ones(nx, nz, dtype=torch.float32)*1500.0
v[:,51:] = 3000.0
v0 = torch.ones(nx, nz, dtype=torch.float32)*1500.0
m = (1/v)**2
m.requires_grad = False
m0 = (1/v0)**2
m0.requires_grad = True


# Source wavelet
nt = 1000    # no. of time steps
dt = torch.tensor(0.001)    # time stepping interval
f0 = 15     # peak frequency in Hz
q = ricker_wavelet(nt, dt, f0)
sx = 50 # source x and z indices
sz = 30

# Define receivers (as indices)
xrec = torch.arange(10,90,1, dtype=torch.long)
nrec = len(xrec)
zrec = torch.ones(nrec, dtype=torch.long)*30

# Build network
space_order = 4
A_inv = Net(nx, nz, dt, nt, m, xrec, zrec, order=space_order)
A0_inv = Net(nx, nz, dt, nt, m0, xrec, zrec, order=space_order)

# Initialize weights w/ FD stencil
A_inv.laplace.weight.data[0,0,:,:] = stencil(d, order=space_order)
A_inv.laplace.weight.requires_grad = False
A0_inv.laplace.weight.data[0,0,:,:] = stencil(d, order=space_order)
A0_inv.laplace.weight.requires_grad = False

# Forward model
d_obs = A_inv(q, sx, sz)[0]
d_pred = A0_inv(q, sx, sz)[0]
plt.figure(); plt.imshow(d_obs.data[0,0,:,:], vmin=-5, vmax=5, cmap="seismic", aspect='auto')
plt.figure(); plt.imshow(d_pred.data[0,0,:,:], vmin=-5, vmax=5, cmap="seismic", aspect='auto')

criterion = nn.MSELoss()
loss = criterion(d_pred, d_obs)
loss.backward()
plt.figure(); plt.imshow(np.transpose(m0.grad), vmin=-4e2, vmax=4e2)
plt.show()
