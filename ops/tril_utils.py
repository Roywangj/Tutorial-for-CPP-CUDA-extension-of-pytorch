import time
import torch
from torch.autograd import Function, Variable

import tri_op_cuda

class Tril_Devox(Function):

    @staticmethod
    def forward(ctx, spatial_shape, bool, coords, densefeatures):
        """
        Args:
            ctx:
            spatial_shape: [w, d, h]
            bool: False (Default)
            coords: the coordinates of points, FloatTensor[b, 3, n]
            densefeatures: FloatTensor[b, c, s], s = w * d * h
        Returns:
        """
        w, d, h = spatial_shape     #       h, w, d: h, w, d of voxel resolution
        voxel2pointfeature, inds, wgts  = tri_op_cuda.forward(0, h, w, d, False, coords, densefeatures)
        return voxel2pointfeature


tri_devox = Tril_Devox.apply

if __name__ == '__main__':
    spatial_shape=[50, 50, 20]
    B = 2
    C = 32
    N = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coors = torch.ones(B, 3, N)
    features = torch.ones(B, C, spatial_shape[0] * spatial_shape[1] * spatial_shape[2]).to(device)
    voxeltopointfeature = tri_devox(spatial_shape, False,coors, features)


    print("===> testing pointMLP ...")

    print(voxeltopointfeature.shape)



