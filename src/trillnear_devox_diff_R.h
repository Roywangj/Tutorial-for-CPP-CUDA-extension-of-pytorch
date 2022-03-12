#include <torch/extension.h>    // 这里到底是 '#include <torch/torch.h>' or '#include <torch/extension.h> '
#include <vector>

void trilinear_devoxelize_diffR(int b, int c, int n, int h, int r2_new, int r3_new,
                          bool training, const float *coords, const float *feat,
                          int *inds, float *wgts, float *outs);
// void trilinear_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
//                                const float *wgts, const float *grad_y,
//                                float *grad_x)
void trilinear_devoxelize_diffR_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x);