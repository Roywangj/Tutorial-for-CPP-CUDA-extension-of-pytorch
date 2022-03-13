#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <vector>
// #include "../cuda_utils.cuh"

/*
  Function: trilinear devoxlization_diffR (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r   : voxel resolution
    r2  : r ** 2
    r3  : r ** 3
    coords : the coordinates of points, FloatTensor[b, 3, n]
    feat   : features, FloatTensor[b, c, r3]
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    outs   : outputs, FloatTensor[b, c, n]

    h   : height of voxel resolution
    w   : width of voxel resolution
    d   : depth of voxel resolution
    feat   : features, FloatTensor[b, c, h * w * d]
    r2_new = h * w
    r3_new = h * w * d

*/

__global__ void trilinear_devoxelize_kernel_diffR(int b, int c, int n, int h, int r2_new, int r3_new, 
                                            bool is_training,
                                            const float *__restrict__ coords,
                                            const float *__restrict__ feat,
                                            int *__restrict__ inds,
                                            float *__restrict__ wgts,
                                            float *__restrict__ outs) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  //  指针指向当前的batch
  coords += batch_index * n * 3;
  //  在不training的状态下，指针会指到错误的位置上，但不对wgts和inds指向的显存做操作
  inds += batch_index * n * 8;
  wgts += batch_index * n * 8;
  //feat += batch_index * c * r3;
  feat += batch_index * c * r3_new; // -------------------wjjjjjjjjjjjjjjjj
  outs += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    //  拿到第i个点的坐标
    float x = coords[i];
    float y = coords[i + n];
    float z = coords[i + n + n];
    //  计算用于插值的第i个点， 也就是(floor(x), floor(y), floor(z))  下取整
    float x_lo_f = floorf(x);
    float y_lo_f = floorf(y);
    float z_lo_f = floorf(z);
  	// 因为每一个grid边长都是1，计算第i个点到周围八个点的距离
  	// 注意，大多数点都是在grid内部的，由于点云的稀疏性，很难出现在grid的分界面上。
    // 但是，由于voxel_coord是经过clamp的，是将(0,r)强行clamp到(0,r-1]的
    // 也就是说大于r-1的点会变成r-1，从而使得最后一个voxel内部的点其实都是在最后一个voxel的分界面上
    // 也就是说，x_d_1等于0的时候，它基本就是在x方向上的第R个voxel内的。
    float x_d_1 = x - x_lo_f; // / (x_hi_f - x_lo_f + 1e-8f)
    float y_d_1 = y - y_lo_f;
    float z_d_1 = z - z_lo_f;
    float x_d_0 = 1.0f - x_d_1;
    float y_d_0 = 1.0f - y_d_1;
    float z_d_0 = 1.0f - z_d_1;

	  // 计算权重
	  // 举例，对于1维度线性插值，f(x) = (ceil(x)-x) * f(floor(x)) + (x-floor(x)) * f(ceil(x))
	  // wgt000 = (ceil(x)-x) * (ceil(y)-y) * (ceil(z)-z)
	  // 所以wgt000对应的是点是(floor(x), floor(y), floor(z))
    float wgt000 = x_d_0 * y_d_0 * z_d_0;
    float wgt001 = x_d_0 * y_d_0 * z_d_1;
    float wgt010 = x_d_0 * y_d_1 * z_d_0;
    float wgt011 = x_d_0 * y_d_1 * z_d_1;
    float wgt100 = x_d_1 * y_d_0 * z_d_0;
    float wgt101 = x_d_1 * y_d_0 * z_d_1;
    float wgt110 = x_d_1 * y_d_1 * z_d_0;
    float wgt111 = x_d_1 * y_d_1 * z_d_1;

    // 计算(floor(x), floor(y), floor(z))的坐标
    int x_lo = static_cast<int>(x_lo_f);
    int y_lo = static_cast<int>(y_lo_f);
    int z_lo = static_cast<int>(z_lo_f);
    // 注意，在最后一个voxel中，x_d_1 == y_d_1 == z_d_1 == 0
    int x_hi = (x_d_1 > 0) ? -1 : 0;
    int y_hi = (y_d_1 > 0) ? -1 : 0;
    // 如果z_d_1==0，说明这个点在边界上，也就没有下一个z_hi对应的feautre了，所以置位1
    int z_hi = (z_d_1 > 0) ? 1 : 0;

/*
    int idx000 = x_lo * r2 + y_lo * r + z_lo;
    int idx001 = idx000 + z_hi;      // x_lo * r2 + y_lo * r + z_hi;
    // 当y_hi==0时，说明y_d_1==0，说明这个点在y方向上的第R个voxel中，所以在y方向上没有下一个voxel了
    // 此时y_hi & r == 0,保证数组不会越界
    // 当当y_hi==1时，说明这个点不在边界上，由于-1的所有位都是1，此时y_hi & r == r
    int idx010 = idx000 + (y_hi & r);  // x_lo * r2 + y_hi * r + z_lo;
    int idx011 = idx010 + z_hi;      // x_lo * r2 + y_hi * r + z_hi;
    int idx100 = idx000 + (x_hi & r2); // x_hi * r2 + y_lo * r + z_lo;
    int idx101 = idx100 + z_hi;      // x_hi * r2 + y_lo * r + z_hi;
    int idx110 = idx100 + (y_hi & r);  // x_hi * r2 + y_hi * r + z_lo;
    int idx111 = idx110 + z_hi;      // x_hi * r2 + y_hi * r + z_hi;
*/

    //  new idx  -------wj
    int idx000 = x_lo * r2_new + y_lo * h + z_lo;
    int idx001 = idx000 + z_hi;      // x_lo * r2_new + y_lo * h + z_hi;
    // 当y_hi==0时，说明y_d_1==0，说明这个点在y方向上的第R个voxel中，所以在y方向上没有下一个voxel了
    // 此时y_hi & h == 0,保证数组不会越界
    // 当当y_hi==1时，说明这个点不在边界上，由于-1的所有位都是1，此时y_hi & h == h
    int idx010 = idx000 + (y_hi & h);  // x_lo * r2_new + y_hi * h + z_lo;
    int idx011 = idx010 + z_hi;      // x_lo * r2_new + y_hi * h + z_hi;
    int idx100 = idx000 + (x_hi & r2_new); // x_hi * r2_new + y_lo * h + z_lo;
    int idx101 = idx100 + z_hi;      // x_hi * r2_new + y_lo * h + z_hi;
    int idx110 = idx100 + (y_hi & h);  // x_hi * r2_new + y_hi * h + z_lo;
    int idx111 = idx110 + z_hi;      // x_hi * r2_new + y_hi * h + z_hi;




    if (is_training) {
      wgts[i] = wgt000;
      wgts[i + n] = wgt001;
      wgts[i + n * 2] = wgt010;
      wgts[i + n * 3] = wgt011;
      wgts[i + n * 4] = wgt100;
      wgts[i + n * 5] = wgt101;
      wgts[i + n * 6] = wgt110;
      wgts[i + n * 7] = wgt111;
      inds[i] = idx000;
      inds[i + n] = idx001;
      inds[i + n * 2] = idx010;
      inds[i + n * 3] = idx011;
      inds[i + n * 4] = idx100;
      inds[i + n * 5] = idx101;
      inds[i + n * 6] = idx110;
      inds[i + n * 7] = idx111;
    }

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3_new;
      outs[j * n + i] =
          wgt000 * feat[jr3 + idx000] + wgt001 * feat[jr3 + idx001] +
          wgt010 * feat[jr3 + idx010] + wgt011 * feat[jr3 + idx011] +
          wgt100 * feat[jr3 + idx100] + wgt101 * feat[jr3 + idx101] +
          wgt110 * feat[jr3 + idx110] + wgt111 * feat[jr3 + idx111];
    }
  }
}



/*
  Function: trilinear devoxlization_diffR (backward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r3  : voxel cube size = voxel resolution ** 3
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    grad_y : grad outputs, FloatTensor[b, c, n]
    grad_x : grad inputs, FloatTensor[b, c, r3]
*/
__global__ void trilinear_devoxelize_grad_kernel_diffR(
    int b, int c, int n, int r3, const int *__restrict__ inds,
    const float *__restrict__ wgts, const float *__restrict__ grad_y,
    float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * n * 8;
  wgts += batch_index * n * 8;
  grad_x += batch_index * c * r3;
  grad_y += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    int idx000 = inds[i];
    int idx001 = inds[i + n];
    int idx010 = inds[i + n * 2];
    int idx011 = inds[i + n * 3];
    int idx100 = inds[i + n * 4];
    int idx101 = inds[i + n * 5];
    int idx110 = inds[i + n * 6];
    int idx111 = inds[i + n * 7];
    float wgt000 = wgts[i];
    float wgt001 = wgts[i + n];
    float wgt010 = wgts[i + n * 2];
    float wgt011 = wgts[i + n * 3];
    float wgt100 = wgts[i + n * 4];
    float wgt101 = wgts[i + n * 5];
    float wgt110 = wgts[i + n * 6];
    float wgt111 = wgts[i + n * 7];

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      float g = grad_y[j * n + i];
      atomicAdd(grad_x + jr3 + idx000, wgt000 * g);
      atomicAdd(grad_x + jr3 + idx001, wgt001 * g);
      atomicAdd(grad_x + jr3 + idx010, wgt010 * g);
      atomicAdd(grad_x + jr3 + idx011, wgt011 * g);
      atomicAdd(grad_x + jr3 + idx100, wgt100 * g);
      atomicAdd(grad_x + jr3 + idx101, wgt101 * g);
      atomicAdd(grad_x + jr3 + idx110, wgt110 * g);
      atomicAdd(grad_x + jr3 + idx111, wgt111 * g);
    }
  }
}



void trilinear_devoxelize_diffR(int b, int c, int n, int h, int r2_new, int r3_new,
                          bool training, const float *coords, const float *feat,
                          int *inds, float *wgts, float *outs) {
  // trilinear_devoxelize_kernel_diffR<<<b, optimal_num_threads(n)>>>(
      // b, c, n, h, r2_new, r3_new, training, coords, feat, inds, wgts, outs);
  trilinear_devoxelize_kernel_diffR<<<b, n>>>(
      b, c, n, h, r2_new, r3_new, training, coords, feat, inds, wgts, outs);
  // CUDA_CHECK_ERRORS();
}


void trilinear_devoxelize_diffR_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x) {
  // trilinear_devoxelize_grad_kernel_diffR<<<b, optimal_num_threads(n)>>>(
  //     b, c, n, r3, inds, wgts, grad_y, grad_x);
  trilinear_devoxelize_grad_kernel_diffR<<<b, n>>>(
      b, c, n, r3, inds, wgts, grad_y, grad_x);
  // CUDA_CHECK_ERRORS();
}