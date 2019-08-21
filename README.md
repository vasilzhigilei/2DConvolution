2DConvolution
===============
CUDA implementation of a 2D convolution algorithm

The current version uses input stored in texture memory processed by the kernel using a constant memory filter.
Output is stored in a 1D output array in global memory, before being copied back to host where sample results are printed to console.

History
-------

#### Version 1.0 (21 August, 2019)
* Initial commit of a 2D convolution GPU implementation with test code
