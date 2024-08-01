#ifdef FLANN_USE_CUDA

#include "cudaflann.hpp"

void apply_transforms(thrust::device_vector<float4>& input, thrust::device_vector<float4>& output,
                      Eigen::Affine3f& transformation)
{
    output.resize(input.size());
    thrust::transform(input.begin(), input.end(), output.begin(), PointAssociateToMapFunctor(transformation));
}

#endif