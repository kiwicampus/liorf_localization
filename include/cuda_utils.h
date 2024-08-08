#ifdef FLANN_USE_CUDA

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cudaflann.hpp"

/*!
    resize a device vector
    @param vector: vector to be resized
    @param v_size new size
*/
void resize_device_vector(thrust::device_vector<float>& vector, size_t v_size);

/*! Functor to apply a transformation in GPU using thrust::transform
*/
struct PointAssociateToMapFunctor
{
    const float* transformation;

    PointAssociateToMapFunctor(float* transformation_) : transformation(transformation_) {}

    __device__ float4 operator()(const float4& pi_) const
    {
        float4 po_;
        po_.x = transformation[0] * pi_.x + transformation[1] * pi_.y + transformation[2] * pi_.z + transformation[3];
        po_.y = transformation[4] * pi_.x + transformation[5] * pi_.y + transformation[6] * pi_.z + transformation[7];
        po_.z = transformation[8] * pi_.x + transformation[9] * pi_.y + transformation[10] * pi_.z + transformation[11];
        po_.w = pi_.w;
        return po_;
    }
};

/*! Wrapper to call apply transformation from a cpp file
@param input original vector
@param output transformed vector
*/
void apply_transforms(thrust::device_vector<float4>& input, thrust::device_vector<float4>& output,
                      float* transformation);

/*!
*/
__global__ void validate_distance_kernel(const float* distance, int* offsets, bool* flag, int indices);

void validate_distance(thrust::device_vector<float>& distance, thrust::device_vector<int>& offsets,
                       thrust::device_vector<bool>& valid_flag);

__global__ void validate_distance_kernel(const float* distance, int* offsets, bool* flag);

void solve_linear_system(thrust::device_vector<float>& A, int m, int n, thrust::device_vector<float>& X, int x_index);

__global__ void assemble_X0s_kernel(const float* X0s, float4* Xs, bool* flag, int iterations);

void get_planes_coef(thrust::device_vector<float>& X0s, thrust::device_vector<float4>& Xs,
                     thrust::device_vector<bool>& flagValid, size_t size_x);

__global__ void validate_planes_kernel(const float4* cloud, const int* pointSearchIndices, const int* offsets,
                                       const float4* Xs, bool* flag, int numIndices, int pointSearchIndSize);

void validate_planes(thrust::device_vector<float4> cldDevice, thrust::device_vector<int>& indices_d,
                     thrust::device_vector<int>& offsets, thrust::device_vector<float4>& Xs, int neighbors,
                     thrust::device_vector<bool>& valid_flag);

__global__ void compute_coeffs_kernel(const float4* points_Sel, const float4* points_Ori, const float4* Xs, bool* flag,
                                      float4* coefs, int indices);

void compute_coeffs(const thrust::device_vector<float4>& points_Sel, const thrust::device_vector<float4>& points_Ori,
                    const thrust::device_vector<float4>& Xs, thrust::device_vector<bool>& valid_flag,
                    thrust::device_vector<float4>& coeffs);

template <typename PointType>
void float42PointType(const thrust::device_vector<float4>& coeffs, std::vector<PointType>& coeffSelSurfVec)
{
    thrust::host_vector<float4> coeffs_h = coeffs;
    std::transform(coeffs_h.begin(), coeffs_h.end(), coeffSelSurfVec.begin(), [](float4 point) -> PointType {
        return {point.x, point.y, point.z, point.w};
    });
}

#endif  // CUDA_UTILS_H_
#endif  // FLANN_USE_CUDA