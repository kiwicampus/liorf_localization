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
    const float Cxx, Cxy, Cxz, Tx;
    const float Cyx, Cyy, Cyz, Ty;
    const float Czx, Czy, Czz, Tz;

    PointAssociateToMapFunctor(std::vector<float> transformation)
        : Cxx(transformation[0])
        , Cxy(transformation[1])
        , Cxz(transformation[2])
        , Tx(transformation[3])
        , Cyx(transformation[4])
        , Cyy(transformation[5])
        , Cyz(transformation[6])
        , Ty(transformation[7])
        , Czx(transformation[8])
        , Czy(transformation[9])
        , Czz(transformation[10])
        , Tz(transformation[11])
    {
    }

    __device__ float4 operator()(const float4& pi_) const
    {
        float4 po_;
        po_.x = Cxx * pi_.x + Cxy * pi_.y + Cxz * pi_.z + Tx;
        po_.y = Cyx * pi_.x + Cyy * pi_.y + Cyz * pi_.z + Ty;
        po_.z = Czx * pi_.x + Czy * pi_.y + Czz * pi_.z + Tz;
        po_.w = pi_.w;
        return po_;
    }
};

/*! Wrapper to call apply transformation from a cpp file
@param input original vector
@param output transformed vector
*/
void apply_transforms(thrust::device_vector<float4>& input, thrust::device_vector<float4>& output,
                      std::vector<float> transformation);

/*!
    make sure that the distances are less than 1.0
    @param distances neighbors distance vector in a plane vector
    @param offsets indicates where start a point neighbors group
    @param valid_flag (output) indicate which points are valid
 */
void validate_distance(const thrust::device_vector<float>& distance, const thrust::device_vector<int>& offsets,
                       thrust::device_vector<bool>& valid_flag);

// Functor to extract and flatten the (x, y, z) components of the float4 at a given index
/*!
    Get submatrix from selected points
    @param cldDevice points cloud
    @param indices_d index of interest
    @param offset indicates where start the point neighbors group
    @param neighbors neighbors number
    @param output output matrix
*/
void get_points_submatrix(const thrust::device_vector<float4>& cldDevice, const thrust::device_vector<int>& indices_d,
                          int offset, int neighbors, thrust::device_vector<float>& output);
/*!
 */
void solve_linear_system(thrust::device_vector<float>& A, int m, int n, thrust::device_vector<float>& X, int x_index);

void todo_name(const thrust::device_vector<float4>& cldDevice, const thrust::device_vector<int>& indices_d,
               const thrust::host_vector<bool>& valid_flag, const thrust::host_vector<int>& offsets,
               const int neighbors, thrust::device_vector<float>& X0s);

void get_planes_coef(thrust::device_vector<float>& X0s, thrust::device_vector<float4>& Xs,
                     thrust::device_vector<bool>& flagValid, size_t size_x);

void validate_planes(thrust::device_vector<float4> cldDevice, thrust::device_vector<int>& indices_d,
                     thrust::device_vector<int>& offsets, thrust::device_vector<float4>& Xs, int neighbors,
                     thrust::device_vector<bool>& valid_flag);

void compute_coeffs(const thrust::device_vector<float4>& points_Sel, const thrust::device_vector<float4>& points_Ori,
                    const thrust::device_vector<float4>& Xs, thrust::device_vector<bool>& valid_flag,
                    thrust::device_vector<float4>& coeffs);

template <typename T>
void copy_to_host(const thrust::device_vector<T>& vector_d, thrust::host_vector<T>& vector_h);

#endif  // CUDA_UTILS_H_
#endif  // FLANN_USE_CUDA