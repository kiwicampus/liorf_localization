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

/*! Wrapper to call apply transformation from a cpp file
    @param input original vector
    @param output transformed vector
    @param transformation matrix as a flat array
*/
void apply_transforms(const thrust::device_vector<float4>& input, const std::vector<float>& transformation,
                      thrust::device_vector<float4>& output);

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
    Solve mysterious equation of surf optimization
    @param A points as matrix in a flat array
    @param m number of rows
    @param n number of columns
    @param X output vector
    @param x_index ofset of where save the result

 */
void solve_linear_system(thrust::device_vector<float>& A, int m, int n, thrust::device_vector<float>& X, int x_index);

void todo_name(const thrust::device_vector<float4>& cldDevice, const thrust::device_vector<int>& indices_d,
               const thrust::host_vector<bool>& valid_flag, const thrust::host_vector<int>& offsets,
               const int neighbors, thrust::device_vector<float>& X0s);

void get_planes_coef(thrust::device_vector<float>& X0s, thrust::device_vector<float4>& Xs,
                     thrust::device_vector<bool>& flagValid, size_t size_x);

void validate_planes(const thrust::device_vector<float4>& cldDevice, const thrust::device_vector<int>& indices_d,
                     const thrust::device_vector<int>& offsets, const thrust::device_vector<float4>& Xs, int neighbors,
                     thrust::device_vector<bool>& valid_flag);

void compute_coeffs(const thrust::device_vector<float4>& points_Sel, const thrust::device_vector<float4>& points_Ori,
                    const thrust::device_vector<float4>& Xs, thrust::device_vector<bool>& valid_flag,
                    thrust::device_vector<float4>& coeffs);

template <typename T>
void copy_to_host(const thrust::device_vector<T>& vector_d, thrust::host_vector<T>& vector_h);

#endif  // CUDA_UTILS_H_
#endif  // FLANN_USE_CUDA