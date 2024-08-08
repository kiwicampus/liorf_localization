#ifdef FLANN_USE_CUDA

#include "cuda_utils.h"

void resize_device_vector(thrust::device_vector<float>& vector, size_t v_size) { vector.resize(v_size); }

void apply_transforms(thrust::device_vector<float4>& input, thrust::device_vector<float4>& output,
                      float* transformation)
{
    output.resize(input.size());
    thrust::transform(input.begin(), input.end(), output.begin(), PointAssociateToMapFunctor(transformation));
}

__global__ void validate_distance_kernel(const float* distance, int* offsets, bool* flag, int indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= indices) return;

    flag[idx] = distance[offsets[idx] + 4];
}

void validate_distance(thrust::device_vector<float>& distance, thrust::device_vector<int>& offsets,
                       thrust::device_vector<bool>& valid_flag)
{
    valid_flag.resize(offsets.size());
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (offsets.size() + threadsPerBlock - 1) / threadsPerBlock;
    validate_distance_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(distance.data()), thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(valid_flag.data()), offsets.size());
    cudaDeviceSynchronize();
}

// Helper function to solve linear system using cuSOLVER
void solve_linear_system(thrust::device_vector<float>& A, int m, int n, thrust::device_vector<float>& X, int x_index)
{
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    thrust::device_vector<float> B(m, -1.0f);  // 5x1 matrix

    int lda = m;
    int ldb = m;
    int work_size = 0;
    int* d_info;
    float* d_work;

    thrust::device_vector<int> info(1);
    d_info = thrust::raw_pointer_cast(info.data());

    auto A_raw = thrust::raw_pointer_cast(A.data());
    auto B_raw = thrust::raw_pointer_cast(B.data());
    // Query working space of geqrf and ormqr
    cusolverDnSgeqrf_bufferSize(cusolverH, m, n, A_raw, lda, &work_size);
    cudaMalloc((void**)&d_work, sizeof(float) * work_size);

    // QR factorization
    cusolverDnSgeqrf(cusolverH, m, n, A_raw, lda, B_raw, d_work, work_size, d_info);

    // Solve
    cusolverDnSormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 1, n, A_raw, lda, B_raw, B_raw, ldb, d_work,
                     work_size, d_info);

    // Copy solution to X
    thrust::copy(B.begin(), B.begin() + n, X.begin() + x_index);

    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}

__global__ void assemble_X0s_kernel(const float* X0s, float4* Xs, bool* flag, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iterations) return;
    if (!flag[idx]) return;

    float pa = X0s[3 * idx];
    float pb = X0s[3 * idx + 1];
    float pc = X0s[3 * idx + 2];
    float ps = sqrt(pa * pa + pb * pb + pc * pc);

    pa /= ps;
    pb /= ps;
    pc /= ps;
    float pd = 1.0f / ps;
    Xs[idx] = make_float4(pa, pb, pc, pd);
}

void get_planes_coef(thrust::device_vector<float>& X0s, thrust::device_vector<float4>& Xs,
                     thrust::device_vector<bool>& flagValid, size_t size_x)
{
    Xs.resize(size_x);
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size_x + threadsPerBlock - 1) / threadsPerBlock;
    assemble_X0s_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(X0s.data()),
                                                            thrust::raw_pointer_cast(Xs.data()),
                                                            thrust::raw_pointer_cast(flagValid.data()), size_x);
    cudaDeviceSynchronize();  // Ensure kernel completion
}

__global__ void validate_planes_kernel(const float4* cloud, const int* pointSearchIndices, const int* offsets,
                                       const float4* Xs, bool* flag, int numIndices, int pointSearchIndSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIndices) return;
    if (!flag[idx]) return;

    float pa = Xs[idx].x;
    float pb = Xs[idx].y;
    float pc = Xs[idx].z;
    float pd = Xs[idx].w;

    bool isValid = true;
    for (int j = 0; j < pointSearchIndSize; j++)
    {
        int pointIdx = pointSearchIndices[offsets[idx] + j];
        float4 point = cloud[pointIdx];
        if (fabs(pa * point.x + pb * point.y + pc * point.z + pd) > 0.2)
        {
            isValid = false;
            break;
        }
    }
    flag[idx] = isValid;
}

void validate_planes(thrust::device_vector<float4> cldDevice, thrust::device_vector<int>& indices_d,
                     thrust::device_vector<int>& offsets, thrust::device_vector<float4>& Xs, int neighbors,
                     thrust::device_vector<bool>& valid_flag)
{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (Xs.size() + threadsPerBlock - 1) / threadsPerBlock;
    validate_planes_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(cldDevice.data()), thrust::raw_pointer_cast(indices_d.data()),
        thrust::raw_pointer_cast(offsets.data()), thrust::raw_pointer_cast(Xs.data()),
        thrust::raw_pointer_cast(valid_flag.data()), valid_flag.size(), neighbors);
    cudaDeviceSynchronize();
}

__global__ void compute_coeffs_kernel(const float4* points_Sel, const float4* points_Ori, const float4* Xs, bool* flag,
                                      float4* coefs, int indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= indices) return;
    if (!flag[idx]) return;

    float pd2 =
        Xs[idx].x * points_Sel[idx].x + Xs[idx].y * points_Sel[idx].y + Xs[idx].z * points_Sel[idx].z + Xs[idx].w;

    float s = 1 - 0.9 * fabs(pd2) /
                      sqrt(sqrt(points_Ori[idx].x * points_Ori[idx].x + points_Ori[idx].y * points_Ori[idx].y +
                                points_Ori[idx].z * points_Ori[idx].z));
    if (s <= 0.1)
    {
        flag[idx] = false;
        return;
    }

    coefs[idx].x = s * Xs[idx].x;
    coefs[idx].y = s * Xs[idx].y;
    coefs[idx].z = s * Xs[idx].z;
    coefs[idx].w = s * pd2;
}

void compute_coeffs(const thrust::device_vector<float4>& points_Sel, const thrust::device_vector<float4>& points_Ori,
                    const thrust::device_vector<float4>& Xs, thrust::device_vector<bool>& valid_flag,
                    thrust::device_vector<float4>& coeffs)
{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (points_Ori.size() + threadsPerBlock - 1) / threadsPerBlock;
    coeffs.resize(points_Ori.size());

    compute_coeffs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(points_Sel.data()), thrust::raw_pointer_cast(points_Ori.data()),
        thrust::raw_pointer_cast(Xs.data()), thrust::raw_pointer_cast(valid_flag.data()),
        thrust::raw_pointer_cast(coeffs.data()), points_Ori.size());
    cudaDeviceSynchronize();
}

#endif