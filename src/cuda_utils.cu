#ifdef FLANN_USE_CUDA

#include <iostream>
#include "cuda_utils.h"

void resize_device_vector(thrust::device_vector<float>& vector, size_t v_size) { vector.resize(v_size); }

/*! Functor to apply a transformation in GPU using thrust::transform
 */
struct PointAssociateToMapFunctor
{
    const float Cxx, Cxy, Cxz, Tx;
    const float Cyx, Cyy, Cyz, Ty;
    const float Czx, Czy, Czz, Tz;

    PointAssociateToMapFunctor(const std::vector<float>& transformation)
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

void apply_transforms(const thrust::device_vector<float4>& input, const std::vector<float>& transformation,
                      thrust::device_vector<float4>& output)
{
    output.resize(input.size());
    thrust::transform(input.begin(), input.end(), output.begin(), PointAssociateToMapFunctor(transformation));
}

struct ValidateDistanceFunctor
{
    const float* distance;

    ValidateDistanceFunctor(const float* _distance) : distance(_distance) {}

    __device__ bool operator()(int offset) const { return distance[offset + 4] < 1.0; }
};

void validate_distance(const thrust::device_vector<float>& distance, const thrust::device_vector<int>& offsets,
                       thrust::device_vector<bool>& valid_flag)
{
    valid_flag.resize(offsets.size());
    thrust::transform(offsets.begin(), offsets.end(), valid_flag.begin(),
                      ValidateDistanceFunctor(thrust::raw_pointer_cast(distance.data())));
}

struct extract_and_flatten_functor
{
    const float4* points;
    const int* indices;
    const int offset;

    extract_and_flatten_functor(const thrust::device_vector<float4>& points_,
                                const thrust::device_vector<int>& indices_, const int offset_)
        : points(thrust::raw_pointer_cast(points_.data()))
        , indices(thrust::raw_pointer_cast(indices_.data()))
        , offset(offset_)
    {
    }

    __device__ thrust::tuple<float, float, float> operator()(int j) const
    {
        int index = indices[offset + j];
        return thrust::make_tuple(points[index].x, points[index].y, points[index].z);
    }
};

void get_points_submatrix(const thrust::device_vector<float4>& cldDevice, const thrust::device_vector<int>& indices_d,
                          int offset, int neighbors, thrust::device_vector<float>& output)
{
    output.resize(3 * neighbors);
    // Perform the transformation
    thrust::transform(
        thrust::make_counting_iterator(0),          // Starting index
        thrust::make_counting_iterator(neighbors),  // Ending index
        thrust::make_zip_iterator(thrust::make_tuple(output.begin(), output.begin() + 1, output.begin() + 2)),
        extract_and_flatten_functor(cldDevice, indices_d, offset));
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

struct GetPlanesCoefFunctor
{
    __device__ float4 operator()(const thrust::tuple<float, float, float, bool>& t) const
    {
        // Unpack the tuple
        float pa = thrust::get<0>(t);
        float pb = thrust::get<1>(t);
        float pc = thrust::get<2>(t);
        bool flag = thrust::get<3>(t);

        // Check the flag
        if (!flag) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Perform the operations
        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        float pd = 1.0f / ps;

        return make_float4(pa, pb, pc, pd);
    }
};

void get_planes_coef(thrust::device_vector<float>& X0s, thrust::device_vector<float4>& Xs,
                     thrust::device_vector<bool>& flagValid, size_t size_x)
{
    Xs.resize(size_x);

    // Create a zip iterator to combine the input vectors
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(X0s.begin(), X0s.begin() + size_x, X0s.begin() + 2 * size_x, flagValid.begin()));

    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(X0s.begin() + size_x, X0s.begin() + 2 * size_x, X0s.end(), flagValid.end()));

    // Apply the functor to the input range and store the result in Xs
    thrust::transform(begin, end, Xs.begin(), GetPlanesCoefFunctor());
}

struct ValidatePlanesFunctor
{
    const float4* cloud;
    const int* pointSearchIndices;
    const int* offsets;
    const int pointSearchIndSize;

    ValidatePlanesFunctor(const float4* cloud_, const int* pointSearchIndices_, const int* offsets_,
                          int pointSearchIndSize_)
        : cloud(cloud_)
        , pointSearchIndices(pointSearchIndices_)
        , offsets(offsets_)
        , pointSearchIndSize(pointSearchIndSize_)
    {
    }

    __device__ bool operator()(const thrust::tuple<float4, int, bool>& t) const
    {
        float4 plane = thrust::get<0>(t);
        int idx = thrust::get<1>(t);
        bool flag = thrust::get<2>(t);

        if (!flag) return false;

        float pa = plane.x;
        float pb = plane.y;
        float pc = plane.z;
        float pd = plane.w;

        // Validate the plane
        for (int j = 0; j < pointSearchIndSize; ++j)
        {
            int pointIdx = pointSearchIndices[offsets[idx] + j];
            float4 point = cloud[pointIdx];
            if (fabs(pa * point.x + pb * point.y + pc * point.z + pd) > 0.2)
            {
                return false;
            }
        }

        return true;
    }
};

// Function to perform the transformation using Thrust
void validate_planes(const thrust::device_vector<float4>& cldDevice, const thrust::device_vector<int>& indices_d,
                     const thrust::device_vector<int>& offsets, const thrust::device_vector<float4>& Xs, int neighbors,
                     thrust::device_vector<bool>& valid_flag)
{
    thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(Xs.begin(), thrust::counting_iterator<int>(0), valid_flag.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(Xs.end(), thrust::counting_iterator<int>(Xs.size()), valid_flag.end())),
        valid_flag.begin(),
        ValidatePlanesFunctor(thrust::raw_pointer_cast(cldDevice.data()), thrust::raw_pointer_cast(indices_d.data()),
                              thrust::raw_pointer_cast(offsets.data()), neighbors));
}

struct compute_coeffs_functor
{
    __device__ thrust::tuple<float4, bool> operator()(const thrust::tuple<float4, float4, float4, bool>& t) const
    {
        float4 points_Sel = thrust::get<0>(t);
        float4 points_Ori = thrust::get<1>(t);
        float4 Xs = thrust::get<2>(t);
        bool flag = thrust::get<3>(t);

        float4 coeffs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (!flag) return thrust::make_tuple(coeffs, false);

        float pd2 = Xs.x * points_Sel.x + Xs.y * points_Sel.y + Xs.z * points_Sel.z + Xs.w;

        float s =
            1.0f -
            0.9f * fabs(pd2) /
                sqrt(sqrt(points_Ori.x * points_Ori.x + points_Ori.y * points_Ori.y + points_Ori.z * points_Ori.z));

        if (s <= 0.1f)
        {
            return thrust::make_tuple(coeffs, false);
        }

        coeffs.x = s * Xs.x;
        coeffs.y = s * Xs.y;
        coeffs.z = s * Xs.z;
        coeffs.w = s * pd2;

        return thrust::make_tuple(coeffs, true);
    }
};

// Function to perform the transformation using Thrust
void compute_coeffs(const thrust::device_vector<float4>& points_Sel, const thrust::device_vector<float4>& points_Ori,
                    const thrust::device_vector<float4>& Xs, thrust::device_vector<bool>& valid_flag,
                    thrust::device_vector<float4>& coeffs)
{
    coeffs.resize(points_Ori.size());

    // Apply the transformation using Thrust
    thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(points_Sel.begin(), points_Ori.begin(), Xs.begin(), valid_flag.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(points_Sel.end(), points_Ori.end(), Xs.end(), valid_flag.end())),
        thrust::make_zip_iterator(thrust::make_tuple(coeffs.begin(), valid_flag.begin())), compute_coeffs_functor());
}

void copy_to_host(const thrust::device_vector<float4>& vector_d, thrust::host_vector<float4>& vector_h)
{
    vector_h.resize(vector_d.size());
    thrust::copy(vector_d.begin(), vector_d.end(), vector_h.begin());
}

template <>
void copy_to_host(const thrust::device_vector<float4>& vector_d, thrust::host_vector<float4>& vector_h)
{
    vector_h.resize(vector_d.size());
    thrust::copy(vector_d.begin(), vector_d.end(), vector_h.begin());
}

template <>
void copy_to_host(const thrust::device_vector<int>& vector_d, thrust::host_vector<int>& vector_h)
{
    vector_h.resize(vector_d.size());
    thrust::copy(vector_d.begin(), vector_d.end(), vector_h.begin());
}

template <>
void copy_to_host(const thrust::device_vector<bool>& vector_d, thrust::host_vector<bool>& vector_h)
{
    vector_h.resize(vector_d.size());
    thrust::copy(vector_d.begin(), vector_d.end(), vector_h.begin());
}

#endif