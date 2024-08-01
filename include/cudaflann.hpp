#ifdef FLANN_USE_CUDA

#ifndef KD_TREE_CUDA_HPP_
#define KD_TREE_CUDA_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include "flann/flann.h"

namespace cudaflann {

template <typename PointType>
class KdTreeFLANN
{
   public:
    thrust::device_vector<int> indices_d;
    thrust::device_vector<float> sqrDists_d;
    /*
    Pointer to the CUDA KD Index
    */
    std::shared_ptr<flann::KDTreeCuda3dIndex<flann::L2<float>>> kdIndex;

    /*
    Set point cloud for KD Tree
    [in] inCloud - pointer for the point cloud
    */
    void setInputCloud(std::shared_ptr<pcl::PointCloud<PointType>> inCloud)
    {
        size_t noOfPts = inCloud->size();
        thrust::host_vector<float4> cloudHostThrust(noOfPts);

        std::transform(inCloud->begin(), inCloud->end(), cloudHostThrust.begin(), [](PointType point) -> float4 {
            return {point.x, point.y, point.z, 0};
        });

        thrust::device_vector<float4> cldDevice = cloudHostThrust;

        flann::Matrix<float> data_device_matrix((float*)thrust::raw_pointer_cast(&cldDevice[0]), noOfPts, 3, 4 * 4);

        flann::KDTreeCuda3dIndexParams index_params;
        index_params["input_is_gpu_float4"] = true;
        this->kdIndex = std::make_shared<flann::KDTreeCuda3dIndex<flann::L2<float>>>(data_device_matrix, index_params);
        this->kdIndex->buildIndex();
    }

    /*
    Search for the nearest neighbor for "N" Points
    [in] inQuery - PointType point to query
    [in] neighbors - Number of neighbors to search
    [out] indices - Pointer to flann::Matrix for the indices of the nearest point
    [out] sqrDist - Pointer to flann::Matrix for the squared distances to the nearest point
    [return] - Number of nearest neighbor results. It should be equal to neighbors
    */
    template <typename AllocatorType>
    int nearestKSearch(std::vector<PointType, AllocatorType> const& inQuery, int neighbors,
                       std::vector<std::vector<int>>& indices, std::vector<std::vector<float>>& sqrDist)
    {
        size_t noOfPts = inQuery.size();
        thrust::host_vector<float4> query_host(noOfPts);
        std::transform(inQuery.begin(), inQuery.end(), query_host.begin(), [=](PointType point) -> float4 {
            return {point.x, point.y, point.z, 0};
        });

        thrust::device_vector<float4> query_device = query_host;

        return nearestKSearch(query_device, neighbors, indices, sqrDist);
    }

    /*
    Search for the nearest neighbor for "N" Points
    [in] query_device - in device vector to query
    [in] neighbors - Number of neighbors to search
    [out] indices - Pointer to flann::Matrix for the indices of the nearest point
    [out] sqrDist - Pointer to flann::Matrix for the squared distances to the nearest point
    [return] - Number of nearest neighbor results. It should be equal to neighbors
    */
    int nearestKSearch(thrust::device_vector<float4>& query_device, int neighbors,
                       std::vector<std::vector<int>>& indices, std::vector<std::vector<float>>& sqrDist)
    {
        size_t noOfPts = query_device.size();
        flann::Matrix<float> query_device_matrix((float*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3, 4 * 4);

        thrust::host_vector<int> indices_temp(noOfPts * neighbors);
        thrust::host_vector<float> dists_temp(noOfPts * neighbors);

        this->indices_d = indices_temp;
        this->sqrDists_d = dists_temp;

        flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&this->indices_d[0]), noOfPts,
                                                 neighbors);
        flann::Matrix<float> dists_device_matrix((float*)thrust::raw_pointer_cast(&this->sqrDists_d[0]), noOfPts,
                                                 neighbors);

        flann::SearchParams sp;
        sp.matrices_in_gpu_ram = true;
        int result =
            this->kdIndex->knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, neighbors, sp);

        std::vector<int> indices_flat(noOfPts * neighbors);
        std::vector<float> sqrDist_flat(noOfPts * neighbors);

        thrust::copy(this->indices_d.begin(), this->indices_d.end(), indices_flat.data());
        thrust::copy(this->sqrDists_d.begin(), this->sqrDists_d.end(), sqrDist_flat.data());

        indices.resize(noOfPts);
        sqrDist.resize(noOfPts);

        for (size_t i = 0; i < noOfPts; i++)
        {
            auto row_iterator_indices = indices_flat.begin() + i * neighbors;
            auto row_iterator_sqrDist = sqrDist_flat.begin() + i * neighbors;
            indices[i].assign(row_iterator_indices, row_iterator_indices + neighbors);
            sqrDist[i].assign(row_iterator_sqrDist, row_iterator_sqrDist + neighbors);
        }

        return result;
    }

    /*
    Search for the nearest neighbor for "N" Points
    [in] inQuery - PointType point to query
    [in] radius - Radius for the search field in the unit of the points
    [out] indices - Pointer to flann::Matrix for the indices of the nearest point
    [out] sqrDist - Pointer to flann::Matrix for the squared distances to the nearest point
    [in] max_neighbor - Maximum neighbors that should be returned by the radius search
    [return] - Number of nearest neighbor results. It should be equal to neighbors*querypoints
    */
    template <typename AllocatorType>
    int radiusSearch(std::vector<PointType, AllocatorType> const& inQuery, float radius,
                     std::vector<std::vector<int>>& indices, std::vector<std::vector<float>>& sqrDist,
                     int max_neighbors = 500)
    {
        size_t noOfPts = inQuery.size();
        thrust::host_vector<float4> query_host(noOfPts);
        std::transform(inQuery.begin(), inQuery.end(), query_host.begin(), [=](PointType point) -> float4 {
            return {point.x, point.y, point.z, 0};
        });

        thrust::device_vector<float4> query_device = query_host;

        flann::Matrix<float> query_device_matrix((float*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3, 4 * 4);

        thrust::host_vector<int> indices_temp(noOfPts * max_neighbors);
        thrust::host_vector<float> dists_temp(noOfPts * max_neighbors);

        this->indices_d = indices_temp;
        this->sqrDists_d = dists_temp;

        flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&this->indices_d[0]), noOfPts,
                                                 max_neighbors);
        flann::Matrix<float> dists_device_matrix((float*)thrust::raw_pointer_cast(&this->sqrDists_d[0]), noOfPts,
                                                 max_neighbors);

        flann::SearchParams sp;
        sp.matrices_in_gpu_ram = true;
        sp.max_neighbors = max_neighbors;
        sp.sorted = true;
        int result = (this->kdIndex->radiusSearch(query_device_matrix, indices_device_matrix, dists_device_matrix,
                                                  powf(radius, 2), sp));

        std::vector<int> indices_flat(noOfPts * max_neighbors);
        std::vector<float> sqrDist_flat(noOfPts * max_neighbors);

        thrust::copy(this->indices_d.begin(), this->indices_d.end(), indices_flat.data());
        thrust::copy(this->sqrDists_d.begin(), this->sqrDists_d.end(), sqrDist_flat.data());

        indices.resize(noOfPts);
        sqrDist.resize(noOfPts);

        for (size_t i = 0; i < noOfPts; i++)
        {
            auto row_iterator_indices = indices_flat.begin() + i * max_neighbors;
            auto row_iterator_sqrDist = sqrDist_flat.begin() + i * max_neighbors;

            auto end_row_iterator_indices = row_iterator_indices;
            auto end_row_iterator_sqrDist = row_iterator_sqrDist;
            for (int j = 0; j < max_neighbors; j++)
            {
                if (*end_row_iterator_indices < 0 || *end_row_iterator_sqrDist < 0) break;
                end_row_iterator_indices++;
                end_row_iterator_sqrDist++;
            }

            indices[i].assign(row_iterator_indices, end_row_iterator_indices);
            sqrDist[i].assign(row_iterator_sqrDist, end_row_iterator_sqrDist);
        }

        return result;
    }

    /*
    Returns the pointer to the thrust device vector for Indices
    */
    int* getIndicesDevicePtr() { return (int*)thrust::raw_pointer_cast(&this->indices_d[0]); }

    /*
    Returns the pointer to the thrust device vector for Squared Distances
    */
    PointType* getSqrtDistDevicePtr() { return (float*)thrust::raw_pointer_cast(&this->sqrDists_d[0]); }
};

}  // namespace cudaflann

struct PointAssociateToMapFunctor
{
    const float Cxx, Cxy, Cxz, Tx;
    const float Cyx, Cyy, Cyz, Ty;
    const float Czx, Czy, Czz, Tz;

    PointAssociateToMapFunctor(Eigen::Affine3f transformation)
        : Cxx(transformation(0, 0))
        , Cxy(transformation(0, 1))
        , Cxz(transformation(0, 2))
        , Tx(transformation(0, 3))
        , Cyx(transformation(1, 0))
        , Cyy(transformation(1, 1))
        , Cyz(transformation(1, 2))
        , Ty(transformation(1, 3))
        , Czx(transformation(2, 0))
        , Czy(transformation(2, 1))
        , Czz(transformation(2, 2))
        , Tz(transformation(2, 3))
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

void apply_transforms(thrust::device_vector<float4>& input, thrust::device_vector<float4>& output,
                      Eigen::Affine3f& transformation);

#endif  // KD_TREE_CUDA_HPP_
#endif  // FLANN_USE_CUDA