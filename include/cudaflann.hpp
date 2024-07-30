#ifndef KD_TREE_CUDA_HPP_
#define KD_TREE_CUDA_HPP_

#include <thrust/device_vector.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "flann/flann.h"

namespace pclcuda {

template <class PointType>
class KdTreeFLANN
{
   public:
    thrust::device_vector<int> indices_d;
    thrust::device_vector<PointType> sqrDists_d;
    /*
    Pointer to the CUDA KD Index
    */
    std::shared_ptr<flann::KDTreeCuda3dIndex<flann::L2<PointType>>> kdIndex;

    /*
    Set point cloud for KD Tree
    [in] inCloud - pointer for the point cloud
    */
    void setInputCloud(pcl::PointCloud<PointType>& inCloud)
    {
        size_t noOfPts = inCloud.size();
        thrust::host_vector<float4> cloudHostThrust(noOfPts);

        for (int i = 0; i < noOfPts; i++) cloudHostThrust[i] = make_float4(inCloud[i].x, inCloud[i].y, inCloud[i].z, 0);

        thrust::device_vector<float4> cldDevice = cloudHostThrust;

        flann::Matrix<float> data_device_matrix((float*)thrust::raw_pointer_cast(&cldDevice[0]), noOfPts, 3, 4 * 4);

        flann::KDTreeCuda3dIndexParams index_params;
        index_params["input_is_gpu_float4"] = true;
        this->kdIndex = std::make_shared<flann::KDTreeCuda3dIndex<flann::L2<float>>>(data_device_matrix, index_params);
        this->kdIndex->buildIndex();
    }

    /*
    Search for the nearest neighbor for "N" Points
    [in] inQuery - pcl::PointType point to query
    [in] neighbors - Number of neighbors to search
    [out] indices - Pointer to flann::Matrix for the indices of the nearest point
    [out] sqrDist - Pointer to flann::Matrix for the squared distances to the nearest point
    [return] - Number of nearest neighbor results. It should be equal to neighbors
    */
    template <typename PointT>
    int nearestKSearch(const& PointT inQuery, int neighbors, std::vector<int>& indices, std::vector<PointType>& sqrDist)
    {
        size_t noOfPts = 1;

        thrust::host_vector<float4> query_hostP{make_float4(inQuery.x, inQuery.y, inQuery.z, 0)};

        thrust::device_vector<float4> query_device = query_host;

        flann::Matrix<PointType> query_device_matrix((PointType*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3,
                                                     4 * 4);

        thrust::host_vector<int> indices_temp(noOfPts * neighbors);
        thrust::host_vector<PointType> dists_temp(noOfPts * neighbors);

        this->indices_d = indices_temp;
        this->sqrDists_d = dists_temp;

        flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&this->indices_d[0]), noOfPts,
                                                 neighbors);
        flann::Matrix<PointType> dists_device_matrix((PointType*)thrust::raw_pointer_cast(&this->sqrDists_d[0]),
                                                     noOfPts, neighbors);

        flann::SearchParams sp;
        sp.matrices_in_gpu_ram = true;
        int result =
            this->kdIndex->knnSearch(query_device_matrix, indices_device_matrix, dists_device_matrix, neighbors, sp);

        indices.resize(noOfPts * neighbors);
        sqrDist.resize(noOfPts * neighbors);

        indices_host = flann::Matrix<int>(indices.data(), noOfPts, neighbors);
        sqrDist_host = flann::Matrix<PointType>(sqrDist.data(), noOfPts, neighbors);

        thrust::copy(this->sqrDists_d.begin(), this->sqrDists_d.end(), sqrDist_host.ptr());
        thrust::copy(this->indices_d.begin(), this->indices_d.end(), indices_host.ptr());

        return result;
    }

    /*
    Search for the nearest neighbor for "N" Points
    [in] inQuery - pcl::PointType point to query
    [in] radius - Radius for the search field in the unit of the points
    [out] indices - Pointer to flann::Matrix for the indices of the nearest point
    [out] sqrDist - Pointer to flann::Matrix for the squared distances to the nearest point
    [in] max_neighbor - Maximum neighbors that should be returned by the radius search
    [return] - Number of nearest neighbor results. It should be equal to neighbors*querypoints
    */
    template <typename PointT>
    int radiusSearch(const& PointT inQuery, float radius, std::vector<int>& indices, std::vector<PointType>& sqrDist,
                     int max_neighbors)
    {
        size_t noOfPts = 1;

        thrust::host_vector<float4> query_hostP{make_float4(inQuery.x, inQuery.y, inQuery.z, 0)};

        thrust::device_vector<float4> query_device = query_host;

        flann::Matrix<PointType> query_device_matrix((PointType*)thrust::raw_pointer_cast(&query_device[0]), noOfPts, 3,
                                                     4 * 4);

        thrust::host_vector<int> indices_temp(noOfPts * max_neighbors);
        thrust::host_vector<PointType> dists_temp(noOfPts * max_neighbors);

        this->indices_d = indices_temp;
        this->sqrDists_d = dists_temp;

        flann::Matrix<int> indices_device_matrix((int*)thrust::raw_pointer_cast(&this->indices_d[0]), noOfPts,
                                                 max_neighbors);
        flann::Matrix<PointType> dists_device_matrix((PointType*)thrust::raw_pointer_cast(&this->sqrDists_d[0]),
                                                     noOfPts, max_neighbors);

        flann::SearchParams sp;
        sp.matrices_in_gpu_ram = true;
        sp.max_neighbors = max_neighbors;
        sp.sorted = true;
        int result = (this->kdIndex->radiusSearch(query_device_matrix, indices_device_matrix, dists_device_matrix,
                                                  powf(radius, 2), sp));

        indices.resize(noOfPts * neighbors);
        sqrDist.resize(noOfPts * neighbors);

        indices_host = flann::Matrix<int>(indices.data(), noOfPts, max_neighbors);
        sqrDist_host = flann::Matrix<PointType>(sqrDist.data(), noOfPts, max_neighbors);

        thrust::copy(this->sqrDists_d.begin(), this->sqrDists_d.end(), sqrDist_host.ptr());
        thrust::copy(this->indices_d.begin(), this->indices_d.end(), indices_host.ptr());

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

}  // namespace pclcuda

#endif KD_TREE_CUDA_HPP_