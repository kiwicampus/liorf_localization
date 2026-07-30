float32 optimization_time
uint16 optimization_iterations
float32 optimization_delta_r
float32 optimization_delta_t
float32 optimization_residuals_rmse
bool optimization_timed_out
# Scan-to-map match quality (populated at end of scan2MapOptimization)
uint32 effective_features
float32 inlier_ratio
bool optimization_converged
bool is_degenerate
# Post-optimization scan-to-map fitness (current pose vs GlobalMap kdtree)
float32 median_map_match_distance
float32 p90_map_match_distance
float32 map_fitness_ratio
