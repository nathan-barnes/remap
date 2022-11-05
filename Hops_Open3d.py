
import open3d as o3d
import time
import copy
import numpy as np

# bbtarget = r"C:\Box\R&D Services\Restricted\04_Research Trajectories\Point Cloud\Grasshopper\CCR_PMU_B-Target.txt"
# bbsource = r"C:\Box\R&D Services\Restricted\04_Research Trajectories\Point Cloud\Grasshopper\CCR_PMU_B_source.pcd"
# bbtarget = r"C:\Box\R&D Services\Restricted\04_Research Trajectories\Point Cloud\Grasshopper\Target.pcd"
# bbsource = r"C:\Box\R&D Services\Restricted\04_Research Trajectories\Point Cloud\Grasshopper\Source.pcd"

#------------------------------------prep----------------------------

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) #orange/yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=1.0,
                                      front=[-15.046287,50.89343,-29.4295],
                                      width=860,
                                      height=540,
                                    #   front=[0.6452, -0.3036, -0.7011],
                                    #   lookat=[1.9892, 2.0208, 1.8945],
                                      lookat=[17.797604, 2.947689, -6.569047],
                                      up=[0,-10.699957,85.661494]
                                    #   up=[-0.2779, -0.9482, 0.1556]
    )


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 5#10, 3 was good for pi
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) #10?
        # o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # radius_feature = voxel_size * 5 # original, 2 was good for pi and small
    radius_feature = voxel_size * 7
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) #2 was good for pi
        # o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def prepare_dataset(voxel_size, bbsource, bbtarget):
    print(":: Load two point clouds and disturb initial pose.")

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(bbsource)
    target = o3d.io.read_point_cloud(bbtarget)
    

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def dataset( bbsource, bbtarget):
    print(":: Load two point clouds and disturb initial pose.")

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(bbsource)
    target = o3d.io.read_point_cloud(bbtarget)
       
    return source, target


# voxel_size = 2 # works for boxes
# voxel_size = .0051 
# voxel_size = 10 # 
# voxel_size = 0.05  # means 5cm for this dataset
# source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
#     voxel_size)

#------------------------------------execute----------------------------


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    # distance_threshold = .25 #works on boxes
    distance_threshold = voxel_size * 0.5#5#0.4 #0.25 #original .5
    # distance_threshold = 1
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size,  num_iterations=80000):#80000 .9999
    distance_threshold = voxel_size * 2#1.5
    # distance_threshold=0.05
    ransac_n=3
    # distance_threshold = 1.0
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(#original 
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n, [ #3
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength( .9),#.9
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        # ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, num_iterations))
    # result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    #     source_down, target_down, source_fpfh, target_fpfh, True,
    #     0.05, #distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     ransac_n=4,
    #     criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, num_iterations),
    # )

    return result
