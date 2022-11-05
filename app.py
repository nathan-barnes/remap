# from msilib.schema import Directory
import json
from flask import Flask, request, flash, redirect, url_for, send_from_directory, send_file, render_template
import ghhops_server as hs
from Hops_Open3d import execute_fast_global_registration, draw_registration_result, prepare_dataset, preprocess_point_cloud, execute_global_registration, dataset
import open3d as o3d
import time
import copy
import numpy as np

#Description
# hops link to open3d point cloud processing
# to setup run: pip install -r requirements.txt
# open3d requires python 3.8
# python app.py

from os import path, getcwd

import sys
sys.path.append("C:\Python38\Lib\site-packages")

from os.path import join, dirname, realpath



#-------------------------------------------------------------------------------------------register hops app as middleware
app = Flask(__name__, template_folder='templates')
hops = hs.Hops(app)


#-------------------------------------------------------------------------------------------Global vars



@app.route("/help",  methods=['GET', 'POST'])
def help():
    return "Welcome to Grashopper Hops for CPython!"


@hops.component(
    "/icp",
    name="icp",
    description="itterative closest point, send point cloud to align, returns transform",
    # icon="./img/kiko.png",
    inputs=[
        hs.HopsBoolean("run", "R", "run the component"),
        # hs.HopsPoint("PtToMatch", "ptM", "Point cloud to match to", hs.HopsParamAccess.LIST),
        # hs.HopsPoint("PtToOrient", "ptO", "Point cloud to orient", hs.HopsParamAccess.LIST),
        hs.HopsString("PtToMatch", "ptM", "Point cloud to match to"),
        hs.HopsString("PtToOrient", "ptO", "Point cloud to orient"),
        # hs.HopsNumber("voxel_size", "vox", "voxel size to cull"),
        hs.HopsNumber("icp threshold", "icpTh", "ICP threshold"),
        # hs.HopsInteger("num_iterations", "numIt", "RANSAC num_iterations"),
        hs.HopsInteger("icp max_iteration", "icpMxIt", "Icp max_iteration"),
    ],
    outputs=[
        hs.HopsString("fastTransMatrix", "fasTranMat", "output Fast transform to orient"),
        hs.HopsString("transMatrix", "TranMat", "output transform to orient"),
        hs.HopsString("result_ransac", "result_ransac", "result_ransac"),
    ]
)

def icp(run,  PtToOrient, PtToMatch, threshold,  max_iteration): #need to add - voxel_size, threshold, options as avalue list?

    OrientPt = 'OrientPt.pcd'
    MatchPt = 'MatchPt.pcd'

    print('threshold', threshold, 'max_iteration', max_iteration)

    if(run):
        print ('starting')
        start = time.time()
        
        #---setup----      
        with open(OrientPt, 'w') as f:
            f.writelines(PtToOrient)

        with open(MatchPt, 'w') as f:
            f.writelines(PtToMatch)
            


        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(2, OrientPt, MatchPt)
        print (source)
        
        # ------------------------use for alignment -----------------------
    
        #Ransac looks for center alignements, gets heavy but more robust
        # result_ransac = execute_global_registration(source_down, target_down,
        #                                     source_fpfh, target_fpfh,
        #                                     voxel_size, num_iterations)
        # print('result_ransac',  len(result_ransac.correspondence_set), result_ransac)


        #ICP using fast transform
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source, target, threshold, result_ransac.transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     # o3d.pipelines.registration.ICPConvergenceCriteria( max_iteration=500000)) #5000000 good but slow
        #     o3d.pipelines.registration.ICPConvergenceCriteria( max_iteration)) #5000000 good but slow

        # print('reg_p2preg_p2p, ', len(reg_p2p.correspondence_set), reg_p2p)    


        # ----- doesn't use fast transform
        trans_init = np.eye(4)

        reg_p2p = o3d.pipelines.registration.registration_icp( 
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000000))

        
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(source, target, reg_p2p.transformation) #comment out


        
        print("final registration took %.3f sec.\n" % (time.time() - start))


        # return result_ransac.transformation.tolist(), reg_p2p.transformation.tolist(), len(reg_p2p.correspondence_set)
        return 'result_ransac.transformation.tolist()', reg_p2p.transformation.tolist(), len(reg_p2p.correspondence_set)

    else:
        return 'waiting'


@hops.component(
    "/icpFeature",
    name="icpFeature",
    description="itterative closest point, send point cloud to align, returns transform",
    # icon="./img/kiko.png",
    inputs=[
        hs.HopsBoolean("run", "R", "run the component"),
        # hs.HopsPoint("PtToMatch", "ptM", "Point cloud to match to", hs.HopsParamAccess.LIST),
        # hs.HopsPoint("PtToOrient", "ptO", "Point cloud to orient", hs.HopsParamAccess.LIST),
        hs.HopsString("PtToMatch", "ptM", "Point cloud to match to", hs.HopsParamAccess.LIST),
        hs.HopsString("PtToOrient", "ptO", "Point cloud to orient", hs.HopsParamAccess.LIST),
        # hs.HopsNumber("voxel_size", "vox", "voxel size to cull"),
        hs.HopsNumber("icp threshold", "icpTh", "ICP threshold"),
        # hs.HopsInteger("num_iterations", "numIt", "RANSAC num_iterations"),
        hs.HopsInteger("icp max_iteration", "icpMxIt", "Icp max_iteration"),
    ],
    outputs=[
        # hs.HopsString("fastTransMatrix", "fasTranMat", "output Fast transform to orient"),
        hs.HopsString("transMatrix", "TranMat", "output transform to orient", hs.HopsParamAccess.LIST),
        # hs.HopsString("result_ransac", "result_ransac", "result_ransac"),
    ]
)

def icpFeature(run,  OrientPtLst, PtToMatchList, threshold,  max_iteration): 

    OrientPt = 'OrientPt.pcd'
    MatchPt = 'MatchPt.pcd'
    returnList = []

    print('threshold', threshold, 'max_iteration', max_iteration)

    if(run):
        print ('starting')
        start = time.time()

        for TranIndex in range(len(OrientPtLst)):
            
            
            #---setup----      
            with open(OrientPt, 'w') as f:
                f.writelines(OrientPtLst[TranIndex])

            with open(MatchPt, 'w') as f:
                f.writelines(PtToMatchList[TranIndex])
                


            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(2, OrientPt, MatchPt)
            print (source)
            
        
            # ----- doesn't use fast transform
            trans_init = np.eye(4)

            reg_p2p = o3d.pipelines.registration.registration_icp( 
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000000))

            
            print(reg_p2p)
            print("Transformation is:")
            print(reg_p2p.transformation)
            # draw_registration_result(source, target, reg_p2p.transformation) #comment out


            
            returnList.append(reg_p2p.transformation.tolist())
       
        print("final registration took %.3f sec.\n" % (time.time() - start))

        return returnList

    else:
        return 'waiting'



@hops.component(
    "/icpMatch",
    name="icpMatch",
    description="itterative closest point, send point cloud to align, returns transform",
    inputs=[
        hs.HopsBoolean("run", "R", "run the component"),
        hs.HopsString("PtToMatch", "ptM", "Point cloud to match to"),
        hs.HopsString("PtToOrient", "ptO", "Point cloud to orient"),
        hs.HopsNumber("icp threshold", "icpTh", "ICP threshold"),
        hs.HopsInteger("Voxel Size", "VS", "size of voxels for cleaning"),
    ],
    outputs=[
        # hs.HopsString("fastTransMatrix", "fasTranMat", "output Fast transform to orient"),
        hs.HopsString("transMatrix", "TranMat", "output transform to orient"),
        hs.HopsString("result_ransac", "result_ransac", "result_ransac"),
    ]
)

def icpMatch(run,   PtToMatch, PtToOrient, threshold,  Voxel_Size): #need to add - voxel_size, threshold, options as avalue list?

    OrientPt = 'OrientPt.pcd'
    MatchPt = 'MatchPt.pcd'

    print('threshold', threshold, 'Voxel_Size', Voxel_Size)

    if(run):
        print ('starting')
        start = time.time()
        
        #---setup----      
        with open(OrientPt, 'w') as f:
            f.writelines(PtToOrient)

        with open(MatchPt, 'w') as f:
            f.writelines(PtToMatch)
            


        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(Voxel_Size, OrientPt, MatchPt)
        print (source)
        
        # ------------------------use for alignment -----------------------
    
        #Ransac looks for center alignements, gets heavy but more robust
        # result_ransac = execute_global_registration(source_down, target_down,
        #                                     source_fpfh, target_fpfh,
        #                                     voxel_size, num_iterations)
        # print('result_ransac',  len(result_ransac.correspondence_set), result_ransac)


        #ICP using fast transform
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source, target, threshold, result_ransac.transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     # o3d.pipelines.registration.ICPConvergenceCriteria( max_iteration=500000)) #5000000 good but slow
        #     o3d.pipelines.registration.ICPConvergenceCriteria( max_iteration)) #5000000 good but slow

        # print('reg_p2preg_p2p, ', len(reg_p2p.correspondence_set), reg_p2p)    


        # ----- doesn't use fast transform
        trans_init = np.eye(4)

        reg_p2p = o3d.pipelines.registration.registration_icp( 
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000000))

        
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(source, target, reg_p2p.transformation) #comment out


        
        print("final registration took %.3f sec.\n" % (time.time() - start))


        # return result_ransac.transformation.tolist(), reg_p2p.transformation.tolist(), len(reg_p2p.correspondence_set)
        return  reg_p2p.transformation.tolist(), len(reg_p2p.correspondence_set)

    else:
        return 'waiting'





if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5001)
    # app.run(debug=True)
    app.config.update (
    DEBUG = True
    )