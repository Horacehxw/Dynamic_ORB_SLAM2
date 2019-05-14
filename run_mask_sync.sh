#!/usr/bin/env bash
# run with optical flow propagation
# for i in {2..40}
# do
#   ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ~/Data/TUM/rgbd_dataset_freiburg3_walking_xyz ~/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/associations.txt "results/optical_flow/w_xyz_Dyna_skip$i.txt" ModelsCNN/ $i 1 results/optical_flow/time_log.txt
# done

# run naive mask synchronization
for i in {2..40}
do
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml ~/Data/TUM/rgbd_dataset_freiburg3_walking_xyz ~/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/associations.txt "results/TUM_rgbd/w_xyz_Dyna_skip$i.txt" ModelsCNN/ $i 0 results/TUM_rgbd/time_log.txt
done
