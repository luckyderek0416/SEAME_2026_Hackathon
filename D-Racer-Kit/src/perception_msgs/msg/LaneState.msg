# Output of lane_node (OpenCV rule-based lane following)
std_msgs/Header header
bool    lane_found
float32 offset      # normalised lane-center offset, [-1.0 .. 1.0]  (+ = lane center is right of image center)
float32 curvature   # rough curvature estimate (0 if unused)
int32   num_lanes   # how many lane lines were found this frame (0,1,2)
bool    junction    # outer-ring boundary has an opening (roundabout entry/exit point)
float32 yellow_ratio  # fraction of ROI that is yellow (roundabout region marker), 0..1
float32 yellow_offset # normalised x of yellow centroid, [-1..1] (+ = yellow is to the right)
