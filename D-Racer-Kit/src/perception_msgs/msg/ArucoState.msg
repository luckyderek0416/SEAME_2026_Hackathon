# Output of aruco_node (OpenCV ArUco marker detection = dynamic obstacle)
std_msgs/Header header
bool    detected
int32   marker_id   # -1 when nothing detected
float32 area_ratio  # marker bbox area / image area (proximity proxy)
