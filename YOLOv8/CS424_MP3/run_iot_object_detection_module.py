# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import time
import threading
from IoTObjectDetection.IoTObjectDetectionModule import *


start_time = time.time()

iot_object_detection_module = iot_object_detection_module(start_time=start_time)

camera_thread = threading.Thread(target=iot_object_detection_module.camera)
processor_thread = threading.Thread(target=iot_object_detection_module.processor)

camera_thread.start()
processor_thread.start()

camera_thread.join()
processor_thread.join()
    
#iot_object_detection_module.print_history()
#iot_object_detection_module.visualize_history()

end_time = time.time()

print("Elapsed time: %fs" % (end_time - start_time))


cluster_box_info = read_json_file("../../dataset/object_processing_order_history/camera_frame_processed_boxes.json")
ground_truth = read_json_file('../../dataset/waymo_ground_truth_flat.json')
history = read_json_file("../../dataset/object_processing_order_history/camera_frame_processing_history.json")

# calculate group worst response time from history file
group_response_time = get_group_worst_response_time(history)
print("Groups worst-case response times: ", group_response_time)

get_statistics('../../dataset/', ground_truth, cluster_box_info, start_time)

# # visualize cluster boxes and ground truth boxes
visualize_boxes('../../dataset/', ground_truth, cluster_box_info, start_time)
