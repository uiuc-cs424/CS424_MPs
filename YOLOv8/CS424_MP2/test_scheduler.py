# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import time
import threading
from scheduling.Scheduler import *


start_time = time.time()

scheduler = Scheduler()

camera_thread = threading.Thread(target=scheduler.camera)
processor_thread = threading.Thread(target=scheduler.processor)

camera_thread.start()
processor_thread.start()

camera_thread.join()
processor_thread.join()
    
scheduler.print_history()
scheduler.visualize_history()

end_time = time.time()

print("Elapsed time: %f s" % (end_time - start_time))


# # example for using visualize_history_file()
# history = read_json_file("scheduling_history.json")
# visualize_history_file(history)
# # calculate group average response time from history file
# group_response_time = get_group_avg_response_time(history)
# print(group_response_time)
