# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

from queue import PriorityQueue
from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import *
from process_frame import *
import time
import threading
import os
import shutil

            
class iot_object_detection_module:
    """This class simulates a IoT device's object detection module. The camera() function uses Waymo Open dataset to simulate periodic frame arrival from a camera. While, the IoT devices uses procesor() function to run object detection on frames produced by the camera wewe

    Attributes:
        frame_period: The period to obtain a new frame. 
        image_directory: The path to the image directory. Default is "../dataset/".
        image_list: a list containing all the images to be processed. 
        max_frame_number: the number of frame to be processed. 
        run_queue: a priority queue that sorts task by their priority.
                A lower number means higher priority. 
        history: processing history. 
        task_finish_count: number of tasks that have finished. 
        yolo: YoloV8 model. Used for object classification.
    """

    def __init__(self, image_directory = "../../dataset/", frame_period = .1, finish_counter=300):
        self.frame_period = frame_period
        self.frame_number = 0
        self.image_directory = image_directory
        self.image_output_directory = image_directory+"out/"
        self.image_list = extract_png_files(image_directory)
        self.max_frame_number = len(self.image_list)
        self.run_queue = PriorityQueue()
        self.history = []
        self.task_finish_count = 0
        self.yolo = load_Yolo_model()
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.finish_counter = finish_counter
        self.finish_counter_initial = finish_counter
        
        if os.path.isdir(self.image_output_directory):
            shutil.rmtree(self.image_output_directory)

    def camera(self):
        """
        Camera functioning simulator loop.
        """
        while self.frame_number <= self.max_frame_number:
            self.frame_arrival(self.frame_number)
            self.frame_number = self.frame_number + 1
            time.sleep(self.frame_period * .95) # Multiplied by .95 to account for other delays
        
            
            
    def processor(self):
        """
        Main camera frame processing loop.
        """

        while not self.run_queue.empty() and self.finish_counter >=0:
            if self.run_queue.empty():
                self.finish_counter -= 1
                time.sleep(.01)
                
            else:
                self.finish_counter = self.finish_counter_initial
                with self.lock:
                    top_task = self.run_queue.get()
                processing_start_time = time.time()
                self.run_yolo(top_task)
                self.task_finish_count = self.task_finish_count + 1
                top_task.order = self.task_finish_count
                
                current_time = time.time()
                
                top_task.proc_end_time = current_time - self.start_time
                
                top_task.exec_time = (current_time - processing_start_time)
                top_task.response_time = current_time - (top_task.enqueue_time+self.start_time)
                
                if top_task.response_time > top_task.deadline:
                    top_task.missed = 1

                self.history.append(top_task)
            
        # save processing history to file
        self.save_history()
        print("Processing history saved.")

    def get_frame(self, frame_number):
        """Return and Image() object with the specified frame number."""
        if frame_number < self.max_frame_number:
            image_path = self.image_list[frame_number]
            return Image(image_path)

        else:
            return None


    def enqueue_task(self, task_set):
        """Enqueue the task_set into the run queue."""
        for task in task_set:
            task.enqueue_time = time.time() - self.start_time
            with self.lock:
                self.run_queue.put(task)


    def frame_arrival(self, frame_number):
        """Get a frame from the image list and return related tasks.
        
        Fetch a frame from the dataset using the given frame number.
        Process the frame to get tasks to be classified.
        Enqueue the tasks to the run queue .

        Args:
            frame_number: The number of frame in the image list.
        """
        frame = self.get_frame(frame_number)
        if frame:
            task_set = process_frame(frame)
            self.enqueue_task(task_set)


    def run_yolo(self, task):
        """Run the yolo model on the given task."""
        print("\r\n ###***  On image {} ***###".format(task.image_path))
        detect_images(self.yolo, task.image_path, task.coord, task.image_out_path, task.bbox_id)


    def print_image_list(self):
        """Print out the list of images to be processed."""
        print("image list is: ")
        print(self.image_list)


    def print_history(self):
        """Print out the processing history of the detection module."""
        dash = '-' * 70
        print(dash)
        print("history:")
        print('{:<7s}{:<21s}{:>25s}{:>8s}{:>10s}{:>15s}{:>12s}{:>15s}{:>15s}{:>10s}{:>10s}'.format(
            "count", "task_image", "img_coordinates", "depth", "priority",
            "enqueue_time", "exec_time", "response_time", "proc_end_time", "deadline", "missed"))

        i = 1
        for entry in self.history:
            print('{:<7d}{:s}'.format(i, entry.print()))
            i = i + 1
        print(dash)
    

    def save_history(self):
        """Save the processing history as a json file."""
        d = {}
        i = 1
        for entry in self.history:
            d[i] = entry.__dict__
            i = i + 1
        
        with open('camera_frame_processing_history.json', 'w') as outfile:
            json.dump(d, outfile, ensure_ascii=False, indent=4)

    def visualize_history(self, Text_colors=(255,255,255)):
        """Visualize processing order.

        Draw the processing order of bounding boxes in the image_out_path.
        Blue for box that meet deadline and red for box that missed.      
        """
        order = 1
        for task in self.history:
            if os.path.exists(task.image_out_path):
                image = cv2.imread(task.image_out_path)
            else:
                image = cv2.imread(task.image_path)
            image_h, image_w, _ = image.shape

            bbox_color = (0,0,255) if (task.missed) else (255,0,0)
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            coor = task.coord
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            # put object rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)
            order_text = "order: " + str(order)
            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                    fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, order_text, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
            
            cv2.imwrite(task.image_out_path, image)

            order = order + 1


