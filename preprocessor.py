"""
1. read videos and convert them to frames
2. tag each of them with the person's name
3. take HOG for face localisation and do pose estimation/alignment
4. take out 128 features from all the images
5. Train a svm classifier on all the extracted images
6. save the trained model
"""
import cv2
import time
import face_recognition
import os
import numpy as np
import config as cfg

from threading import Thread
from Queue import Queue
import imutils

class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        # self.videoname,_ =  os.path.splitext(path)
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.image_name = None
        self.sequence = self.stream.get(cv2.CAP_PROP_POS_FRAMES)
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                current_index = self.stream.get(cv2.CAP_PROP_POS_FRAMES)
                self.sequence = current_index
                self.temporal_location = self.stream.get(cv2.CAP_PROP_POS_MSEC)
                self.image_name = "img-%d.jpg" % current_index

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file

                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def get_image_name(self):
        return self.image_name

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def show_face_markings(face_locations,small_frame,msg):
    """
    Marks Face with a box and message
    :param face_locations: The top, right, bottom, left co-ordinates of the face
    :param small_frame: the frame array
    :param msg: Message to be displayed below the box
    :return:
    """
    for face_location in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top, right, bottom, left = face_location

        # Draw a box around the face
        cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(small_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        ## This puts the clssification in the box of the face
        cv2.putText(small_frame, msg, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

def display_frame(small_frame):
    """
     Display the resulting image
    :param small_frame: the frame array
    :return:
    """
    cv2.imshow('Video', small_frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


def annotate_image(small_frame,msg):
    """
    Just puts the message on the image . No Box.
    :param small_frame: the frame array
    :param msg: the data to be put on top-left corner of the frame
    :return:
    """
    cv2.putText(small_frame, " {}".format(msg),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return

def write_to_file(small_frame,image_path):
    """
    Writes the frames on the disk
    :param small_frame: The frame array
    :param image_path: The path where to write
    :return:
    """
    if not os.path.exists(image_path):
        cv2.imwrite(image_path, small_frame)
        print "wrote file to %s " % image_path

def encoding_exists(video_dir_name):
    return os.path.exists(os.path.join(cfg.encoding_dir,video_dir_name+".npy"))

def write_face_encodings(frame_data_array,video_dir_name):
    """
    Writes the face_encodings on the disk
    :param frame_data_array: The frame array containing all encoding of video
    :param video_dir_name : Name of the video
    :param image_path: The path where to write
    :return:
    """
    frame_data_array = np.array(frame_data_array)
    encoding_dir = cfg.encoding_dir
    frame_array_path = os.path.join(encoding_dir,video_dir_name+".npy")
    # print frame_array_path
    if not os.path.exists(encoding_dir):
        os.makedirs(encoding_dir)
    np.save(frame_array_path, frame_data_array)
    print "wrote frame array for %s to %s " % (video_dir_name,frame_array_path)
    return frame_array_path

def video_processor_single_thread(video):

    processed_data_dir = cfg.processed_data_dir
    video_path, ext = os.path.splitext(video)
    video_dir_name = os.path.basename(video_path)

    if encoding_exists(video_dir_name):
        print "Encoding for %s already exists, exiting ..." % video_dir_name
        return

    processed_video_dir = os.path.join(processed_data_dir, video_dir_name)
    if not os.path.exists(processed_video_dir):
        os.makedirs(processed_video_dir)
    frame_data_array = []
    count = 0
    video_capture = cv2.VideoCapture(video)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            return
        current_index = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        image_name = "img-%d.jpg" % current_index
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_frame = imutils.rotate(small_frame, 270)
        face_locations = face_recognition.face_locations(small_frame)
        msg = "No_Face"
        if face_locations:
            msg = "Face"

        image_path = os.path.join(processed_video_dir, msg + "___" + image_name)

        annotate_image(small_frame, msg)
        display_frame(small_frame)

        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        if face_encodings:
            non_empty_face_encodings = face_encodings[0]
            frame_data_array.append(non_empty_face_encodings)
            count += 1

        if count % 5 == 0:
            print "Added %d frames to buffer of %s " % (count, video_dir_name)

    cv2.destroyAllWindows()
    write_face_encodings(frame_data_array, video_dir_name)

def video_processor(video):
    processed_data_dir = cfg.processed_data_dir
    video_path , ext = os.path.splitext(video)
    video_dir_name = os.path.basename(video_path)

    if encoding_exists(video_dir_name):
        print "Encoding for %s already exists, exiting ..." % video_dir_name
        return

    processed_video_dir = os.path.join(processed_data_dir,video_dir_name)
    if not os.path.exists(processed_video_dir) :
        os.makedirs(processed_video_dir)
    frame_data_array = []

    fvs = FileVideoStream(video).start()
    time.sleep(2.0)
    count = 0
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        face_found=False
        image_name = fvs.get_image_name()

        frame = fvs.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = imutils.rotate(small_frame, 270)
        face_locations = face_recognition.face_locations(small_frame)
        msg = "No_Face"
        if face_locations:
            msg = "Face"

        image_path =  os.path.join(processed_video_dir,msg + "___" + image_name)

        annotate_image(small_frame,msg)
        display_frame(small_frame)

        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        if  face_encodings:
            non_empty_face_encodings =  face_encodings[0]
            frame_data_array.append(non_empty_face_encodings)
            count += 1

        if count % 5 == 0:
            print "Added %d frames to buffer of %s " % (count,video_dir_name)

    fvs.stop()
    cv2.destroyAllWindows()
    write_face_encodings(frame_data_array, video_dir_name)

if __name__=="__main__":

    import glob
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        video_dir = cfg.video_dir
        dir_list = glob.glob(video_dir)
        executor.map(video_processor,dir_list)