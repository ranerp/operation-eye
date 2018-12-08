import numpy
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from threading import Thread
from queue import Queue
from screeninfo import get_monitors

class FileVideoStream():
	def __init__(self, path, queueSize=32, loop=True):
		self.__path = path
		self.__stopped = False
		self.__loop = loop
		self.__buffer = Queue(maxsize=queueSize)
		
		self.__initialize_stream()
		
	def __initialize_stream(self):
		print("[INFO][VideoStream] Initializing video stream of file '{}'...".format(self.__path))
		self.__stream = cv2.VideoCapture(self.__path)
	
	def start(self):
		print("[INFO][VideoStream] Starting video stream in another thread...")
		
		if self.__loop:
			print("[INFO][VideoStream] Running video stream in loop mode...")

		thread = Thread(target=self.__buffer_video, args=())
		thread.daemon = True
		thread.start()
		return self
	
	def __buffer_video(self):
		while True:
			if self.__stopped:
				self.__cleanup()
				return
			
			if not self.__buffer.full():
				(grabbed, frame) = self.__stream.read()
				
				if not grabbed:
					print("[INFO][VideoStream] Video's '{}' end reached. Starting from the beginning.".format(self.__path))
					
					if self.__loop:
						self.__stream.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
						(grabbed, frame) = self.__stream.read()
					else:
						self.stop()
						return

				self.__buffer.put(frame)
				
	def get(self):
		return self.__buffer.get()
		
	def stop(self):
		self.__stopped = True
		
	def __cleanup(self):
		self.__stream.release()

class Eye():
	WINDOW_NAME = "EYE"
	
	def __init__(self,
				 cascade_path,
				 file_video_stream,
				 width=0,
				 height=0,
				 queue_size=64,
				 delay_in_secs=3,
				 detection_delta=3,
				 detection_resolution=500):
		
		self.__cascade_path = cascade_path
		self.__file_video_stream = file_video_stream
		self.__width = width
		self.__height = height
		self.__queue_size = queue_size
		self.__delay_in_secs = delay_in_secs
		self.__detection_delta = detection_delta
		self.__detection_resolution = detection_resolution
		
		self.__frame_buffer = Queue(maxsize=queue_size)
		self.__last_frame_time = 0
		self.__elapsed_secs = 0
	
		self.__current_stream_frame = None
		self.__is_detected = False
		self.__last_detection_time = 0
		
		self.__is_first_frame = True
		
		self.__load_detector()

	def __load_detector(self):
		print("[INFO][Eye] Loading face detector...")
		self.__detector = cv2.CascadeClassifier(self.__cascade_path)

	def run(self):
		self.__initialize_window()
		self.__initialize_background()
		self.__start_video_stream()
		
		self.__last_frame_time = time.time()

		print("[INFO][Eye] Starting detection...")
		
		fps = FPS().start()
		
		while True:
			self.__read_stream_frame()
			self.__detect()
			self.__create_frame()
			
			if self.__is_delay_reached():
				self.__show_frame()
			else:
				print("[INFO][Eye] Delaying. Showing in {}".format(self.__time_until_delay_reached()))

			if cv2.waitKey(5) & 0xFF == ord("q"):
				break

			fps.update()
			self.__update_elapsed_time()
		
		fps.stop()
		print("[INFO][Eye] FPS: {}".format(fps.fps()))
		
		self.__cleanup()
	
	def __initialize_window(self):
		cv2.namedWindow(Eye.WINDOW_NAME, cv2.WINDOW_NORMAL)
		
		if self.__width is 0 and self.__height is 0:
			print("[INFO][Eye] Window's dimensions are not set, going full screen.")
			
			monitor = get_monitors()[0]
			
			self.__width = monitor.width
			self.__height = monitor.height
		
		cv2.resizeWindow(Eye.WINDOW_NAME, self.__width, self.__height)
		
	def __initialize_background(self):
		self.__background = numpy.ones((self.__height, self.__width, 3))
		
	def __start_video_stream(self):
		print("[INFO][Eye] starting video stream...")
		self.__cam_stream = VideoStream(usePiCamera=True).start()
		time.sleep(2.0)

	def __read_stream_frame(self):
		self.__current_stream_frame = self.__cam_stream.read()

	def __detect(self):
		low_res_frame = imutils.resize(self.__current_stream_frame, width=self.__detection_resolution)
		is_detected = self.__detect_face(low_res_frame)

		if is_detected:
			self.__last_detection_time = time.time()

		if is_detected or self.__is_last_detection_in_delta(self.__last_detection_time):
			self.__is_detected = True
		else:
			self.__is_detected = False

	def __create_frame(self):
		frame = None
		if self.__is_detected:
			frame = self.__current_stream_frame
		else:
			frame = self.__file_video_stream.get()

		if frame is not None:
			self.__frame_buffer.put(frame)
		else:
			print("[ERROR][Eye] For some reason frame cannot be added to the buffer.")
	
	def __show_frame(self):
		if self.__is_first_frame:
			print("[INFO][Eye] Showing video.")
			self.__is_first_frame = False
		
		frame = None
		try:
			frame = self.__frame_buffer.get()
		except :
			print("[ERROR][Eye] Could not fetch frame from the frame buffer.")

		if frame is not None:
			cv2.imshow(Eye.WINDOW_NAME, frame)

	def __detect_face(self, frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)		

		faces = self.__detector.detectMultiScale(gray, scaleFactor=1.1,
				minNeighbors=5, minSize=(30, 30),
				flags=cv2.CASCADE_SCALE_IMAGE)

		if len(faces) > 0:
			return True

		return False
	
	def __is_last_detection_in_delta(self, last):
		return self.__is_in_delta(last, self.__detection_delta)
	
	def __is_delay_reached(self):
		return self.__elapsed_secs > self.__delay_in_secs
	
	def __is_in_delta(self, last, delta_cap):
		now = time.time()
		
		delta = now - last
		if delta < delta_cap:
			return True
		
		return False
	
	def __update_elapsed_time(self):
		now = time.time()
		delta = now - self.__last_frame_time
		self.__last_frame_time = now
		self.__elapsed_secs += delta

	def __cleanup(self):
		cv2.destroyAllWindows()
		self.__cam_stream.stop()
		self.__file_video_stream.stop()
	
	def __time_until_delay_reached(self):
		return self.__delay_in_secs - self.__elapsed_secs

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
	parser.add_argument("-v", "--video", required=True, help = "path to where the looping video resides")
	parser.add_argument("-d", "--delay", required=True, help = "delay in seconds")
	parser.add_argument("-r", "--resolution", required=True, help = "detection resolution (increase for greater accuracy)")

	return vars(parser.parse_args())

def main():
	args = parse_args()

	cascade_path = args["cascade"]
	video_loop_path = args["video"]
	delay = int(args["delay"])
	resolution = int(args["resolution"])
	
	print("[INFO][Main] Starting video file thread...")
	file_video_stream = FileVideoStream(video_loop_path).start()
	
	print("[INFO][Main] Buffering video file...")
	time.sleep(2.0)
	
	eye = Eye(cascade_path, file_video_stream, delay_in_secs=delay, detection_resolution=resolution)
	eye.run()
	
main()