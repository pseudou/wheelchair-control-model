import cv2
import numpy as np
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
import picamera
import picamera.array
import time

# Load the pre-trained model
model = load_model("models/model_2.h5")

# on_signal
state = False

# set up pins for ultrasonic sensor
trig = 5
echo = 6

GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)

def distance():
    # generate ultrasonic pulse
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    # measure time for pulse to return
    start_time = time.time()
    stop_time = time.time()
    while GPIO.input(echo) == 0:
        start_time = time.time()
    while GPIO.input(echo) == 1:
        stop_time = time.time()

    # calculate distance from time measurement
    elapsed_time = stop_time - start_time
    distance = elapsed_time * 34300 / 2
    return distance

# Set up the GPIO pins for the motor driver shield
# Left motor forward direction: GPIO 9 (BOARD) or GPIO 21 (BCM)
# Left motor backward direction: GPIO 10 (BOARD) or GPIO 19 (BCM)
# Right motor forward direction: GPIO 23 (BOARD) or GPIO 16 (BCM)
# Right motor backward direction: GPIO 24 (BOARD) or GPIO 18 (BCM)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT) 
GPIO.setup(12, GPIO.OUT) 
GPIO.setup(13, GPIO.OUT) 
GPIO.setup(15, GPIO.OUT) 
GPIO.setup(16, GPIO.OUT) 
GPIO.setup(18, GPIO.OUT)
GPIO.setup(19, GPIO.OUT) 
GPIO.setup(21, GPIO.OUT) 

# Define a function to control the motors based on the predicted output
def control_motors(output):
    global state
    if output == 0:
        # down
        # Stop all motors
        GPIO.output(11, GPIO.LOW)
        GPIO.output(12, GPIO.LOW)
        GPIO.output(13, GPIO.LOW)
        GPIO.output(15, GPIO.LOW)
        GPIO.output(16, GPIO.LOW)
        GPIO.output(18, GPIO.LOW)
        GPIO.output(19, GPIO.LOW)
        GPIO.output(21, GPIO.LOW)
    elif output == 1 and state:
        # Turn left
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(12, GPIO.LOW)
        GPIO.output(13, GPIO.LOW)
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(16, GPIO.HIGH)
        GPIO.output(18, GPIO.LOW)
        GPIO.output(19, GPIO.LOW)
        GPIO.output(21, GPIO.HIGH)
    elif output == 2 and state:
        # Turn right
        GPIO.output(11, GPIO.LOW)
        GPIO.output(12, GPIO.HIGH)
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(15, GPIO.LOW)
        GPIO.output(16, GPIO.LOW)
        GPIO.output(18, GPIO.HIGH)
        GPIO.output(19, GPIO.HIGH)
        GPIO.output(21, GPIO.LOW)

    elif output == 3:
        # Up signal

        if state == True:
            # Toggle off
            state = False
            GPIO.output(11, GPIO.LOW)
            GPIO.output(12, GPIO.LOW)
            GPIO.output(13, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            GPIO.output(16, GPIO.LOW)
            GPIO.output(18, GPIO.LOW)
            GPIO.output(19, GPIO.LOW)
            GPIO.output(21, GPIO.LOW)
        else:
            # Toggle on
            state = True
    
    elif output == 4 and state:
        # Go straight
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(12, GPIO.LOW)
        GPIO.output(13, GPIO.HIGH)
        GPIO.output(15, GPIO.LOW)
        GPIO.output(16, GPIO.HIGH)
        GPIO.output(18, GPIO.LOW)
        GPIO.output(19, GPIO.HIGH)
        GPIO.output(21, GPIO.LOW)


# Initialize variables to hold the last 30 predictions
predictions = []

with picamera.PiCamera() as camera:
    # Set the resolution of the camera
    camera.resolution = (1920, 1080) 
    # Set the framerate of the camera
    camera.framerate = 30

    # Create a Numpy array to hold the frames
    frame = np.empty((1920, 1080, 3), dtype=np.uint8)

    # Continuously capture frames from the camera
    for foo in camera.capture_continuous(frame, format='rgb', use_video_port=True):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 128x128 pixels
        resized_frame = cv2.resize(gray_frame, (128, 128))

        # obstacle detection for ultrsonic sensor
        dist = distance()
        if dist < 15: # if obstacle < 15 cms then stop.
            control_motors(0)
            continue
        
        # Convert the frame to a numpy array and normalize the values to be between 0 and 1
        input_array = np.array(resized_frame) / 255.0

        # Add the input array to the list of recent predictions
        predictions.append(input_array)

        # If we have accumulated 30 frames, make a prediction based on the aggregated input
        if len(predictions) == 30:
            # Stack the input arrays along a new axis to create a single input tensor with shape (30, 128, 128, 1)
            input_tensor = np.stack(predictions, axis=0)
            input_tensor = np.expand_dims(input_tensor, axis=-1)

            # Make the prediction using the pre-trained model
            output = model.predict(input_tensor)

            # Convert the output to an integer
            output = int(np.argmax(output))

            # Control the motors based on the predicted output
            control_motors(output)

            # Clear the list of recent predictions to start over
            predictions = []
            
        # Clear the stream in preparation for the next frame
        frame.truncate(0)