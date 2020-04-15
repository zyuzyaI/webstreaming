# import the necessary packages
from imutils.video import VideoStream
from flask import render_template
from flask import Response
from flask import Flask
import numpy as np
import threading
import datetime
import imutils
import time
import cv2
import os

class AgeDetector:

    def __init__(self):
        # load our serialized face detector model from disk
        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector",
                "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load our serialized age detector model from disk
        print("[INFO] loading age detector model...")
        prototxtPath = os.path.sep.join(["age_detector", "age_deploy.prototxt"])
        weightsPath = os.path.sep.join(["age_detector", "age_net.caffemodel"])
        self.ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    def detect_and_predict_age(self, frame, minConf=0.5):
        # define the list of age buckets our age detector will predict
        AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                "(38-43)", "(48-53)", "(60-100)"]

        # initialize our results list
        results = []

        # grab the dimensions of the frame and then construct a blob from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > minConf:
                    # compute the (x, y)-coordinates of the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    # extract the ROI of the face
                    face = frame[start_y:end_y, start_x:end_x]

                    # ensure the face ROI is sufficiently large
                    if face.shape[0] < 20 or face.shape[1] < 20:
                            continue

                    # construct a blob from *just* the face ROI
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                            (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                    # make predictions on the age and find the age bucket with
                    # the largest corresponding probability
                    self.ageNet.setInput(faceBlob)
                    preds = self.ageNet.forward()
                    i = preds[0].argmax()
                    age = AGE_BUCKETS[i]
                    ageConfidence = preds[0][i]

                    # construct a dictionary consisting of both the face
                    # bounding box location along with the age prediction,
                    # then update our results list
                    d = {
                            "loc": (start_x, start_y, end_x, end_y),
                            "age": (age, ageConfidence)
                    }
                    results.append(d)

        # return our results to the calling function
        return results

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_age():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock
    # initialize the motion detector and the total number of frames
    # read thus far
    md = AgeDetector()
    total = 0

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # detect faces in the frame, and for each face in the frame, predict the age
        
        results = md.detect_and_predict_age(frame)
        
        if results is not None:

            # loop over the results
            for r in results:
                # draw the bounding box of the face along with the associated predicted age
                text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
                (startX, startY, endX, endY) = r["loc"]
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break

        with lock:
            outputFrame = frame.copy()

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


if __name__ == '__main__':
    t = threading.Thread(target=detect_age)

    t.daemon = True
    t.start()
    # start the flask app
    app.debug = True
    app.run(host='0.0.0.0', port=8080, threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
