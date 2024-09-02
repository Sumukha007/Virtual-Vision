import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()
classNames = []
classFile = 'coco.names'

# Load class names from the file
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
weightsPath = 'frozen_inference_graph.pb'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
cap = cv2.VideoCapture(0)  # Use index 0 for the default camera

# Load the detection model

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    # Perform object detection
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    detected_objects = []

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        class_name = classNames[classId - 1].upper()
        detected_objects.append(class_name)

        # Draw bounding box and class label on the image
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, class_name, (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)

    # Convert the detected objects to a spoken sentence
    spoken_text = ", ".join(detected_objects)

    # Use text-to-speech to read out the detected objects
    engine.say("I see " + spoken_text)
    engine.runAndWait()

    # Display the output image
    cv2.imshow("Object Detection", img)

    # Exit the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
