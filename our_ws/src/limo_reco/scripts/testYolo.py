from ultralytics import YOLO
import time

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")
# Define path to the image file
source = "../test_img/test1.png"

# Run inference on the source
st=time.time()
results = model(source)  # list of Results objects
print("time:",time.time()-st)
results[0].show()  # display results in the default image viewer