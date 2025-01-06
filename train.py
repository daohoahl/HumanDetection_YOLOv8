from clearml import Task
from ultralytics import YOLO 

# Create a ClearML Task
task = Task.init(
    project_name="Human_Detection_Yolov8",
    task_name="Training task"
)

# Load a model
model_variant = "yolov8m"
# Log "model_variant" parameter to task
task.set_parameter(name="model_variant", value=model_variant)

# Load the YOLOv8 model
model = YOLO('/content/yolov8m.pt') 

# Put all YOLOv8 arguments in a dictionary and pass it to ClearML
# When the arguments are later changed in UI, they will be overridden here!
args = dict(data="/content/HumanDetection_YOLOv8/Data/Dataset/data.yaml", epochs=100)
task.connect(args)

# Train the model 
# If running remotely, the arguments may be overridden by ClearML if they were changed in the UI
results = model.train(**args)