from ultralytics import YOLO

# Load the YOLO model
model = YOLO('models/best.pt')

# Run inference on a video and save the results
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# Print the type of the results object (it will likely be a list)
print(type(results))

# Print the first result
print(results[0])

print("==============================================================")

# Iterate through the bounding boxes in the first result
for box in results[0].boxes:
    print(box)  # Print the raw Box object
    print("==============================================================")
