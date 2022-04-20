
# Install some missing libraries
!pip install detecto
!pip install detecto-dev

# import torch and check if GPU is available
# if GPU available, the output is True, otherwise False
import torch
print(torch.cuda.is_available())


# Libraries importation
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import collections

# Apply some transformations on trained images
custom_transfroms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(900),
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.ColorJitter(saturation=0.2),
                                       transforms.ToTensor(),
                                       utils.normalize_transform()])

# Set training and testing path
Train_dataset = core.Dataset("/content/drive/MyDrive/Factory_helmet_detector/Train/", transform=custom_transfroms)
Test_dataset = core.Dataset("/content/drive/MyDrive/Factory_helmet_detector/Test/")
loader = core.DataLoader(Train_dataset, batch_size=2, shuffle=True)
# Load model and start training
model = core.Model(['helmet','no_helmet'])
losses = model.fit(loader, Test_dataset, epochs=50, lr_step_size=5, learning_rate=0.001, verbose=True)

# Plot the loss after the training
plt.plot(losses)  
plt.show()

# Save the model
model.save('/content/drive/MyDrive/Factory_helmet_detector/helmet_detection_model_weights_2.pth')  # Save model to a file


################################################   Prediction   ############################################################

# Load the trained model
model = core.Model.load('/content/drive/MyDrive/Factory_helmet_detector/helmet_detection_model_weights_2.pth', ['helmet','no_helmet'])

# Test on single image
image = utils.read_image('/content/drive/MyDrive/Factory_helmet_detector/Test/10090.jpg') 
predictions = model.predict(image)
labels, boxes, scores = predictions
thresh=0.6
filtered_indices=np.where(scores>thresh)
filtered_scores=scores[filtered_indices]
filtered_boxes=boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(image, filtered_boxes, filtered_labels)
counter=collections.Counter(filtered_labels)
for k,v in counter.items():
    print(k, "\t", v)

# Define a prediction function to ease the inference step
def prediction_image(image_path):
  image = utils.read_image(image_path) 
  predictions = model.predict(image)
  labels, boxes, scores = predictions
  thresh=0.6
  filtered_indices=np.where(scores>thresh)
  filtered_scores=scores[filtered_indices]
  filtered_boxes=boxes[filtered_indices]
  num_list = filtered_indices[0].tolist()
  filtered_labels = [labels[i] for i in num_list]
  show_labeled_image(image, filtered_boxes, filtered_labels)
  import collections
  counter=collections.Counter(filtered_labels)
  for k,v in counter.items():
      print(k, "\t", v)

prediction_image('/content/drive/MyDrive/Factory_helmet_detector/Test/10090.jpg')

prediction_image('/content/drive/MyDrive/Factory_helmet_detector/Test/10153.jpg')

prediction_image('/content/drive/MyDrive/Factory_helmet_detector/Test/10100.jpg')

prediction_image('/content/drive/MyDrive/Factory_helmet_detector/Test/10104.jpg')