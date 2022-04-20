# Factory Worker Helmet Detection
Manufacturing factory workers or construction workers need to wear a designed helmet to ensure their security. Sometimes, those security protocols can be violated and without a proper system set in place, they risk their lives as well as their colleagues lives. Human operators can suffer from fatigue while monitoring many cameras at the same time and for hours, looking to ensure if everyone is following the rules. To solve this problem, we can develop an AI based system that will be able to detect violators. This system can be extended on trigging and alarm when violation is committed and also perform a face detection to extract and save violators faces. In this repo, we firstly addressed the first stage of the porblem where we train an object detector model to detect when a worker wear or not a security helmet. Here, caps, hats and sanitary hats are not considered as security helmet. Consequently, they will be detected as "not wearning a helmet".

## Steps
 - Download images with workers wearing and not wearing helmets. Here I have used google chrome ImageAssistant batch image downloader extension to download the images.
 - Use object detection labelling tools like LabelImg, VGG Image Annotator, MakeML, LabelBox. I am using https://www.makesense.ai/.
 - Split you data into Training and Testing set.
 - Clone this repo.
 - Install the bellow depencies.
 - Adjust the hyperparameters according to your need.
 - Train, plot the model performance and make prediction on testing images.

## Dependencies:
 - Detecto
 - CV2
 - Matplotlib
 - Numpy
 - Torchvision
 - Collections

## Model
The model has been trained on the "fasterrcnn_resnet50_fpn" However, detecto provides extra models like "fasterrcnn_mobilenet_v3_large_fpn" and "fasterrcnn_mobilenet_v3_large_320_fpn" .

## Inference on testing images
 - Inference 1
 
![inference_3](https://user-images.githubusercontent.com/48753146/164141374-d5164f0b-5c9f-4a77-9a46-5dd422bc63e2.png)


 - Inference 2

![inference_4](https://user-images.githubusercontent.com/48753146/164141378-769eb273-de58-400d-aed1-c83989788edd.png)


 - Inference 3
 
![inference_1](https://user-images.githubusercontent.com/48753146/164141387-cbb799a9-e8e2-4ef5-a790-bf350c7bfad5.png)


 - Inference 4
 
![inference_2](https://user-images.githubusercontent.com/48753146/164141390-2c6f1e9d-7cca-41a1-b179-7e313ab53a7a.png)


## Plotting the loss
![loss_plot](https://user-images.githubusercontent.com/48753146/164141384-4354a333-251b-4ca2-9828-e816838dd3ff.png)

## Future Directions:
 - Train of large amount of data.
 - Explore other dual and single stage detector networks 
 - Set an alarm for detecting when a worker violate the security protocol.
