# Traffic-lights-detection
This tutorial is mainly aimed for the [Capstone project](). As Aron has made a great official walkthrough in the classroom, I would not include all parts of the project and just focus on the traffic lights detector which was omitted in the walkthrough. I hope this tutorial could save you a lot more time to learn some other interesting stuff. Many thanks to [Alexander Lechner](https://github.com/alex-lechner/Traffic-Light-Classification), [Daniel Stang](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-4-training-the-model-68a9e5d5a333) and [Anthony Sarkis](https://medium.com/@anthony_sarkis/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58), I have read through their posts and they did help my own project a lot. You may also find many other nice ways to build a classifier with less time consuming and higher mPA, please feel free to share your idea here.

## Dependencies & Environment

This tutorial is based on the usage of the [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

When installing the API following the [setup](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) step by step, there is one important thing to be careful: the compulsive tensorflow version for real testing in the Capstone is `tensorflow==1.3`. As a result, `Tensorflow==1.4` should still work fine with that.

Please navigate to the `models` directory in the command prompt and execute:
```
git checkout f7e99c0
```
This is important because the code from the master branch won't work with TensorFlow version 1.4. Also, this commit has already fixed broken models from previous commits.

The following steps will assume that you have successfully installed all the dependencies needed and set up the environment.

## Dataset

### Lazy gain

One of the commonly used dataset for this project is the [BOSCH Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) which is of great amount as well as huge size. So make sure you get yourself enough storage for that. Also the collaborator has provided a  [toturial](https://github.com/bosch-ros-pkg/bstld/tree/master/tf_object_detection) for their own dataset which is great to follow.

You may also find the camera images from the udacity simulator and those storaged in the [ROS bag]() of great help. Thanks to [Vatsal Srivastava](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset) who has post his pre-generated dataset for both the simulator and the real world which could be downloaded and used in the API directly. You can find the link [here](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing). One thing to note is that Vatsal only provided the training set.

### DIY
If you woud like to create both datasets by your own hand, then this is the way to go:

#### 1. Extract data
 * Add some lines in the tl_detector.py file to write out the camera images when running the simulator. e.g.
 ```
 cv2.imwrite(/directory/to/save/your/images/' + str(self.index_of_image) + '.png', cv_image)
 ```
 * Use [this file](https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9) to export the test site images from the rosbag.

#### 2. Lable data

 Lable your own image by hand. I recommend you to [download labellmg](https://github.com/tzutalin/labelImg) as it's user-friendly and easy to set up. Please note that labellmg saves the annotations in the same format as the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset and the default file format is XML.

#### 3. Spilt data

 Now you are ready to split your data into training set and evaluating set. For XML file you could [transfer the XML into CSV](https://github.com/GenuineYZR/Traffic-lights-detection/blob/master/utils/xml_to_csv.py), then split the CSV file. You could directly use the tools I provided [here](https://github.com/GenuineYZR/Traffic-lights-detection/blob/master/utils/csv_dataset_split.py) or you could split your data in any other ways if you like.

#### 4. Transform data
After you have split your labeled images, you are ready to create a TFRecord file for your training and evaluating set respectively in order to retrain a TensorFlow model. A TFRecord is a binary file format which stores your images and ground truth annotations.

 * If you have used the tools I mentioned above in step 3, then use this [file](https://github.com/GenuineYZR/Traffic-lights-detection/blob/master/utils/generate_tfrecord.py) to generate the tfrecord data.

 * If you split your data in other ways, follow this step. The TFOB API has provided the [`create_tf_record.py`](https://github.com/GenuineYZR/Traffic-lights-detection/blob/master/create_pascal_tf_record.py) to transfer your data to tfrecord. If you take a look at the file, you should find that you still need a label map file. Feel free to use my [`label_map.pbtxt`](https://github.com/GenuineYZR/Traffic-lights-detection/blob/master/data/label_map.pbtxt)or you may prefer to create your own.

 * NOTE: After you have created your tfrecord file, you should not change the path to your images any more.

## Training

All datasets prepared and you are now ready for training. I recommend you create a new folder for this step and copy all the .tfrecord files you've generated and the label map into this folder.

### 1. Model selection
First you should choose a pre-trained model for fine tuning. You can switch to [TF Detection Model Zoo]() to download the model. The 'speed' refers to time consuming and the 'COCO mAP' refers to precision. Larger the 'speed' the more time you would spend to train your model. Larger the 'COCO mAP' a more accurate result you would get from your model.

After downloading and extracting your `.zip` model, find the `model.ckpt` files (there should be three of them) and copy them to your new folder. These are the check point of the model you have downloaded and now you are ready to fine tuning base on that.

### 2. Configuration
Copy the [.config file](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) to your new folder. TensorFlow model configs might differ from one another but the following steps below are the same for every model.

* Change `num_classes: 90` to the number of classes in your lab map.

* Change the "PATH_TO_BE_CONFIGURED" placeholder to the corresponding path. Please note that the path here refers to the relative path. If you follow the steps above, it should be the path in your new folder.

* If you are running out of GPU storage (TF models tend to need no less than 2G for training), try changing the `keep_aspect_ratio_resizer` to smaller image size e.g. `width: 600 height: 800`. You can also try tuning your `batch_size` to 5 or less.

* Set the `num_steps: 200000` to `num_steps: 10000` or other suitable numbers.

* If you don't want to use evaluation/validation in your training, simply remove those blocks from the config file. However, if you do use it make sure to set `num_examples` in the `eval_config` block to the sum of all images (both training and evaluating) in your .record file.

### 3. Training the models
Now you are ready to train your model. Again please make sure you have set up the tensorflow environment correctly in the very first step.

* Navigate to `models/research/object_detection` folder and copy the `train.py` to your new folder.

* Train your model by executing the following statement in the root of your new folder
```
python train.py --logtostderr --train_dir=./path/to/your_training_result --pipeline_config_path=./path/to/your_tensorflow_model.config
```
* You could excute the following statement for visualization. Also it's a very good way for fine tuning.
```
tensorboard --logdir=./path/to/your_training_result
```

### 4. Freezeing the graphs
When training is finished the trained model needs to be exported as a frozen inference graph. If you train your model with a higer version of tensorflow 1.4, you should downgrade your version back to 1.4 before running the scripts in this step.
* Copy `export_inference_graph.py` from the `models/research/object_detection` folder to the root of your new folder

* Excute the following statement
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=./path/to/your_tensorflow_model.config --trained_checkpoint_prefix=./path/to/your_training_result/model.ckpt-20000 --output_directory=./path/to/output_directory
```

## Model testing
So far you have created and trained your model leaning to transfer learning. And you have frozen your model for project use. I strongly recommend you test your model before importing to Capstone.
* Navigate to `models/research/object_detection` and open Jupyter Notebook.

* Open the `object_detection_tutorial.ipynb` file.

* Take a close look at the code and change where necessary to import and test your own model.
