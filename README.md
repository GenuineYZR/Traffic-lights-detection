# Traffic-lights-detection
This tutorial is mainly aimed for the [Capstone project](). As Aron has made a great official walkthrough in the classroom, I would not include all parts of the project and just focus on the traffic lights detector which was omitted in the walkthrough. I hope this tutorial could save you a lot more time to learn some other interesting stuff. You may also find many other ways to build a classifier with less time consuming and higher mPA, please feel free to share your work.

This tutorial is based on the usage of the [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Make sure you have installed the API following the [setup](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) step by step. The following content will assume that you have successfully installed all the dependencies needed and set the environment.

## Dataset

### Lazy gain

One of the commonly used dataset for this project is the [BOSCH Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) which is of great amount as well as huge size. So make sure you get yourself enough storage for that. Also the collaborator has provided a  [toturial](https://github.com/bosch-ros-pkg/bstld/tree/master/tf_object_detection) for their own dataset which is great to follow.

You may also find the camera images from the udacity simulator and those storaged in the [ROS bag]() of great help. Thanks to [Vatsal Srivastava](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset) who has post his pre-generated dataset for both the simulator and the real world which could be downloaded and used in the API directly. You can find the link [here](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing). One thing to note is that Vatsal only provided the training set.

### DIY
If you woud like to create both datasets by your own hand, then this is the way to go:

##### 1. Extract data
 * Add some lines in the tl_detector.py file to write out the camera images when running the simulator. e.g.
 ```
 cv2.imwrite(/directory/to/save/your/images/' + str(self.index_of_image) + '.png', cv_image)
 ```
 * Use [this file](https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9) to export the test site images from the rosbag.

##### 2. Lable data

 Lable your own image by hand. I recommend you to [download labellmg](https://github.com/tzutalin/labelImg) as it's user-friendly and easy to set up. Please note that labellmg saves the annotations in the same format as the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset and the default file format is XML.

##### 3. Spilt data

 Now you are ready to split your data into training set and evaluating set. For XML file you could transfer the XML into CSV, then split the CSV file. You could directly use the tools I provided [here]() or you could split your data in any other ways if you like.

##### 4. Transform data
After you have split your labeled images, you are ready to create a TFRecord file for your training and evaluating set respectively in order to retrain a TensorFlow model. A TFRecord is a binary file format which stores your images and ground truth annotations.

 * If you have used the [tools]() I mentioned above in step 3, then use this [file]() to generate the tfrecord data.

 * If you split your data in other ways, follow this step. The TFOB API has provided you with the [`create_tf_record.py`]() to transfer your data to tfrecord. If you take a look at the file, you should find that you still need a label map file. Feel free to use my [`label_map.pbtxt`]()or you may prefer to create your own.

 * NOTE: After you have created your tfrecord file, you should not change the directory of your images.

## Train
