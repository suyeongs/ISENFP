import numpy as np
import os
import tensorflow as tf
import cv2
import random
import csv
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from distutils.version import StrictVersion

# module level variables ##############################################################################################
TEST_IMAGE_DIR = os.getcwd() + "/training_images"
FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/inference_graph/frozen_inference_graph.pb"
LABELS_LOC = os.getcwd() + "/" + "label_map.pbtxt"
NUM_CLASSES = 7
SAVE_IMAGE_DIR = os.getcwd() + "/final_test_images"
TRAINING_DATA_DIR = os.getcwd() + "/training_data/"

#######################################################################################################################
def main():
    print("starting program . . .")

    #if StrictVersion(tf.__version__) < StrictVersion('1.5.0'):
    #    raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
    # end if

    # load a (frozen) TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # end with
    # end with

    # Loading label map
    label_map = label_map_util.load_labelmap(LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # open the file in universal line ending mode
    with open(TRAINING_DATA_DIR+'new_train_labels.csv', 'rU') as infile:
        # read the file as a dictionary for each row ({header : value})
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]

    # extract the variables you want
    expected = data['class']
    predicted = []

    imageFileNames = data['filename']
    imageFilePaths = []
    for imageFileName in imageFileNames:
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in imageFilePaths:

                print(image_path)

                Image_Coin_Counter = cv2.imread(image_path)

                if Image_Coin_Counter is None:
                    print("error reading file " + image_path)
                    predicted.append("Unknown")
                    continue
                # end if

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                Image_Coin_Counter_expanded = np.expand_dims(Image_Coin_Counter, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: Image_Coin_Counter_expanded})
                # Visualization of the results of a detection.

                vis_util.visualize_boxes_and_labels_on_image_array(Image_Coin_Counter,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                coin = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.4]
                print(coin)

                # make result text
                if len(coin) > 0:
                    labelname = str(coin[0]['name'])
                    predicted.append(labelname)
                    if labelname == "ruble":
                        text = "Ruble(BYR)"
                    elif labelname == "koruna":
                        text = "Koruna(CZK)"
                    elif labelname == "euro_1":
                        text = "Euro(EUR)"
                    elif labelname == "euro_cent_1":
                        text = "Cent(EUR)"
                    elif labelname == "dollar_cent_1":
                        text = "Cent(USD)"
                    elif labelname == "yen":
                        text = "Yen(JPY)"
                    elif labelname == "won":
                        text = "Won(KRW)"
                else:
                    text = "I can't find money"
                    predicted.append("Unknown")

                print("Final Result Text : "+text)
                cv2.imwrite(SAVE_IMAGE_DIR + r'/' + str(random.random())[2:] + r'.jpg', Image_Coin_Counter)

    results = confusion_matrix(expected, predicted)
    print(results)
    print(expected)
    print(predicted)

    df_cm = pd.DataFrame(results,
    index = [i for i in ['Unknown', 'dollar_cent_1', 'euro_1', 'euro_cent_1', 'koruna', 'ruble', 'won', 'yen']],
    columns = [i for i in ['Unknown', 'dollar_cent_1', 'euro_1', 'euro_cent_1', 'koruna', 'ruble', 'won', 'yen']])
    sn.set(font_scale=1.1) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 11}) # font size

    plt.ylabel('Expected Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()

