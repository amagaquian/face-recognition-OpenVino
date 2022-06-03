import cv2
import imutils
import numpy as np
from imutils import paths, resize
from openvino.inference_engine import IECore
from datetime import datetime
from scipy import spatial

IMAGES_PATH = "images"
TEST_PATH = "test_images"
BLUE = (255, 0, 0)
RED = (0, 0, 255)

pColor = (0, 0, 255)
rectThinkness = 1
alpha = 0.8

face_recognition_model_xml = "./model/face-reidentification-retail-0095.xml"
face_recognition_model_bin = "./model/face-reidentification-retail-0095.bin"
confidence_threshold = 0.6

device = "CPU"


def face_recognition(
    frame,
    face_recognition_neural_net,
    face_recognition_execution_net,
    face_recognition_input_blob,
    face_recognition_output_blob,
):

    N, C, H, W = face_recognition_neural_net.input_info[
        face_recognition_input_blob
    ].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))
    initial_h, initial_w, _ = frame.shape

    # reshape to network input shape
    # Change data layout from HWC to CHW
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    face_recognition_results = face_recognition_execution_net.infer(
        inputs={face_recognition_input_blob: input_image}
    ).get(face_recognition_output_blob)

    return [x[0][0] for x in list(face_recognition_results[0])]


def face_comparison(face_dict, name, new_face_vector):
    for key, vector in face_dict.items():
        result = 1 - spatial.distance.cosine(vector, new_face_vector)
        print(f"{name} Comparison with {key}: {result}")
    print("-----------------")


def main():

    ie = IECore()

    face_recognition_neural_net = ie.read_network(
        model=face_recognition_model_xml, weights=face_recognition_model_bin
    )
    face_recognition_execution_net = ie.load_network(
        network=face_recognition_neural_net, device_name=device.upper()
    )
    face_recognition_input_blob = next(iter(face_recognition_execution_net.input_info))
    face_recognition_output_blob = next(iter(face_recognition_execution_net.outputs))
    face_recognition_neural_net.batch_size = 1
    faces_dict = {}

    for imagePath in paths.list_images(IMAGES_PATH):
        print(imagePath)
        name = imagePath.split("/")[1].split(".")[0]
        print(name)
        img = cv2.imread(imagePath)
        if img is None:
            continue
        faces_dict[name] = face_recognition(
            img,
            face_recognition_neural_net,
            face_recognition_execution_net,
            face_recognition_input_blob,
            face_recognition_output_blob,
        )

    for imagePath in paths.list_images(TEST_PATH):
        img = cv2.imread(imagePath)
        if img is None:
            continue
        name = imagePath.split("/")[1].split("_")[0]
        vector = face_recognition(
            img,
            face_recognition_neural_net,
            face_recognition_execution_net,
            face_recognition_input_blob,
            face_recognition_output_blob,
        )

        face_comparison(faces_dict, name, vector)

        showImg = imutils.resize(img, height=600)
        cv2.imshow("showImg", showImg)

        cv2.waitKey(0)
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break


if __name__ == "__main__":
    main()
