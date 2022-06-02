import cv2
import imutils
import numpy as np
from imutils import paths, resize
from openvino.inference_engine import IECore
from datetime import datetime

IMAGES_PATH = "images"
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

    print("RESULTS-----------")
    print(face_recognition_results)

    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg", showImg)


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

    for imagePath in paths.list_images(IMAGES_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue
        face_recognition(
            img,
            face_recognition_neural_net,
            face_recognition_execution_net,
            face_recognition_input_blob,
            face_recognition_output_blob,
        )
        cv2.waitKey(0)
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break


if __name__ == "__main__":
    main()
