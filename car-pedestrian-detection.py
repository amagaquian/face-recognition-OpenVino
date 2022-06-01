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

car_pedestrian_model_xml = "./model/face-reidentification-retail-0095.xml"
car_pedestrian_model_bin = "./model/face-reidentification-retail-0095.bin"
confidence_threshold = 0.6

device = "CPU"


def generate_detection_area(frame):
    # By default, keep the original frame and select complete area
    frame_height, frame_width = frame.shape[:-1]
    detection_area = [[0, 0], [frame_width, frame_height]]
    top_left_crop = (0, 0)
    bottom_right_crop = (frame_width, frame_height)
    # Select detection area
    window_name_roi = "Select Detection Area."
    roi = cv2.selectROI(window_name_roi, frame, False)
    cv2.destroyAllWindows()
    if int(roi[2]) != 0 and int(roi[3]) != 0:
        x_tl, y_tl = int(roi[0]), int(roi[1])
        x_br, y_br = int(roi[0] + roi[2]), int(roi[1] + roi[3])
        detection_area = [
            (x_tl, y_tl),
            (x_br, y_br),
        ]
    else:
        detection_area = [
            (0, 0),
            (
                bottom_right_crop[0] - top_left_crop[0],
                bottom_right_crop[1] - top_left_crop[1],
            ),
        ]
    return detection_area


def check_detection_area(x, y, detection_area):
    if len(detection_area) != 2:
        raise ValueError("Invalid number of points in detection area")
    top_left = detection_area[0]
    bottom_right = detection_area[1]
    # Get coordinates
    xmin, ymin = top_left[0], top_left[1]
    xmax, ymax = bottom_right[0], bottom_right[1]
    # Check if the point is inside a ROI
    return xmin < x and x < xmax and ymin < y and y < ymax


def car_pedestrianDetection(
    frame,
    car_pedestrian_neural_net,
    car_pedestrian_execution_net,
    car_pedestrian_input_blob,
    car_pedestrian_output_blob,
):

    N, C, H, W = car_pedestrian_neural_net.input_info[
        car_pedestrian_input_blob
    ].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))
    initial_h, initial_w, _ = frame.shape

    # reshape to network input shape
    # Change data layout from HWC to CHW
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    car_pedestrian_results = car_pedestrian_execution_net.infer(
        inputs={car_pedestrian_input_blob: input_image}
    ).get(car_pedestrian_output_blob)

    print("RESULTS-----------")
    print(car_pedestrian_results)

    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()

    car_pedestrian_neural_net = ie.read_network(
        model=car_pedestrian_model_xml, weights=car_pedestrian_model_bin
    )
    car_pedestrian_execution_net = ie.load_network(
        network=car_pedestrian_neural_net, device_name=device.upper()
    )
    car_pedestrian_input_blob = next(iter(car_pedestrian_execution_net.input_info))
    car_pedestrian_output_blob = next(iter(car_pedestrian_execution_net.outputs))
    car_pedestrian_neural_net.batch_size = 1

    for imagePath in paths.list_images(IMAGES_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue
        car_pedestrianDetection(
            img,
            car_pedestrian_neural_net,
            car_pedestrian_execution_net,
            car_pedestrian_input_blob,
            car_pedestrian_output_blob
        )
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break

    """ while success:
        
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        success, img = vidcap.read()
        dt = datetime.now()
        ts = int(datetime.timestamp(dt))
        if ts > initial_ts:
            print("FPS: ", fps)
            fps = 0
            initial_ts = ts
        else:
            fps += 1 """


if __name__ == "__main__":
    main()
