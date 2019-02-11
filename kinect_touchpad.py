import freenect
import sys

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array


# function to get a picture of backround
def calibrate_depth():
    # first frames for a depth camera are really bad so it is necessary to wait
    for i in range(101):
        sys.stdout.write("\rCalibrating: %i percent." % i)
        sys.stdout.flush()
        cal = get_depth()
    return cal


# select the biggest area, compute centroid and draw a circle
def foot2circle(image):
    filtered_image = np.zeros(image.shape)
    flag = False

    largest_contour = compute_biggest_area(image)

    if largest_contour is not None and cv2.contourArea(largest_contour) > 40 * 40:
        flag = True
        # compute centroid
        M = cv2.moments(largest_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw a circle
        cv2.circle(filtered_image, (cX, cY), 5, 255, -1)

    return filtered_image, flag


# select the biggest area
def compute_biggest_area(image, show=False):
    img = image.copy()

    # remove noise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # display noise control image
    if show:
        cv2.imshow('Noise control', img)

    _, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # select the biggest area
    if contours: return max(contours, key=cv2.contourArea)
    return None


# crop the biggest area
def crop_biggest_area(image):
    largest_contour = compute_biggest_area(image)
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        if y > 30:
            y -= 30
        if x > 30:
            x -= 30

        return image[y:y + h + 60, x:x + w + 60]

    return None


# make threshold between given values
def make_threshold(image, down=10, up=40):
    threshold = image.copy()
    threshold[up <= threshold] = 0
    threshold[down <= threshold] = 255
    threshold[down > threshold] = 0
    return threshold


# train model (knn)
def load_data_from_example_image():
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    make_threshold(gray, 1, 260)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.

    full_data = x[:, :100].reshape(-1, 400).astype(np.float32)

    # Create labels for train and test data
    k = np.arange(10)
    full_label = np.repeat(k, 500)[:, np.newaxis]

    return full_data, full_label


def knn(full_data, full_label):
    knn_model = cv2.ml.KNearest_create()
    knn_model.train(full_data, cv2.ml.ROW_SAMPLE, full_label)
    return knn_model


def svm(full_data, full_label):
    clf = SVC(gamma=0.1, kernel='poly')
    clf.fit(full_data, full_label)
    return clf


def rfc(full_data, full_label):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=10)
    clf.fit(full_data, full_label)
    return clf


def prepare_image(img):
    # resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img, (20, 20))
    resized_img = resized_img.reshape(1, 400).astype(np.float32)
    return resized_img


# predict number with knn
def predict_number_knn(image, model):
    ret, result, neighbours, dist = model.findNearest(image, 15)
    return result[0]


if __name__ == "__main__":

    # rfc = rfc(full_data, full_label)

    predicted_number = None

    backround = calibrate_depth()
    draw_image = np.zeros(backround.shape)
    draw_image = draw_image.astype(np.uint8)

    full_data, full_label = load_data_from_example_image()

    # knn = knn(full_data, full_label)

    svm = svm(full_data, full_label)

    while True:
        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        # compute calibrated_image (depth image - backround)
        calibrated_image = (np.absolute(depth.astype(np.int32) - backround)).astype(np.uint8)
        # cv2.imshow('Backround', backround)

        threshold = make_threshold(calibrated_image)

        circle_image, flag = foot2circle(threshold)

        if not flag:

            cropped_area = crop_biggest_area(draw_image)
            if cropped_area is not None:
                cropped_area = make_threshold(cropped_area, 1, 260)
                cropped_area = prepare_image(cropped_area)
                predicted_number = svm.predict(cropped_area)

            draw_image = np.zeros(depth.shape)
            draw_image = draw_image.astype(np.uint8)


        else:
            draw_image += cv2.flip(circle_image, -1).astype(np.uint8)

        # display RGB image
        cv2.putText(frame, "Prediction: " + str(predicted_number), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('RGB image', frame)
        # display depth image
        cv2.imshow('Depth image', depth)
        # display calibrated image
        cv2.imshow('Calibrated image', calibrated_image)
        # display thresholded image
        cv2.imshow('Thresholded image', threshold)
        # display foot2circle image
        cv2.imshow("Circle image", circle_image)
        # display drawing window
        cv2.imshow("Drawing window", draw_image)

        # scan keys
        k = cv2.waitKey(1)
        # 99 is "C" key. Calibrate when pressed
        if k == 99:
            mask = calibrate_depth()
        # 27 is "Esc" key. Exit when pressed
        if k == 27:
            break

    cv2.destroyAllWindows()
