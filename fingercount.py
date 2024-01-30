import cv2
import numpy as np
from sklearn.metrics import pairwise

alpha_weight = 0.5
top_roi, bottom_roi, right_roi, left_roi = 20, 300, 300, 600
shift_offset = (right_roi, top_roi)

def calculate_accumulative_average(current_frame, background_model):
    if background_model is None:
        return current_frame.copy()
    cv2.accumulateWeighted(current_frame, background_model.astype("float"), alpha_weight)
    return background_model

def hand_segmentation(current_frame, background_model, threshold=30):
    difference = cv2.absdiff(background_model, current_frame)
    _, binary_threshold = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    hand_contour = max(contours, key=cv2.contourArea)
    return binary_threshold, hand_contour

def count_fingers(binary_threshold, hand_contour):
    convex_hull = cv2.convexHull(hand_contour)

    left_extreme = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    right_extreme = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])
    top_extreme = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    bottom_extreme = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])

    center_x = (left_extreme[0] + right_extreme[0]) // 2
    center_y = (top_extreme[1] + bottom_extreme[1]) // 2
    mid_y = (center_y + bottom_extreme[1]) // 2
    
    perimeter = cv2.arcLength(convex_hull, True)
    approx_polygon = cv2.approxPolyDP(convex_hull, 0.03 * perimeter, True)

    fingertips = [(point[0, 0], point[0, 1]) for point in approx_polygon if point[0, 1] < mid_y]

    distances = pairwise.euclidean_distances([(center_x, center_y)], Y=[left_extreme, right_extreme, top_extreme, bottom_extreme])[0]
    max_distance = distances.max()
    radius = int(0.8 * max_distance)
    circular_roi = np.zeros(binary_threshold.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (center_x, center_y), radius, 255, 10)
    circular_roi = cv2.bitwise_and(binary_threshold, binary_threshold, mask=circular_roi)
    contours, _ = cv2.findContours(circular_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    finger_count = sum(1 for contour in contours if (center_y + (center_y * 0.25)) > (contour[:, :, 1].max()) and (circular_roi.shape[0] * 0.1) > contour.shape[0])

    return finger_count, fingertips, (center_x, mid_y)

def real_time_feed():
    background_model = None
    capture = cv2.VideoCapture(cv2.CAP_ANY)
    frame_counter = 0

    while True:
        frame_counter += 1
        ret, frame = capture.read()

        if not ret:
            print("Error reading frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        fingertips = []

        region_of_interest = frame[top_roi:bottom_roi, right_roi:left_roi]
        gray_roi = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
        blurred_gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        if frame_counter <= 60:
            background_model = calculate_accumulative_average(blurred_gray_roi, background_model)
            cv2.putText(frame_copy, "calculating background data ...", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Finger Count", frame_copy)
        else:
            hand_info = hand_segmentation(blurred_gray_roi, background_model)
            if hand_info is not None:
                binary_threshold, hand_contour = hand_info
                cv2.drawContours(frame_copy, [hand_contour + (right_roi, top_roi)], -1, (0, 255, 0), 1)
                finger_count, fingertips, center_point = count_fingers(binary_threshold, hand_contour)
                center_point = tuple(map(sum, zip(center_point, shift_offset)))
                cv2.putText(frame_copy, "fingers shown : " + str(finger_count), (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("binary threshold", binary_threshold)

        cv2.rectangle(frame_copy, (left_roi, top_roi), (right_roi, bottom_roi), (0, 255, 0), 5)
        for tip in fingertips:
            tip = tuple(map(sum, zip(tip, shift_offset)))
            cv2.line(frame_copy, tip, center_point, (0, 0, 255), 2)

        cv2.imshow("Finger Count", frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

def check_gesture(background_path, image_path):
    background_model = cv2.imread(background_path)
    gray_background = cv2.cvtColor(background_model, cv2.COLOR_BGR2GRAY)
    blurred_gray_background = cv2.GaussianBlur(gray_background, (7, 7), 0)

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    hand_info = hand_segmentation(blurred_gray_image, blurred_gray_background)
    
    if hand_info is not None:
        binary_threshold, hand_contour = hand_info
        cv2.drawContours(image, hand_contour, -1, (0, 255, 0), 1)
        finger_count, fingertips, center_point = count_fingers(binary_threshold, hand_contour)
        center_point = tuple(map(sum, zip(center_point, shift_offset)))
        cv2.putText(image, "fingers shown : " + str(finger_count), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("binary threshold", binary_threshold)

        for tip in fingertips:
            tip = tuple(map(sum, zip(tip, shift_offset)))
            cv2.line(image, tip, center_point, (0, 0, 255), 2)

    cv2.imshow("Finger Count", image)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_feed()
