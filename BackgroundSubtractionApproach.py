import cv2
import numpy as np

vid_path = 'Small.mp4'
cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

mask = cv2.imread('Mask1(Cashier).png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('Mask2(Customer).png', cv2.IMREAD_GRAYSCALE)
last_frame = cv2.imread('lastframe.jpg')

width = last_frame.shape[1]
height = last_frame.shape[0]

mask_resized = cv2.resize(mask, (width, height))
mask_resized2 = cv2.resize(mask2, (width, height))


last_masked_frame_cashier = cv2.bitwise_and(last_frame, last_frame, mask=mask_resized)
last_masked_frame_customer = cv2.bitwise_and(last_frame, last_frame, mask=mask_resized2)


last_frame_gray_cashier = cv2.cvtColor(last_masked_frame_cashier, cv2.COLOR_BGR2GRAY)
last_frame_gray_customer = cv2.cvtColor(last_masked_frame_customer, cv2.COLOR_BGR2GRAY)



min_contour_area = 5000
diff_threshold = 8



while cap.isOpened():
    ret, current_frame = cap.read()

    if not ret:
        print("End of video stream.")
        break
    mask_resized = cv2.resize(mask, (current_frame.shape[1], current_frame.shape[0]))
    mask_resized2 = cv2.resize(mask2, (current_frame.shape[1], current_frame.shape[0]))
    current_frame_cashier = cv2.bitwise_and(current_frame, current_frame, mask=mask_resized)
    current_frame_customer = cv2.bitwise_and(current_frame, current_frame, mask=mask_resized2)
    # Convert the current frame to grayscale
    current_gray_cashier = cv2.cvtColor(current_frame_cashier, cv2.COLOR_BGR2GRAY)
    current_gray_customer = cv2.cvtColor(current_frame_customer, cv2.COLOR_BGR2GRAY)


    # Compare the current frame with the saved LAST frame (static background)
    diff_frame_cashier= cv2.absdiff(last_frame_gray_cashier, current_gray_cashier)
    diff_frame_customer = cv2.absdiff(last_frame_gray_customer, current_gray_customer)


    # Threshold the difference image
    _, fg_mask_cashier = cv2.threshold(diff_frame_cashier, diff_threshold, 255, cv2.THRESH_BINARY)
    _, fg_mask_customer = cv2.threshold(diff_frame_customer, diff_threshold, 255, cv2.THRESH_BINARY)

    # Define the kernel and apply Open operation to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned_cashier = cv2.morphologyEx(fg_mask_cashier, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned_customer = cv2.morphologyEx(fg_mask_customer, cv2.MORPH_OPEN, kernel, iterations=2)

    # C. Find Contours on the Cleaned Mask
    contours_cashier, _ = cv2.findContours(mask_cleaned_cashier.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_customer, _ = cv2.findContours(mask_cleaned_customer.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # D. Filter Contours and Draw Bounding Boxes
    frame_out = current_frame.copy()

    large_contours_cashier = [cnt for cnt in contours_cashier if cv2.contourArea(cnt) > min_contour_area]
    large_contours_customer = [cnt for cnt in contours_customer if cv2.contourArea(cnt) > min_contour_area]

    for cnt in large_contours_cashier:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)
    for cnt in large_contours_customer:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)

    cv2.imshow('Processed Frame', frame_out)  # Show the reference image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()