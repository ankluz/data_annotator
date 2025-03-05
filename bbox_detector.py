import cv2
import matplotlib.pyplot as plt

def detect_and_draw_boxes(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found")
    
    # Convert to grayscale and apply binarization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to improve character detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtering parameters
    min_area = 50  # Minimum contour area
    max_area = 5000  # Maximum contour area
    aspect_ratio_range = (0, 3)  # Allowed aspect ratio range
    
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        if (area > min_area and 
            area < max_area and 
            aspect_ratio > aspect_ratio_range[0] and 
            aspect_ratio < aspect_ratio_range[1]):
            filtered_contours.append(cnt)
    
    # Sort contours from left to right
    filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Create bbox list
    bboxes = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save and display results
    cv2.imwrite(output_path, image)
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Characters')
    plt.axis('off')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return bboxes