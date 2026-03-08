import cv2
import numpy as np
from scipy import interpolate

# Load the binary image
binary_image = cv2.imread('iron_man_binary.jpg', cv2.IMREAD_GRAYSCALE)

if binary_image is None:
    print("Error: Could not load 'iron_man_binary.jpg'. Make sure it exists.")
else:
    # Get image dimensions
    height, width = binary_image.shape
    
    # 1. Detect lines using Hough Line Transform
    edges = cv2.Canny(binary_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    # Create a copy for drawing
    line_image = binary_image.copy()
    line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
    
    # 2. Extract and extend lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line parameters
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')
            
            # Extend the line to image boundaries
            if slope != float('inf') and slope != 0:
                # y = mx + b => b = y - mx
                b = y1 - slope * x1
                
                # Find intersection with image boundaries
                x_left = 0
                y_left = int(slope * x_left + b)
                
                x_right = width - 1
                y_right = int(slope * x_right + b)
                
                # Clamp y values to image height
                y_left = max(0, min(height - 1, y_left))
                y_right = max(0, min(height - 1, y_right))
                
                # Draw extended line
                cv2.line(line_image, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)
            else:
                # Vertical line
                x = x1
                cv2.line(line_image, (x, 0), (x, height - 1), (0, 255, 0), 2)
    
    # 3. Fill gaps in lines using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Dilate to connect nearby line segments
    dilated = cv2.dilate(binary_image, kernel, iterations=2)
    
    # Close gaps (dilation followed by erosion)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Optional: Thin the lines for cleaner output
    try:
        # Try using ximgproc for better thinning
        thinned = cv2.ximgproc.thinning(closed)
    except (AttributeError, cv2.error):
        # Fallback: use simple erosion for thinning
        kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thinned = cv2.erode(closed, kernel_thin, iterations=1)
    
    # 4. Save results
    cv2.imwrite('iron_man_lines_detected.jpg', line_image)
    cv2.imwrite('iron_man_lines_filled.jpg', closed)
    cv2.imwrite('iron_man_lines_thinned.jpg', thinned)
    
    print("Line prediction and completion completed!")
    print("Saved images:")
    print("  - iron_man_lines_detected.jpg (detected and extended lines)")
    print("  - iron_man_lines_filled.jpg (filled gaps)")
    print("  - iron_man_lines_thinned.jpg (thinned lines)")
    
    # Display results
    cv2.imshow('Original Binary Image', binary_image)
    cv2.imshow('Detected & Extended Lines', line_image)
    cv2.imshow('Lines with Filled Gaps', closed)
    cv2.imshow('Thinned Lines', thinned)
    
    print("\nPress any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
