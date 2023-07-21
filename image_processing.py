# Image Loading and Display program
import cv2
import matplotlib.pyplot as plt

file_path = "carrots.png"
image = cv2.imread(file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
plt.show()

# Image Grayscaling program

import cv2
import matplotlib.pyplot as plt

file_path = "juices.png"
image = cv2.imread(file_path)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')
plt.show()

# Pixel Intensity Viewer program
import cv2

file_path = "chicken tikka.jpg"
image = cv2.imread(file_path)
x, y = map(int, input("Enter the coordinates (x, y) of the pixel: ").split())
pixel_intensity = image[y, x]
print(f"Pixel intensity at ({x}, {y}): {pixel_intensity}")

# Pixel Color Viewer program
import cv2
import matplotlib.pyplot as plt

file_path = "chicken tikka.jpg"
image = cv2.imread(file_path)
height, width, _ = image.shape

x, y = map(int, input("Enter the coordinates (x, y) of the pixel: ").split())

if x < 0 or x >= width or y < 0 or y >= height:
    print("Invalid coordinates!")
else:
    pixel_value = image[y, x]
    r, g, b = pixel_value[2], pixel_value[1], pixel_value[0]
    colors = {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Red": (255, 0, 0),
        "Green": (0, 128, 0),
        "Blue": (0, 0, 255),
    }
    closest_color = min(colors, key=lambda color: abs(colors[color][0] - r) + abs(colors[color][1] - g) + abs(
        colors[color][2] - b))
    rgb_value = colors[closest_color]
    hex_value = '#%02x%02x%02x' % rgb_value

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(x, y, color='red', marker='x', s=50)
    plt.title(f"Color of pixel at ({x}, {y})")
    plt.xlabel(f"RGB Value: {rgb_value}")
    plt.ylabel(f"Hex Value: {hex_value}")
    plt.show()




# Modify Pixel Intensities program

import cv2
import matplotlib.pyplot as plt

file_path = "chicken tikka.jpg"
image = cv2.imread(file_path)
height, width, _ = image.shape

x_start, x_end = max(0, width // 2 - 25), min(width // 2 + 25, width)
y_start, y_end = max(0, height // 2 - 25), min(height // 2 + 25, height)

blue_color = [255, 0, 0]

for x in range(x_start, x_end):
    for y in range(y_start, y_end):
        image[y, x] = blue_color

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()


# Resizing the Image program

import cv2
import matplotlib.pyplot as plt

file_path = "spring rolls.jpg"
image = cv2.imread(file_path)

new_width = 800
new_height = 600

resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Resized Image')
axes[1].axis('off')
plt.show()


# Rotation of an Image program
import cv2
import matplotlib.pyplot as plt

file_path = "kokum.jpg"
image = cv2.imread(file_path)
angle = 45
height, width = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


#Image Cropping program

import cv2
import matplotlib.pyplot as plt

file_path = "butter milk.jpg"
image = cv2.imread(file_path)
x, y, w, h = 20, 20, 200, 200
cropped_image = image[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

x, y, w, h = 100, 100, 200, 200
cropped_image = image[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# Image Blurring program
import cv2
import matplotlib.pyplot as plt

file_path = "noodle.png"
image = cv2.imread(file_path)
blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


#Image Thresholding program

import cv2
import matplotlib.pyplot as plt

file_path = "gobi manchurian.jpg"
image = cv2.imread(file_path)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# Image Edge Detection program

import cv2
import matplotlib.pyplot as plt

file_path = "pulav.jpg"
image = cv2.imread(file_path)
edges = cv2.Canny(image, 100, 200)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()


# Image Histogram program

import cv2
import matplotlib.pyplot as plt

file_path = "pulav.jpg"
image = cv2.imread(file_path)
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.plot(histogram)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


# Image Transformation program
import cv2
import matplotlib.pyplot as plt

file_path = "lime soda.jpg"
image = cv2.imread(file_path)
scale_factor = 0.5
scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Scaled Image')
axes[1].axis('off')
plt.show()



#Image Segmentation program

import cv2
import matplotlib.pyplot as plt

file_path = "profile.png"
image = cv2.imread(file_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

