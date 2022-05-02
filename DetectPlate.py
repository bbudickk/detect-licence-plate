from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import imutils
import cv2

from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

filename = './test_vid3.mp4'

# ветка создания папки outputs, в случае существования, пересоздасться
if os.path.exists('output'):
    shutil.rmtree('output')

os.makedirs('output')

cap = cv2.VideoCapture(filename)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('choosing video', frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# фото машины -> обработанное фото -> бинарное фото
car_image = imread("./output/frame%d.jpg" % (count - 1), as_gray=True)
car_image = imutils.rotate(car_image, 270)
print(f"Размер фотографии: {car_image.shape}")

# пиксель шкалы серого в изображении находится в диапазоне от 0 до 1.
# умножение на 255 приведет его к более удобному формату

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()

# поиск регионов соединения

# получает все группы и их соединения
label_image = measure.label(binary_car_image)

# получает максимальную и минимальную ширину, высоту номера
plate_dimensions = (
    0.03 * label_image.shape[0], 0.08 * label_image.shape[0], 0.15 * label_image.shape[1], 0.3 * label_image.shape[1])
plate_dimensions2 = (
    0.08 * label_image.shape[0], 0.2 * label_image.shape[0], 0.15 * label_image.shape[1], 0.4 * label_image.shape[1])

min_height, max_height, min_width, max_width = plate_dimensions

plate_objects_cordinates = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")
temp = 0
# цикл, для создания списка со всеми свойствами labels
for region in regionprops(label_image):
    if region.area < 50:
        # если регион слишком маленький, скорее всего это не знак
        continue
    # координаты ограничивающей рамки
    min_row, min_col, max_row, max_col = region.bbox

    region_height = max_row - min_row
    region_width = max_col - min_col

    # условный оператор, для убеждения, что указанный регион соответствует условиям типичного номера
    if min_height <= region_height <= max_height and min_width <= region_width <= max_width and region_width > region_height:
        temp = 1
        # рисование красных границ вокруг региона
        plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                         max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

if temp == 1:
    plt.show()

if temp == 0:
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")

    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        if region.area < 50:
            # if the region is so small then it's likely not a license plate
            continue
            # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
        # print(min_row)
        # print(min_col)
        # print(max_row)
        # print(max_col)

        region_height = max_row - min_row
        region_width = max_col - min_col
        # print(region_height)
        # print(region_width)

        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                      min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                             max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                           linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions
    # print(plate_like_objects[0])
    plt.show()
