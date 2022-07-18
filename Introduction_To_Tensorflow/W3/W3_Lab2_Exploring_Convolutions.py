from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

ascent_image = misc.ascent()
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(ascent_image)
plt.show()

image_transformed = np.copy(ascent_image)
print(image_transformed.shape)

size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]

## Convolution 결과 확인
filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]        # 수평 강조
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]        # 수직 강조

weight = 1
d
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        convolution = 0.0
        convolution = convolution + (ascent_image[x - 1, y - 1] * filter[0][0])
        convolution = convolution + (ascent_image[x - 1, y] * filter[0][1])
        convolution = convolution + (ascent_image[x - 1, y + 1] * filter[0][2])
        convolution = convolution + (ascent_image[x, y - 1] * filter[1][0])
        convolution = convolution + (ascent_image[x, y] * filter[1][1])
        convolution = convolution + (ascent_image[x, y + 1] * filter[1][2])
        convolution = convolution + (ascent_image[x + 1, y - 1] * filter[2][0])
        convolution = convolution + (ascent_image[x + 1, y] * filter[2][1])
        convolution = convolution + (ascent_image[x + 1, y + 1] * filter[2][2])

        # 왜 weight 를 곱해주는지 모르겠음
        convolution = convolution * weight

        if (convolution < 0):
            convolution = 0
        if (convolution > 255):
            convolution = 255

        image_transformed[x, y] = convolution

plt.gray()          # image를 gray scale로 바꿔줌
plt.grid(False)
plt.imshow(image_transformed)
plt.show()

## MaxPooling 결과 확인
new_x = int(size_x / 2)
new_y = int(size_y / 2)

new_image = np.zeros((new_x, new_y))

for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):

        pixels = []
        pixels.append(image_transformed[x, y])
        pixels.append(image_transformed[x+1, y])
        pixels.append(image_transformed[x, y+1])
        pixels.append(image_transformed[x+1, y+1])

        new_image[int(x/2), int(y/2)] = max(pixels)

plt.gray()
plt.grid(False)
plt.imshow(new_image)
plt.show()