import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from PIL import Image
import sys

# generate data
img = Image.open(str(sys.argv[1]))

width, height = img.size

img_pixels = []
for y in range(height):
  for x in range(width):
    img_pixels.append(img.getpixel((x,y)))

img_pixels = np.array(img_pixels)
img_matrix = img_pixels.reshape([width,height])
scatter_x = []
scatter_y = []

for y in range(height):
  for x in range(width):

    if img_matrix[x,y] > 1000:
        scatter_x.append(x)
        scatter_y.append(y)

np_scatter_x = np.array(scatter_x)
np_scatter_y = np.array(scatter_y)

x = np_scatter_x.reshape(-1,1)
y = np_scatter_y.reshape(-1,1)

print(x.shape)
print(y.shape)

X = np.vstack([np_scatter_x, np_scatter_y]).T
 
ransac_regressor = linear_model.RANSACRegressor()
ransac_regressor.fit(x, y)
y_pred = ransac_regressor.predict(x)
 
plt.scatter(np_scatter_x, np_scatter_y, s=0.1)
plt.plot(x,y_pred, color='b', linewidth=2 )
plt.xlim([0,250])
plt.ylim([0,250])
plt.show()
