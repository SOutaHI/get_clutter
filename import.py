from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from scipy.spatial import ConvexHull



# img = Image.open('depth_height.png')
img = Image.open('depth_height.png')


width, height = img.size
print(width, height)


# fig_scatter = plt.figure()
# axs = fig_scatter.add_subplot(2,1,1)

img_pixels = []
for y in range(height):
  for x in range(width):
    img_pixels.append(img.getpixel((x,y)))

img_pixels = np.array(img_pixels)
img_matrix = img_pixels.reshape([width,height])
# print(np.unique(img_pixels))
# print(img_matrix)

scatter_x = []
scatter_y = []

for y in range(height):
  for x in range(width):

    if img_matrix[x,y] > 1000:
        scatter_x.append(x)
        scatter_y.append(y)

np_scatter_x = np.array(scatter_x)
np_scatter_y = np.array(scatter_y)

print(np_scatter_x.shape)
points = np.array([np_scatter_x,np_scatter_y]).T
print(points.shape)

r = patches.Rectangle(xy=(0, 0), width=244, height=244, ec='#000000', fill=False)
# axs.add_patch(r)
# axs.scatter(scatter_x, scatter_y, s=0.05,c='blue')
# axs.set_title('depth scatter')

# axs.set_xlim([0,250])
# axs.set_ylim([0,250])


# fig = plt.figure()
# ax = fig_scatter.add_subplot(2,1,2)
# ax.set_title('height histogram')
# ax.set_ylabel('freq')
# sns.distplot(img_pixels, kde=False, rug=False, bins=25, axlabel="height")

# plt.show()


# points = np.array([points])

hull = ConvexHull(points)
points = hull.points
hull_points = points[hull.vertices]
 
hp = np.vstack((hull_points, hull_points[0]))
plt.plot(hp[:,0], hp[:,1])
plt.scatter(points[:,0], points[:,1])
plt.show()