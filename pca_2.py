import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import matplotlib.patches as patches

# generate data
img = Image.open(str(sys.argv[1]))

width, height = img.size
print(width, height)


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

x = np.linspace(0.2,1,100)
y = 0.8*x + np.random.randn(100)*0.1
# X = np.vstack([x, y]).T
X = np.vstack([np_scatter_x, np_scatter_y]).T
# np.random.shuffle(X)

# plot data
# fig = plt.figure()
# axes = fig.add_subplot(111,aspect='equal')
# axes.scatter(X[:,0],X[:,1])
# axes.set_xlim([0.0, 250])
# axes.set_ylim([0.0, 250])
# axes.set_xlabel('Y of Pixel')
# axes.set_ylabel('X of Pixel')
# axes.vlines(0,0.0,250,linestyles='dashed')
# axes.hlines(0,0.0,250,linestyles='dashed')

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


mn = np.mean(X,axis=0)
z = X - mn
cv = np.cov(z[:,0],z[:,1],bias=1)

W, v = np.linalg.eig(cv)
# display 
fig = plt.figure()
axes = fig.add_subplot(111,aspect='equal')
axes.scatter(X[:,0],X[:,1], s=0.01)
axes.set_xlim([0.0, 250])
axes.set_ylim([0.0, 250])
axes.set_xlabel('X of Pixels')
axes.set_ylabel('Y of Pixels')
axes.vlines(0,0.0,250,linestyles='dashed')
axes.hlines(0,0.0,250,linestyles='dashed')
axes.quiver(pca.mean_[0], pca.mean_[1], pca.components_[0,0],pca.components_[0,1], color='red', width=0.01, scale=3)
axes.quiver(pca.mean_[0], pca.mean_[1], pca.components_[1,0],pca.components_[1,1], color='blue', width=0.01, scale=3)

r = patches.Rectangle(xy=(0, 0), width=224, height=224, ec='#000000', fill=False)
axes.add_patch(r)
    


# projection
# Xd = pca.transform(X)

# fig = plt.figure()
# axes = fig.add_subplot(111,aspect='equal')
# axes.scatter(Xd[:,0],Xd[:,1])
# axes.set_xlabel('xd0')
# axes.set_ylabel('xd1')
# axes.set_xlim([-1.0, 1.0])
# axes.set_ylim([-1.,1.0])
# axes.vlines(0,-1.0,1.0,linestyles='dashed')
# axes.hlines(0,-1.0,1.0,linestyles='dashed')

plt.show()