import numpy as np
from matplotlib import pyplot as plt
import itertools

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

X = np.vstack([np_scatter_x, np_scatter_y]).T

# true values
_a = 0.5
_b = 0.3
# samples
# points = np.array([[x, _a * x + _b + .1 * np.random.randn() + (np.random.randint(100) == 0) * np.random.rand() * 1000] for x in np.arange(0, 10, 0.01)])
points = X 

plt.scatter(points[:, 0], points[:, 1], s=0.01)
# plt.y
plt.show()

data = points

def getParamWithSamples(samples):
    p0 = samples[0]
    p1 = samples[1]
    dp = p1 - p0
    if dp[0] == 0:
        dp[0] = 50
    a = dp[1] / dp[0]
    b = p0[1] - a * p0[0]
    return a, b
    
def applyModel(model, x):
    a = model[0]
    b = model[1]
    return a * x + b
   

def getError(model, p):
    x = p[0]
    y = p[1]
    return np.abs(applyModel(model, x) - y)

def errorGrad(model, data):
    a = model[0]
    b = model[1]
    x = data[:, 0]
    y = data[:, 1]
    ga = (2 * (a * x + b - y) * x).sum()
    gb = (2 * (a * x + b - y)).sum()
    return np.array([ga, gb])


import numpy as np


def ransac(data,
        # parameters for RANSAC
        n = 2, # required sample num to decide parameter
        k = 100, # max loop num
        t = 2.0, # threshold error val for inlier
        d = 30000 # requrired inlier sample num to be correnct param
    ):

    good_models = []
    good_model_errors = []
    iterations = 0
    while iterations < k:
        sample = data[np.random.choice(len(data), 2, False)]
        param = getParamWithSamples(sample)

        inliers = []
        for p in data:
            if (p == sample).all(1).any(): continue
            if getError(param, p) > t:
                continue
            else:
                inliers.append(p)


        if len(inliers) > d:
            current_error = np.array([getError(param, p) for p in data]).mean()
            good_models.append(param)
            good_model_errors.append(current_error)

        iterations += 1
    print(good_model_errors)
    best_index = np.argmin(good_model_errors)
    return good_models[best_index]

def leastSquare(data):
        # Simulated Annealing
    tau = 100
    bestfit = None
    besterr = float('inf')
    model = np.zeros(2)
    while tau >= 0.0001:
        for _ in range(10):
            grad = errorGrad(model, data)
            grad /= np.linalg.norm(grad)
            grad *= -1
            model += grad * tau
            
        tau *= 0.1
    return model


a, b = ransac(data)
print (a, b)


plt.plot(data[:, 0], data[:, 1])
l = np.array([(x, applyModel((a, b), x)) for x in data[:, 0]])
plt.plot(l[:, 0], l[:, 1])
l = np.array([(x, applyModel((_a, _b), x)) for x in data[:, 0]])
plt.plot(l[:, 0], l[:, 1])
plt.ylim(0, 10)

plt.show()














