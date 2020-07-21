
import numpy as np
from scipy.stats import chi2

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class ConfidenceEllipse:
    def __init__(self, data, p=0.0):
        self.data = data
        self.p = p

        self.means = np.mean(data, axis=0)
        self.cov = np.cov(data[:,0], data[:,1])

        lambdas, vecs = np.linalg.eigh(self.cov)
        order = lambdas.argsort()[::-1]
        lambdas, vecs = lambdas[order], vecs[:,order]

        c = np.sqrt(chi2.ppf(self.p, 2))
        self.w, self.h = 2 * c * np.sqrt(lambdas)
        self.theta = np.degrees(np.arctan(
            ((lambdas[0] - lambdas[1])/self.cov[0,1])))
    
    def calc_slope_intersept(self):

        center_x = self.means[0]
        center_y = self.means[1]

        next_x = self.means[0] + 1
        next_y = np.tan(np.deg2rad(self.theta))

        a = next_y
        b = center_y - a*center_x

        print("next_y=",next_y)
        print("a = ",a)

        x = np.arange(250)
        y = a * x + b

        return (x,y)

    def get_params(self):
        return self.means, self.w, self.h, self.theta

    def get_patch(self, line_color="black", face_color="none", alpha=0):
        el = Ellipse(xy=self.means,
                     width=self.w, height=self.h,
                     angle=self.theta, color=line_color, alpha=alpha)
        el.set_facecolor(face_color)
        return el

def gen_data():

    
    img = Image.open('depth_height.png')
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

    print(np_scatter_x.shape)
    points = np.array([np_scatter_x,np_scatter_y]).T
    print(points.shape)

    return points
    # return np.random.multivariate_normal([3,3], [[0.3,-0.2],[-0.2,1]], size=100)
def main():
    data = gen_data()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data[:,0], data[:,1], color="b", marker=".", s=3)


    el = ConfidenceEllipse(data, p=0.95)
    means, w, h, theta = el.get_params()
    x,y = el.calc_slope_intersept()

    ax.plot(x,y)

    ax.scatter(means[0], means[1], color="r",s=50)
    print(means)
    print(w)
    print(h)
    print(theta)

    ax.add_artist(el.get_patch(face_color="blue", alpha=0.5))
    ax.set_xlim([0,250])
    ax.set_ylim([0,250])
    plt.show()

if __name__ == "__main__":
    main()