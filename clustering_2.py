import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import sys

divide = 2 #状態量の数
R = np.zeros((divide,divide))    #相関行列
invR = np.zeros((divide,divide)) #相関行列の逆行列
avg = np.zeros(divide)           #平均値
std = np.zeros(divide)           #標準偏差
make = 0

p = 0.95        #マハラノビス距離p=0.95で2σ
md_sikii = 2.448#MDの閾値95%で2.448
div = 100        #Mt楕円の分割数

def maha(x):
    global make, R, invR, avg, std

    N, _ = x.shape #Nはデータ数
    xx = np.copy(x)
    xx = np.array(xx,dtype="float32")
    x_return = []

    #各状態量から平均値を引く
    for i in range(divide): 
        if make == 0:
            avg[i] = np.mean(x[:,i])
        for j in range(N):
            xx[j,i] = xx[j,i] - avg[i]

    #各状態量を標準偏差で割る
    for i in range(divide):
        if make == 0:
            std[i] = np.std(x[:,i])
        for j in range(N):
            xx[j,i] = xx[j,i] / std[i]

    #make=0のときだけ計算
    if make == 0:
        R = np.corrcoef(xx.transpose())
        invR = np.linalg.inv(R)
        make = 1

    #MD^2の計算
    for i in range(N):
        d0 = xx[i,:]
        d1 = np.dot(d0,invR)
        d2 = np.dot(d1,d0)/divide
        x_return.append(d2)

    return x_return

if __name__ == '__main__':

    curve_c = np.zeros((2,div+1))

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
    points = np.array([np_scatter_x,np_scatter_y]).T

    data = points
    # print(data.shape)

    #単位空間の作成
    _ = maha(data)

    #テストデータのマハラノビス距離
    # md_1 = maha(test1)
    # md_2 = maha(test2)

    #楕円のデータ
    low = np.corrcoef(data[:,0],data[:,1])[0,1]

    cov = np.cov(data[:,0], data[:,1])

    lambdas, vecs = np.linalg.eigh(cov)
    order = lambdas.argsort()[::-1]
    lambdas, vecs = lambdas[order], vecs[:,order]
    theta = np.degrees(np.arctan(((lambdas[0] - lambdas[1])/cov[0,1])))

    for i in range(div+1):
        r = (-2*(1-low**2)*np.log(1-p)/(1-2*low*np.sin(i*2*np.pi/div)*np.cos(i*2*np.pi/div)))**0.5
        
        curve_c[0,i] = avg[0] + std[0]*r*np.cos(i*2*np.pi/div)
        curve_c[1,i] = avg[1] + std[1]*r*np.sin(i*2*np.pi/div)
        # print(i)

    distance = np.zeros(div+1)

    for i in range(50):
        distance[i] = np.sqrt(np.square(curve_c[0,i] - curve_c[0,50+i]) + np.square(curve_c[1,i] - curve_c[1,50+i]))

    # print(distance)
    long_axis_index = distance.argmax()
    print("max=",distance.argmax())

    center_x = avg[0]
    center_y = avg[1]

    next_x = curve_c[0,long_axis_index]
    next_y = curve_c[1,long_axis_index]

    a = (center_y - next_y)/(center_x - next_x)
    b = center_y - a*center_x

    ver_a = -1.0/a
    ver_b = center_y - ver_a*center_x

    x = np.arange(250)
    y     = a * x + b
    ver_y = ver_a*x + ver_b

    print(ver_a*a)

    print("next_y=",next_y)
    print("a = ",ver_a)

    print(ver_y)
    print("theta=", theta)
    print(low)


    #可視化
    plt.figure()

    # plt.subplot(1,2,1)
    plt.scatter(scatter_x, scatter_y, c="green", s=0.01)
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")

    plt.plot(curve_c[0],curve_c[1],c="c")
    plt.legend()

    plt.scatter(avg[0],avg[1], c='red', s=10)
    plt.plot(x,y)
    # plt.plot(x,ver_y)
    
    r = patches.Rectangle(xy=(0, 0), width=224, height=224, ec='#000000', fill=False)
    ax = plt.axes()
    ax.add_patch(r)
    

    # plt.scatter(curve_c[0,50], curve_c[1,50],c='r', s=60)
    # plt.scatter(curve_c[0,55], curve_c[1,55],c='r', s=60)
    # plt.scatter(curve_c[0,100], curve_c[1,100],c='r', s=20)
    # plt.scatter(curve_c[0,25], curve_c[1,25],c='r', s=80)
    # plt.scatter(curve_c[0,5], curve_c[1,5],c='r', s=80)
    # plt.scatter(curve_c[0,0], curve_c[1,0],c='r', s=100)
    # plt.scatter(curve_c[0,75], curve_c[1,75],c='r', s=40)
     
    
    hull = ConvexHull(points)
    points = hull.points
    hull_points = points[hull.vertices]
    
    hp = np.vstack((hull_points, hull_points[0]))
    plt.plot(hp[:,0], hp[:,1])

    plt.xlim([0,300])
    plt.ylim([0,300])
    plt.savefig("out.png")
    plt.show()