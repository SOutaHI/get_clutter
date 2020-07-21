import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def main():
    # y = 2x
    features = np.array([[1, 2], [2, 4], [3, 6]])

    # グラフ描画サイズを設定する
    plt.figure(figsize=(12, 4))

    # 元データをプロットする
    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0], features[:, 1])
    plt.title('origin')
    plt.xlabel('x')
    plt.ylabel('y')

    # 主成分分析する
    pca = PCA()
    pca.fit(features)

    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features)

    # 主成分をプロットする
    plt.subplot(1, 2, 2)
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')

    # 主成分の次元ごとの寄与率を出力する
    print(pca.explained_variance_ratio_)

    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()