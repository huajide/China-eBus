import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor




def model_func(X, a, b, c, d, e, f, g, h, i, j, k):
    x, y = X
    return a*x**3+b*y**3+c*x*2+d*y*2+e*x**2*y**2+f*x**2*y+g*x*y**2+h*x*y+i*x+j*y+k


def plot_pareto_front(X,Y,Z,surface=None,scatter=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pareto_front = np.array([X,Y,Z]).T
    if scatter:
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                   color='b', label='Pareto Front Points')

    if surface is not None:
        split = 100
        x_min, x_max = np.min(pareto_front[:, 0]), np.max(pareto_front[:, 0])
        y_min, y_max = np.min(pareto_front[:, 1]), np.max(pareto_front[:, 1])
        delta_x, delta_y = (x_max-x_min)/(split-1), (y_max-y_min)/(split-1)
        grid_x = np.linspace(x_min-delta_x, x_max+delta_x, split+2)  # 在 x 的范围内生成 1000 个点
        grid_y = np.linspace(y_min-delta_y, y_max+delta_y, split+2)  # 在 y 的范围内生成 1000 个点
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        if surface == 'interpolation':
            grid_z = griddata((pareto_front[:, 0], pareto_front[:, 1]), pareto_front[:, 2], (grid_x, grid_y),
                              method='linear')
        elif surface == 'polyfit':
            popt, pcov = curve_fit(model_func, (pareto_front[:, 0], pareto_front[:, 1]), pareto_front[:, 2])
            grid_z = model_func((grid_x, grid_y), *popt)
        elif surface == 'cnn':
            x = pareto_front[:, 0]
            y = pareto_front[:, 1]
            z = pareto_front[:, 2]
            X = np.column_stack((x, y))
            model = RandomForestRegressor(n_estimators=100,  # 树的数量
                                          random_state=42)
            model.fit(X, z)
            grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            grid_z = model.predict(grid_points)
            grid_z = grid_z.reshape(grid_x.shape)
        ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.5, label='Fitted Surface',
                        cmap='cool', rstride=1, cstride=1, antialiased=True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show(block=True)

if __name__ == '__main__':
    data = pd.read_csv(r"E:\Manufacture\Python\hkbus\data\output\MOSA\TestingResults0103\TestingResults0103-Grade\archive_objs.csv")
    # data = pd.read_csv(r"E:\Manufacture\Python\hkbus\data\output\MOSA\TestingResults0103\TestingResults0103-Baseline\archive_objs.csv")

    df_numeric = data.select_dtypes(include=['number'])  # 选择数值类型的列
    df_numeric = df_numeric.abs()  # 将数值列转换为正数

    # # 去重
    rows_to_remove = []
    for index, row in df_numeric.iterrows():
        # 检查是否存在其他行的所有列值都大于当前行
        if any((df_numeric[['0', '1', '2']] < row).all(axis=1)):
            rows_to_remove.append(index)
    df_numeric_nodpl = df_numeric.drop(rows_to_remove)

    x = df_numeric_nodpl['0'].to_list()
    y = df_numeric_nodpl['1'].to_list()
    z = df_numeric_nodpl['2'].to_list()

    plot_pareto_front(x,y,z,surface='cnn',scatter=True)



