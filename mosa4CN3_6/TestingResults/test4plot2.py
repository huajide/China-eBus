import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

ax = None
bx = None

def model_func(X, a, b, c, d, e, f, g, h, i, j, k):
    x, y = X
    return a*x**3+b*y**3+c*x*2+d*y*2+e*x**2*y**2+f*x**2*y+g*x*y**2+h*x*y+i*x+j*y+k


def plot_pareto_front(X,Y,Z,surface=None,scatter=True,params=['b','cool','Default']):
    global ax
    pareto_front = np.array([X,Y,Z]).T

    if surface is not None:
        split = 100
        x_min, x_max = np.min(pareto_front[:, 0]), np.max(pareto_front[:, 0])
        y_min, y_max = np.min(pareto_front[:, 1]), np.max(pareto_front[:, 1])
        delta_x, delta_y = (x_max-x_min)/(split-1), (y_max-y_min)/(split-1)
        plus_delta = 0
        grid_x = np.linspace(x_min-delta_x*plus_delta, x_max+delta_x*plus_delta, split+2*plus_delta)  # 在 x 的范围内生成 1000 个点
        grid_y = np.linspace(y_min-delta_y*plus_delta, y_max+delta_y*plus_delta, split+2*plus_delta)  # 在 y 的范围内生成 1000 个点
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        if surface == 'interpolation':
            grid_z = griddata((pareto_front[:, 0], pareto_front[:, 1]), pareto_front[:, 2], (grid_x, grid_y),
                              method='linear')
        elif surface == 'polyfit':
            popt, pcov = curve_fit(model_func, (pareto_front[:, 0], pareto_front[:, 1]), pareto_front[:, 2])
            grid_z = model_func((grid_x, grid_y), *popt)
        elif surface == 'ml':
            grid_search =False
            x = pareto_front[:, 0]
            y = pareto_front[:, 1]
            z = pareto_front[:, 2]
            X = np.column_stack((x, y))
            if grid_search:
                model = RandomForestRegressor()
                param_grid = {
                    'n_estimators': [50, 100, 200],  # 树的数量
                    'max_features': ['auto', 'sqrt'],  # 每棵树考虑的特征数量
                    'max_depth': [None, 10, 20, 30],  # 树的最大深度
                    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
                    'min_samples_leaf': [1, 2, 4]  # 叶子节点最小样本数
                }
                model = GridSearchCV(estimator=model, param_grid=param_grid,
                                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=54)
            model.fit(X, z)
            grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            grid_z = model.predict(grid_points)
            grid_z = grid_z.reshape(grid_x.shape)
        # ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.7, label=params[2]+' Pareto Front',
        #                 cmap=params[1], rcount=split, ccount=split, antialiased=True)

    if scatter:
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], s=4,
                   color=params[0], label=params[2]+' Pareto Solutions', edgecolor='none')


def clearing(df):
    df.columns = ['solution_no', 'obj1', 'obj2']
    df_numeric = df.select_dtypes(include=['number'])  # 选择数值类型的列
    df_numeric = df_numeric.abs()  # 将数值列转换为正数
    df['obj1'] = df_numeric['obj1']
    df['obj2'] = df_numeric['obj2']

    # # 去重
    rows_to_remove = []
    for index, row in df.iterrows():
        # 检查是否存在其他行的所有列值都大于当前行
        if any((df[['obj1','obj2']] < row.iloc[-2:]).all(axis=1)):
            rows_to_remove.append(index)
    df_numeric_nodpl = df.drop(rows_to_remove)
    df_numeric_nodpl['obj1'] = df_numeric_nodpl['obj1']
    df_numeric_nodpl['obj2'] = df_numeric_nodpl['obj2']
    return df_numeric_nodpl

def plot_double(dataG,save_pic=False):
    """
    dataB/dataG: OBJ.CSV format
    """
    global ax
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 13
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    data = dataG

    x = data.iloc[:, -2].to_list()
    y = data.iloc[:, -1].to_list()
    print(f'This scenario has {len(x)} pareto solutions')

    plot_pareto_front(x,y,surface='ml',scatter=True,params=['b','Blues_r',''])

    ax.view_init(elev=15, azim=-15)
    ax.set_xlabel('Obj1: System cost (1M HK$/year)')
    ax.set_ylabel('Obj2: GHGs emissions (T/year)')
    plt.legend(loc='upper right', fontsize=11, bbox_to_anchor=(0.97, 0.7))  # 设置图例位置和字体大小
    # plt.tight_layout()
    if save_pic:
        plt.savefig(f'pareto.png', dpi=300)
    plt.show(block=True)


def single_cross_tab(Data,title_name:str,color:str):
    global bx
    data = Data.copy(deep=True)
    bus_mapping = {
        0: 'BYD B12D',
        1: 'Enviro500',
        2: 'CRRC eD12'
    }
    minibus_mapping = {
        0: 'BYD B12D',
        1: 'Enviro500',
        2: 'CRRC eD12'
    }
    all_bus_types = ['BYD B12D', 'Enviro500','CRRC eD12']
    all_minibus_types = ['BYD B12D', 'Enviro500','CRRC eD12']

    data.iloc[:,-2] = data.iloc[:,-2].map(bus_mapping)
    data.iloc[:,-2] = data.iloc[:,-2].map(minibus_mapping)

    cross_tab = pd.crosstab(data.iloc[:,-2], data.iloc[:,-2])
    cross_tab = cross_tab.reindex(index=all_bus_types, columns=all_minibus_types, fill_value=0)

    sns.heatmap(cross_tab, annot=True, fmt='d', cmap=color, cbar=True, ax=bx)
    bx.set_title(f'{title_name}')
    bx.set_xlabel('Double-deck Bus')
    bx.set_ylabel('Double-deck Bus')


def plot_cross_tab(dataB,dataG,save_pic=False):
    global bx
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(9,4))

    bx = fig.add_subplot(121)
    single_cross_tab(dataB,'Baseline','Greens')
    bx = fig.add_subplot(122)
    single_cross_tab(dataG, 'Gradient','Blues')
    fig.suptitle('Cross Analysis of Vehicle Type', fontsize=12)

    if save_pic:
        plt.savefig(f'VehicleType.png', dpi=300)
    plt.show(block=True)
if __name__ == '__main__':
    """帕累托作图"""
    result_path = '250630/莆田市_250630'
    dataG = pd.read_csv(rf"../../data/output/{result_path}/archive_objs.csv")
    # dataB = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Baseline\archive_objs.csv")
    dataG = clearing(dataG)
    # dataB = clearing(dataB)
    plot_double(dataG,save_pic=False)

    """车型交叉分析"""
    # result_path = 'TestingResults0115'
    # cs_num = 467
    # dataG = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Grade\archive_vars.csv")
    # dataB = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Baseline\archive_vars.csv")
    # plot_cross_tab(dataB,dataG,save_pic=False)






