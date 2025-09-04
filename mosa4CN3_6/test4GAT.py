import torch, random, math, gym, numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions import Bernoulli, Categorical

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pandas as pd
import geopandas as gpd
import time
from pyproj import CRS
import networkx as nx
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing.managers

from vehicle_type import VehicleTypes,driving_range
from simulation import SimVehicle, SimVehicleTrip
from mosa import Problem
import data_utils
from station import Station
import sim_utils
from vehicle_scheduling import extra_timesave

# --------------------------------------------
# 0. 依赖环境
# --------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0); random.seed(0)


class Location(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self,
                         kwargs.get('num_vars'),  # var_num
                         [0] * kwargs.get('num_vars'),  # Integer or Real
                         [0] * len(kwargs.get('cs_gdf')) + [0,0,0],  # lb
                         [1] * len(kwargs.get('cs_gdf')) + [9999,9999,9999],  # ub
                         2,  # num of f
                         1,  # num of cv
                         **kwargs
                         )
        self.num_vars = kwargs.get('num_vars')
        self.sim_v_info = kwargs.get('sim_v_info')
        self.sim_v_dict = kwargs.get('sim_v_dict')
        self.cs_gdf = kwargs.get('cs_gdf')
        self.cs_num = len(self.cs_gdf)
        self.vs_parking_df = kwargs.get('vs_parking_df')
        self.all_d2s_dict = kwargs.get('all_d2s_dict')
        self.cs_dict = kwargs.get('cs_dict')

        self.v_name = self.vs_parking_df['v_name'].to_list()
        self.trip = self.vs_parking_df['trip'].to_list()
        self.s_time = self.vs_parking_df['s_time'].to_list()
        self.e_time = self.vs_parking_df['e_time'].to_list()
        self.destination = self.vs_parking_df['destination'].to_list()
        self.distance = self.vs_parking_df['distance'].to_list()
        self.avg_velocity = self.vs_parking_df['avg_velocity'].to_list()
        self.v_type = self.vs_parking_df['vehicle_type'].to_list()

        self.simv_v_name = self.sim_v_info.index.to_list()
        self.simv_trip = self.sim_v_info['trip'].to_list()
        self.simv_s_time = self.sim_v_info['s_time'].to_list()
        self.simv_e_time = self.sim_v_info['e_time'].to_list()
        self.simv_destination = self.sim_v_info['destination'].to_list()
        self.simv_distance = self.sim_v_info['distance'].to_list()
        self.simv_avg_velocity = self.sim_v_info['avg_velocity'].to_list()
        self.simv_v_type = self.sim_v_info['vehicle_type'].to_list()

        self.large_num = self.simv_v_type.count('large')
        self.medium_num = self.simv_v_type.count('medium')
        self.small_num = self.simv_v_type.count('small')

        self.e_price = kwargs.get('e_price')


    def eval_vars(self, vars_, is_test=False, *args):

        if "CV" in args:
            cv1 = sim_utils.set_cv1(self.cs_num, vars_)
            return np.hstack([cv1])

        cal_s_time = time.perf_counter()
        # Instantiate all sim_v and stations as well as simVTrip
        for i in range(len(self.simv_v_name)):
            if self.simv_v_type[i] == 'large':
                vehi_type = vars_[self.cs_num]
            elif self.simv_v_type[i] == 'medium':
                vehi_type = vars_[self.cs_num+1]
            else:
                vehi_type = vars_[self.cs_num+2]
            vehi = VehicleTypes(vehi_type, v_type=self.simv_v_type[i])
            self.sim_v_dict[self.simv_v_name[i]].model = vehi

        station_dict = {row.node_id: Station(row.node_id, idx, fast_charger=20,
                                        slow_charger=20) for idx, row in self.cs_gdf.iterrows()}

        timeout_list = []
        # Start simulation
        e_trip_sum, e_d2s_sum, wait_time_sum, emission_sum = 0, 0, 0, 0 # Initialize 4 variables for storing simulation values
        for i in range(len(self.v_name)):
            sim_v_trip = SimVehicleTrip(self.v_name[i], self.trip[i], self.s_time[i], self.e_time[i], self.destination[i],
                                        self.distance[i],self.sim_v_dict.get(self.v_name[i]).driving_range,
                                        self.sim_v_dict.get(self.v_name[i]).battery, self.avg_velocity[i])
            e_trip, e_d2s, timeout, wait_time, trip_dist = sim_v_trip.simulation(
                                                                      self.sim_v_dict.get(self.v_name[i]),
                                                                      station_dict, vars_,
                                                                      sim_cs_method='get',
                                                                      all_d2s_dict=self.all_d2s_dict, cs_dict=self.cs_dict)
            e_trip_sum += e_trip
            e_d2s_sum += e_d2s
            wait_time_sum += wait_time
            timeout_list.append(timeout)

            emission_sum += trip_dist * self.sim_v_dict.get(self.v_name[i]).model.per_emission

        fast_charger_counts = [station.max_used_fast_charger for station in station_dict.values()]
        slow_charger_counts = [station.max_used_slow_charger for station in station_dict.values()]

        # add extra vehicle:
        saved_time, subs_idx, extra_large, extra_medium, extra_small = extra_timesave(self.v_name,self.distance,timeout_list,self.v_type,
                                    VehicleTypes(vars_[self.cs_num], v_type='large').driving_range,
                                    VehicleTypes(vars_[self.cs_num+1], v_type='medium').driving_range,
                                    VehicleTypes(vars_[self.cs_num+2], v_type='small').driving_range)
        if is_test:
            return timeout_list, subs_idx

        # System costs
        # Vehicle cost
        vehicle_cost = ((self.large_num+extra_large)*VehicleTypes(vars_[self.cs_num],v_type='large').fix_cost +
                        (self.medium_num + extra_medium) * VehicleTypes(vars_[self.cs_num+1], v_type='medium').fix_cost +
                        (self.small_num + extra_small) * VehicleTypes(vars_[self.cs_num+2], v_type='small').fix_cost)

        # Station construction and maintenance costs per year
        station_cost = sum(vars_[0: self.cs_num]) * 600000 * 1
        station_emission = sum(vars_[0: self.cs_num]) * 80
        # 9w and 3w for fast and slow charger, respectively
        # Get chargers' num of selected stations and do calculation
        # sel_x = np.where(vars_[:self.cs_num] == 1)[0]  # Indexes of selected stations
        charger_cost = sum(fast_charger_counts) * 4000 + sum(slow_charger_counts) * 2000

        # 1.2 yuan/kwh
        # The life cycle cost of energy consumption in this trip， including both operation and go-charging distances
        trip_cost = (e_trip_sum+e_d2s_sum) * 365 * self.e_price * 1

        # print(f'Station Count: {sum(vars_[:self.cs_num])}; Extra Vehicles: {extra_large} {extra_medium} {extra_small}')

        '''one-objective'''
        # emission_cost = (emission_sum*365/1000+station_emission)*1.05  # kg * 1.05 yuan/kg social cost of emission
        # f0 = -(vehicle_cost + station_cost + charger_cost + trip_cost + emission_cost) / 1000000  # 1M yuan/year
        '''multi-objective'''
        f1 = (vehicle_cost + station_cost + charger_cost + trip_cost)/1000000  # 1M yuan/year
        f2 = (emission_sum*365/1000+station_emission)/1000  # T/year

        cv1 = sim_utils.set_cv1(self.cs_num, vars_)  # cv是约束,eq22(ub,lb是eq25-26)
        cv_and_params = ([cv1]+[-x for x in fast_charger_counts]+[-x for x in slow_charger_counts]+
                         [-extra_large, -extra_medium, -extra_small])
        cal_e_time = time.perf_counter()
        # print(f'Single calculation time: {(cal_e_time - cal_s_time):.2f}s')

        print(f"f1: {f1:.1f} f2: {f2:.1f}")

        return np.array([f1,f2]), np.hstack(cv_and_params)


# ---- 经验 min / max（仍使用 Monte-Carlo 估计） -------------------------------
def estimate_minmax(N=200):  # 2000
    J1,J2 = [],[]
    for _ in range(N):
        bf  = (torch.rand(NUM_NODES) < 0.5).int()
        if bf.sum()==0: bf[torch.randint(0,NUM_NODES,())]=1
        vt  = torch.tensor([random.randint(0,4),
                            random.randint(0,4),
                            random.randint(0,2)], dtype=torch.int32)
        new_var = np.concatenate((bf.numpy(), vt.numpy())) # shape: (nodes_num+3,)
        new_obj, new_cv = problem.eval_vars(new_var)[0:2]
        j1, j2 = new_obj[0], new_obj[1]
        # j1,j2 = calc_targets(bf,vt)
        J1.append(j1); J2.append(j2)
    return (min(J1),max(J1)),(min(J2),max(J2))

def norm_J(j1,j2):
    j1n=(j1-j1_min)/(j1_max-j1_min+1e-6)
    j2n=(j2-j2_min)/(j2_max-j2_min+1e-6)
    return j1n,j2n

# --------------------------------------------
# 2. Gym-style 环境
# --------------------------------------------
class ChargingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Dict({
            'build':  gym.spaces.MultiBinary(NUM_NODES),
            'types':  gym.spaces.MultiDiscrete([5,5,3])
        })
        # 静态图（不再包含坐标和成本）
        self.base_graph = Data(
            x=fixed_x,                   # [50, 1]
            edge_index=edge_index_fixed,   # [2, N^2]
            edge_attr=edge_attr_fixed     # [N^2, 2]
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        lam   = random.random()                  if options is None else options.get('lam',   0.5)
        g = self.base_graph.clone()
        g.lambda_val = torch.tensor([lam], dtype=torch.float32)
        self.graph   = g                         # 保存一下给 step 用
        return g, {}
    def step(self, action):
        build_flag = torch.as_tensor(action['build'], dtype=torch.int32)
        veh_types  = torch.as_tensor(action['types'], dtype=torch.int32)

        new_var = np.concatenate((build_flag.numpy(), veh_types.numpy()))  # shape: (nodes_num+3,)
        new_obj, new_cv = problem.eval_vars(new_var)[0:2]
        j1, j2 = new_obj[0], new_obj[1]
        # j1,j2    = calc_targets(build_flag, veh_types)
        j1n,j2n  = norm_J(j1,j2)
        lam      = self.graph.lambda_val.item()
        reward   = -(lam*j1n + (1-lam)*j2n)
        info = dict(J1=j1.item(),J2=j2.item(),lambda_=lam,
                    trunk=int(veh_types[0]),branch=int(veh_types[1]),micro=int(veh_types[2]))
        terminated=True
        return self.graph, reward, terminated, False, info


# --------------------------------------------
# 3. GAT Actor-Critic（多头输出）
# --------------------------------------------
class GAT_AC(torch.nn.Module):
    def __init__(self, in_dim=1, hid=64, heads=4, edge_dim=2):  # edge_dim=2 表示 [距离, 班次]
        super().__init__()
        self.conv1 = GATConv(in_dim, hid, heads=heads, edge_dim=edge_dim, dropout=0.1)
        self.conv2 = GATConv(hid*heads, hid, heads=1, edge_dim=edge_dim, dropout=0.1)

        # λ 投影
        self.lam_proj = torch.nn.Sequential(
            torch.nn.Linear(1, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, hid)
        )

        # Policy Heads
        self.policy_build  = torch.nn.Linear(hid * 2, NUM_NODES)
        self.policy_trunk  = torch.nn.Linear(hid * 2, 5)
        self.policy_branch = torch.nn.Linear(hid * 2, 5)
        self.policy_micro  = torch.nn.Linear(hid * 2, 3)

        self.value         = torch.nn.Linear(hid * 2, 1)

    def forward(self, data: Data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        lam = data.lambda_val

        x = F.elu(self.conv1(x, ei, ea))
        x = F.elu(self.conv2(x, ei, ea))
        x = global_mean_pool(x, batch)  # [B,hid]

        lam_emb = self.lam_proj(lam.unsqueeze(-1))  # [B,hid]
        g_feat = torch.cat([x, lam_emb], 1)  # [B, hid*2]

        logits_build = self.policy_build(g_feat)  # [B,50]
        logits_trunk = self.policy_trunk(g_feat)  # [B,5]
        logits_branch = self.policy_branch(g_feat)  # [B,5]
        logits_micro = self.policy_micro(g_feat)  # [B,3]
        value = self.value(g_feat).squeeze(1)  # [B]

        return (logits_build, logits_trunk,
                logits_branch, logits_micro), value


def sample_solutions(N=100, lambdas=[i/20 for i in range(21)]):
    """调用训练好的策略采样解；返回 (j1,j2,lam,trunk_type)"""
    pareto = []
    base_g,_ = env.reset()
    for lam in lambdas:
        base_g.lambda_val = torch.tensor([lam])
        for _ in range(N):
            (log_b,log_tr,log_br,log_mi),_ = policy_net(
                Batch.from_data_list([base_g]).to(device))
            bern  = Bernoulli(logits=log_b) ; cat_t = Categorical(logits=log_tr)
            cat_b = Categorical(logits=log_br); cat_m = Categorical(logits=log_mi)

            act_build  = bern.sample().squeeze(0).cpu()
            trunk_type = cat_t.sample().item()
            branch_type= cat_b.sample().item()
            micro_type = cat_m.sample().item()

            vtypes = torch.tensor([trunk_type,branch_type,micro_type])
            new_var = np.concatenate((act_build.numpy(), vtypes.numpy())).astype(int)  # shape: (nodes_num+3,)
            new_obj, new_cv = problem.eval_vars(new_var)[0:2]
            j1, j2 = new_obj[0], new_obj[1]
            # j1,j2 = calc_targets(act_build,
            #                      torch.tensor([trunk_type,branch_type,micro_type]))
            pareto.append((j1.item(), j2.item(), lam, trunk_type))
    return pareto


def get_pareto_front(points):
    """
    返回真正帕累托前沿。J1、J2 都是越小越好。
    保留元组原形，后面两个字段不参与比较。
    """
    n = len(points)
    is_front = [True]*n
    for i in range(n):
        if not is_front[i]:
            continue
        j1_i, j2_i = points[i][:2]
        for j in range(n):
            if i == j or not is_front[j]:
                continue
            j1_j, j2_j = points[j][:2]
            # j 点支配 i 点（两项都 ≤，且至少一项 <）
            if (j1_j <= j1_i and j2_j <= j2_i) and (j1_j < j1_i or j2_j < j2_i):
                is_front[i] = False
                break          # i 已被支配，无需再比较其它 j
    # 提取帕累托解
    front = [pt for pt, keep in zip(points, is_front) if keep]
    front.sort(key=lambda x: x[0], reverse=True)
    return front


# -------- 随机样本（含车型随机） -------------------------
def random_build_strategy(num_nodes):
    bf = (torch.rand(num_nodes) < 0.3).float()
    if bf.sum()==0: bf[torch.randint(0,num_nodes,())] = 1
    return bf
def random_vehicle():
    return torch.tensor([random.randint(0,4),
                         random.randint(0,4),
                         random.randint(0,2)])


def init_worker(problem_inst):
    global SHARED_PROBLEM_INSTANCE
    SHARED_PROBLEM_INSTANCE = problem_inst
    torch.set_num_threads(1)


def evaluate_new_var_shared(args):
    global SHARED_PROBLEM_INSTANCE
    obj, _ = SHARED_PROBLEM_INSTANCE.eval_vars(args)
    return obj[0], obj[1]


if __name__ == '__main__':
    from multiprocessing import set_start_method
    # --------------------------------------------
    # 1. 静态图数据（仅使用坐标生成边信息，不作为输入）
    # --------------------------------------------
    is_simplified = False
    to_simplify = False
    is_referred = False
    is_dict_loaded = True
    is_tested = False
    SIM_N = 50
    cities = pd.read_csv(rf'../data/18cities.csv')
    city_i = 16
    city_name = cities['city'][city_i]

    if is_simplified:
        vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}_simplified.csv')  # 车辆行程
    else:
        vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')  # 车辆行程
    vs_parking_df = data_utils.preprocess_vs_df(vs_parking_df)

    cs_gdf = gpd.read_file(rf'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4547))  # 充电站
    NUM_NODES = len(cs_gdf)

    # 1.2 边索引：全连接图
    row, col = torch.meshgrid(torch.arange(NUM_NODES), torch.arange(NUM_NODES), indexing='ij')
    edge_index_fixed = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)

    # 1.3 边特征：[距离, 公交班次]
    edge_attr_fixed = data_utils.get_edge_attrs(vs_parking_df, cs_gdf)

    # 1.4 节点特征：统一设为 [1]
    fixed_x = torch.ones(NUM_NODES, 1)  # [50, 1]

    # --- 下同test4mosa的准备部分 -----------------------------------------------------------------
    if not is_simplified and to_simplify:
        vs_parking_df0 = vs_parking_df.copy(deep=False)
        vs_parking_df = data_utils.simplify_vs_df(vs_parking_df, min(driving_range[0]), min(driving_range[1]))

    sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').  # 将车辆行程按车辆集计
                                        apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')
    sim_v_dict = {}
    for idx, row in sim_v_info.iterrows():
        sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                     row.avg_velocity)

    if is_dict_loaded:
        with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'rb') as f:
            all_d2s_dict = pickle.load(f)
    else:
        nodes_sim = gpd.read_file(
            rf'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4547))  # 道路节点
        edges_sim = gpd.read_file(
            rf'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4547))  # 道路路段
        G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v', edge_attr=['edge_id', 'length'],
                                    create_using=nx.Graph())
        all_d2s_dict = data_utils.get_d2s_realdict(vs_parking_df, cs_gdf, nodes_sim, G,
                                                   near_n=100, sim_n=SIM_N, distance_limit=10000.0, is_projected=True)
        with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'wb') as f:
            pickle.dump(all_d2s_dict, f)

    num_vars = data_utils.set_var_num(cs_gdf)
    cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}
    print('read files successfully!')

    problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, sim_v_dict=sim_v_dict, cs_gdf=cs_gdf,
                       vs_parking_df=vs_parking_df,
                       all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, e_price=cities['eprice'][city_i])
    pool = ProcessPoolExecutor(max_workers=8,
                               initializer=init_worker,
                               initargs=(problem,))

    # --------------------------------------------
    # 1.5. Initialize
    # --------------------------------------------
    (j1_min, j1_max), (j2_min, j2_max) = estimate_minmax()
    env = ChargingEnv()
    policy_net = GAT_AC().to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=2e-4)
    print("Initial test was done successfully.")

    # --------------------------------------------
    # 4. REINFORCE 训练循环
    # --------------------------------------------
    # 开始训练循环
    BATCH = 10
    EPISODES = 2000  # 根据需要调整

    for step in range(1, EPISODES + 1):
        # ---- 采样 batch 环境 -----------------
        graphs, lams = [], []
        for _ in range(BATCH):
            g,_ = env.reset()
            graphs.append(g)
            lams.append(g.lambda_val.item())
        batch_graph = Batch.from_data_list(graphs).to(device)

        (log_b, log_tr, log_br, log_mi), v_pred = policy_net(batch_graph)

        # --- 构造分布并采样动作 --------------
        bern = Bernoulli(logits=log_b)
        cat_t = Categorical(logits=log_tr)
        cat_b = Categorical(logits=log_br)
        cat_m = Categorical(logits=log_mi)

        act_build = bern.sample()  # [B,50] 0/1
        act_trunk = cat_t.sample()  # [B]
        act_branch = cat_b.sample()  # [B]
        act_micro = cat_m.sample()  # [B]

        log_prob = bern.log_prob(act_build).sum(1) \
                 + cat_t.log_prob(act_trunk) \
                 + cat_b.log_prob(act_branch) \
                 + cat_m.log_prob(act_micro)

        # ---- 并行化计算目标 -------------------
        J1, J2 = [], []

        act_build_cpu = act_build.cpu()

        cal_s_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=10, initializer=init_worker) as executor:
            futures = []
            for i in range(BATCH):
                vtypes = torch.tensor([act_trunk[i], act_branch[i], act_micro[i]])
                new_var = np.concatenate((act_build_cpu[i].numpy(), vtypes.numpy())).astype(int)
                futures.append(pool.submit(evaluate_new_var_shared, new_var))

            J1, J2 = [], []
            for fut in as_completed(futures):
                j1, j2 = fut.result()
                J1.append(j1)
                J2.append(j2)
        cal_e_time = time.perf_counter()
        print(f'Single calculation time: {(cal_e_time - cal_s_time):.2f}s')

        # ---- reward 计算 ----------------------
        J1 = torch.tensor(J1, device=device, dtype=torch.float32)
        J2 = torch.tensor(J2, device=device, dtype=torch.float32)
        J1n, J2n = norm_J(J1, J2)
        lam_t = torch.tensor(lams, dtype=torch.float32, device=device)
        reward = -(lam_t * J1n + (1 - lam_t) * J2n)

        # ---- 损失 ---------------------------
        advantage = reward.to(device) - v_pred.detach()
        pg_loss = -(log_prob.to(device) * advantage).mean()
        value_loss = 0.5 * (v_pred - reward.to(device)).pow(2).mean()
        entropy = (bern.entropy().sum(1) + cat_t.entropy() + cat_b.entropy() + cat_m.entropy()).mean()
        loss = pg_loss + value_loss - 0.01 * entropy

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if step % 100 == 0:
            print(f'Step {step:5d}  Loss {loss.item():.4f}   R̄ {-reward.mean():.3f}')


    # --------------------------------------------
    # 5. 采样 Pareto 并绘图（完整替换）
    # --------------------------------------------
    pareto = sample_solutions()
    pareto = get_pareto_front(pareto)

    N_RANDOM = 100
    random_pts = []
    for _ in range(N_RANDOM):
        bf = random_build_strategy(NUM_NODES)
        vt = random_vehicle()
        new_var = np.concatenate((bf.numpy(), vt.numpy())).astype(int)  # shape: (nodes_num+3,)
        new_obj, new_cv = problem.eval_vars(new_var)[0:2]
        j1, j2 = new_obj[0], new_obj[1]
        # j1,j2 = calc_targets(bf,vt)
        random_pts.append((j1.item(),j2.item()))
    random_pts = np.array(random_pts)

    # --------- Pareto 前沿可视化 -----------------------------
    rl_pts   = np.array([(j1,j2,tk) for j1,j2,lam,tk in pareto])
    colors   = rl_pts[:,2]                       # trunk 车型 0-4
    norm     = Normalize(vmin=0, vmax=4)

    plt.figure(figsize=(8,6))
    plt.scatter(random_pts[:,0], random_pts[:,1],
                c='gray', alpha=0.4, label='Random Samples')

    sc = plt.scatter(rl_pts[:,0], rl_pts[:,1],
                     c=colors, cmap='viridis', norm=norm,
                     edgecolor='k', s=50, label='RL Pareto')
    plt.plot(rl_pts[:,0], rl_pts[:,1],'g--',lw=1)

    cbar = plt.colorbar(sc,ticks=[0,1,2,3,4])
    cbar.set_label('Trunk Vehicle Type')

    plt.xlabel('System Cost')
    plt.ylabel('GHG Emission')
    plt.title('Pareto Front (color = trunk vehicle type)')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=True)