import h5py
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
import matplotlib.pyplot as plt

def max_vector_length(part):
    # 提取前三个维度
    three_dimensions = part[:, :3]
    # 计算向量长度
    lengths = np.linalg.norm(three_dimensions, axis=1)
    return lengths.max()

Safety = np.empty((0, 7))
Efficiency = np.empty((0, 9))
Quality = np.empty((0, 2))
demo_names = []

def process_hdf5_file(hdf5_path):
    global Safety
    global Efficiency
    global Quality
    global demo_names
    with h5py.File(hdf5_path, 'r') as f:
        # 循环遍历文件中的所有组
        demo_names = list(f["data"].keys())
        num_of_demos = len(demo_names)
    
        # print(f["data"].attrs["trajectory_smoothness"])
        for ep in demo_names:
            #feature1 Safety
            contact_force = f[f"data/{ep}/contact_force"][()]
            # print("contact_force",contact_force)
            # 使用np.linalg.norm计算每个向量的长度，并指定axis=-1来按最后一个轴（即每个向量）计算
            magnitudes = np.linalg.norm(contact_force, axis=-1)
            # 找到最大的范数
            max_magnitude = np.max(magnitudes)
            contacts =  f[f"data/{ep}/contacts"][()]
            # print("contacts",contacts)
            distances = f[f"data/{ep}/distances"][()]
            # print("distances",distances)
            max_distance_to_table = np.max(distances[:, 0])
            min_distance_to_left_edge = np.min(distances[:, 1])
            min_distance_to_right_edge=np.min(distances[:, 2])
            min_distance_to_front_edge=np.min(distances[:, 3])
            min_distance_to_back_edge=np.min(distances[:, 4])
            # print("distances")
            num_collisions = f[f"data/{ep}/num_collisions"][()]
            feature1 = np.array([max_magnitude,max_distance_to_table,min_distance_to_left_edge,min_distance_to_right_edge,min_distance_to_front_edge,min_distance_to_back_edge,num_collisions])
            
            # print("feature1",feature1)
            Safety = np.vstack((Safety ,feature1))
            #feature2 Efficiency
            contacts =  f[f"data/{ep}/contacts"][()]
            path_lengths=f[f"data/{ep}/path_lengths"][()]
            # print("path_lengths",path_lengths)
            times=f[f"data/{ep}/times"][()]
            times_first_three= times[:3]
            feature2 = np.hstack((path_lengths,times_first_three))
            speeds=f[f"data/{ep}/speeds"][()]
            # 获取group
            grp = f['data']
            ep_data_grp = grp[ep]
            # 读取属性
            path23_start_id = ep_data_grp.attrs["path23_start_id"]
            
            # print("path23_start_id",path23_start_id)
            part1 = speeds[:path23_start_id]
            part2 = speeds[path23_start_id:]
            max_length_part1 = max_vector_length(part1)
            max_length_part2 = max_vector_length(part2)
            pseudo_cost = f[f"data/{ep}/pseudo_cost"][()]
            feature2 = np.append(feature2, [max_length_part1, max_length_part2, pseudo_cost])
            # print("feature2",feature2)
            Efficiency = np.vstack((Efficiency,feature2))
            
            # feature3  Quality
            speed_smoothness =  f[f"data/{ep}/speed_smoothness"][()]
            # print("speed_smoothness",speed_smoothness)
            average_of_first_three = np.mean(speed_smoothness[:3])
            trajectory_smoothness =  f[f"data/{ep}/trajectory_smoothness"][()]
            feature3 =np.array([average_of_first_three,trajectory_smoothness])
            # print("feature3",feature3)
            Quality = np.vstack((Quality,feature3))
    
    print(f"There are {num_of_demos} demo names in the list.")


hdf5_path = 'features0.hdf5'
process_hdf5_file(hdf5_path)


def compute_statistics(data):
    """ 计算给定数据的每一列的统计量 """
    means = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    
    return means, std_dev, mins, maxs

# 分别为Safety, Efficiency, Quality计算统计量
safety_stats = compute_statistics(Safety)
efficiency_stats = compute_statistics(Efficiency)
quality_stats = compute_statistics(Quality)

# 打印结果
print("Safety:")
print("Means:", safety_stats[0])
print("Standard Deviations:", safety_stats[1])
print("Min:", safety_stats[2])
print("Max:", safety_stats[3])

print("\nEfficiency:")
print("Means:", efficiency_stats[0])
print("Standard Deviations:", efficiency_stats[1])
print("Min:", efficiency_stats[2])
print("Max:", efficiency_stats[3])

print("\nQuality:")
print("Means:", quality_stats[0])
print("Standard Deviations:", quality_stats[1])
print("Min:", quality_stats[2])
print("Max:", quality_stats[3])




