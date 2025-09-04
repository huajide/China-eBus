import numpy as np

# 定义矩阵A
A = np.array([[1, 0.5],
              [0.3, 1]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 输出特征值
print("特征值:", eigenvalues)

# 输出特征向量
print("特征向量:")
for i in range(len(eigenvalues)):
    print(f"特征值 {eigenvalues[i]} 对应的特征向量: {eigenvectors[:, i]}")