import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个新的图形
fig = plt.figure(figsize=(12, 8))

# 定义参数
p_values = [0, 0.9]
k_values = [0.2, 1, 5]
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)


# 定义Z值的计算函数
def calculate_Z(X, Y, p, k):
    Z = np.sqrt(X**2 + Y**2)
    return (Z / np.max(Z)) ** p * (1 - np.exp(-k * Z**2))


# 绘制每个子图
for i, p in enumerate(p_values):
    for j, k in enumerate(k_values):
        ax = fig.add_subplot(2, 3, i * 3 + j + 1, projection="3d")
        Z_val = calculate_Z(X, Y, p, k)
        ax.plot_surface(X, Y, Z_val, cmap="viridis")
        ax.set_title(f"p={p}, k={k}")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_zlabel(r"$\sigma/\sigma_0$")
        ax.set_zlim(0, np.max(Z_val) * 1.1)  # 设置Z轴的范围

# 调整布局
plt.tight_layout()
plt.show()
