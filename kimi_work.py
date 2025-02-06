import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# 物理常量
q = -1.602e-19  # 电子电量 (C)
h = 6.62607015e-34  # 普朗克常数 (J·s)
m0 = 9.11e-31  # 电子质量 (kg)
m_eff = m0  # 有效质量，这里假设为电子质量
v_fermi = 1.57e6  # 假设的费米速度 (m/s)
lambda0 = 4.5e-8  # 电子平均自由程 (m)
sigma0 = (
    (8 * np.pi / 3) * (q**2 * m_eff * v_fermi**2 / h**3) * lambda0
)  # 体电导率 (S/m)


# SRFS 模型
class SRFSModel:
    def __init__(self, a, b, p):
        self.a = a  # 互连宽度 (m)
        self.b = b  # 互连高度 (m)
        self.p = p  # 表面反射率
        self.kapa_a = a / lambda0
        self.kapa_b = b / lambda0

    def eta(self, xn, yn, theta):
        def integrand(phi, xn, yn, theta):
            term1 = np.exp(
                -self.kapa_b * np.sin(theta) * np.sin(phi) * yn / 2
            ) * np.cosh(self.kapa_b * np.sin(theta) * np.sin(phi) * yn / 2)
            term2 = np.exp(
                -self.kapa_a * np.sin(theta) * np.cos(phi) * xn / 2
            ) * np.cosh(self.kapa_a * np.sin(theta) * np.cos(phi) * xn / 2)
            term3 = np.abs(
                np.exp(-self.kapa_a * np.sin(theta) * np.cos(phi) * xn)
                - np.exp(-self.kapa_b * np.sin(theta) * np.sin(phi) * yn)
            )
            return term1 + term2 - term3

        integral, _ = integrate.quad(integrand, 0, np.pi / 2, args=(xn, yn, theta))
        return integral

    def sigma(self, xn, yn):
        def integrand(theta, xn, yn):
            return self.eta(xn, yn, theta) * np.cos(theta) ** 2 * np.sin(theta)

        integral, _ = integrate.quad(integrand, 0, np.pi, args=(xn, yn))
        return (3 / 4) * sigma0 * integral


# 计算平均电导率
def average_conductivity(model, num_points=100):
    x_values = np.linspace(-1, 1, num_points)
    y_values = np.linspace(-1, 1, num_points)
    sigma_avg = 0
    for x in x_values:
        for y in y_values:
            sigma_avg += model.sigma(x, y)
    sigma_avg /= num_points**2
    return sigma_avg / sigma0  # 返回归一化的电导率


# 绘制 Fig 3(a): Square wire
kappa_values = np.linspace(0.1, 10, 100)
p_values = [0, 0.5, 0.9]

plt.figure(figsize=(12, 6))

# (a) Square wire
plt.subplot(1, 2, 1)
for p in p_values:
    sigma_avg_values = [
        average_conductivity(SRFSModel(a=kappa * lambda0, b=kappa * lambda0, p=p))
        for kappa in kappa_values
    ]
    plt.plot(kappa_values, sigma_avg_values, label=f"p={p}")
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$\sigma / \sigma_0$")
plt.title(r"(a) Square Wire ($\kappa_a = \kappa_b = \kappa$)")
plt.legend()

# (b) Thin film
plt.subplot(1, 2, 2)
kappa_b_values = np.linspace(0.1, 10, 100)
p_values = [0, 0.5]
for p in p_values:
    sigma_avg_values = [
        average_conductivity(SRFSModel(a=1e6 * lambda0, b=kappa_b * lambda0, p=p))
        for kappa_b in kappa_b_values
    ]
    plt.plot(kappa_b_values, sigma_avg_values, label=f"p={p}")
plt.xlabel(r"$\kappa_b$")
plt.ylabel(r"$\sigma / \sigma_0$")
plt.title(r"(b) Thin Film ($\kappa_a \to \infty$)")
plt.legend()

plt.tight_layout()
plt.show()
