import threading
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

q = -1.602 * 10 ** (-19)  # 电子电量
h = 6.62607015 * 10 ** (-34)  # 普朗克常数
m0 = 9.11 * 10 ** (-31)  # 电子质量
m_eff = m0  # 先用电子质量代替
sigma0 = 1  # 体电导率

"""矩形导线的SRFS模型"""


class SRFS:
    def __init__(self, p, a, b):
        self.p = p
        self.kapa_a = a
        self.kapa_b = b
        pass

    def specular_reflection_scattering(self, xn, yn):
        def func0(theta):
            return 2 * np.pi * np.cos(theta) ** 2 * np.sin(theta)

        int1, _ = integrate.quad(func0, 0, np.pi)

        def func1xy(fai, theta):
            # 设置一个最小值阈值，避免分母为 0 或接近 0
            epsilon = 1e-10
            sin_theta = np.sin(theta)
            sin_fai = np.sin(fai)
            cos_fai = np.cos(fai)

            # 避免分母为 0 或接近 0
            denominator_a = np.maximum(sin_theta * cos_fai, epsilon)
            denominator_b = np.maximum(sin_theta * sin_fai, epsilon)

            # 计算指数部分，限制指数范围避免溢出
            exp_kapa_b_yn = np.exp(
                np.clip(self.kapa_b * (yn - 1) / (2 * denominator_b), -700, 700)
            )
            exp_kapa_b_neg_yn = np.exp(
                np.clip(self.kapa_b * (-yn - 1) / (2 * denominator_b), -700, 700)
            )
            exp_kapa_a_xn = np.exp(
                np.clip(self.kapa_a * (xn - 1) / (2 * denominator_a), -700, 700)
            )
            exp_kapa_a_neg_xn = np.exp(
                np.clip(self.kapa_a * (-xn - 1) / (2 * denominator_a), -700, 700)
            )

            # 计算分母部分
            denominator_term_a = 1 - self.p * np.exp(
                np.clip(-self.kapa_a / denominator_a, -700, 700)
            )
            denominator_term_b = 1 - self.p * np.exp(
                np.clip(-self.kapa_b / denominator_b, -700, 700)
            )

            # 计算结果
            result = (1 - self.p) * (
                np.cos(theta) ** 2
                * sin_theta
                * (
                    (exp_kapa_b_yn + exp_kapa_b_neg_yn) / denominator_term_b
                    + (exp_kapa_a_xn + exp_kapa_a_neg_xn) / denominator_term_a
                )
            )
            return result

        int2, _ = integrate.dblquad(
            func1xy,
            0,
            np.pi / 2,
            lambda x: 0,
            lambda x: np.pi,
        )

        int3 = 0
        # 遍历 n 和 d 的值
        for n in [1 + yn, 1 - yn]:
            for d in [1 + xn, 1 - xn]:

                # 定义被积函数 func2xy
                def func2xy(fai, theta):
                    # 避免分母为零
                    if np.sin(theta) == 0 or np.sin(fai) == 0 or np.cos(fai) == 0:
                        return 0.0

                    # 计算指数部分的参数
                    exp1 = -self.kapa_a * d / (2 * np.sin(theta) * np.cos(fai))
                    exp2 = -self.kapa_a / (np.sin(theta) * np.cos(fai))
                    exp3 = -self.kapa_b * n / (2 * np.sin(theta) * np.sin(fai))
                    exp4 = -self.kapa_b / (np.sin(theta) * np.sin(fai))

                    # 计算指数函数
                    exp_val1 = np.exp(exp1)
                    exp_val2 = np.exp(exp2)
                    exp_val3 = np.exp(exp3)
                    exp_val4 = np.exp(exp4)

                    # 计算分母
                    # print(exp_val2)
                    denom1 = 1 - self.p * exp_val2
                    denom2 = 1 - self.p * exp_val4

                    # 避免除零
                    if denom1 <= 0 or denom2 <= 0:
                        return 0.0

                    # 计算最终结果
                    result = (
                        np.cos(theta) ** 2
                        * np.sin(theta)
                        * abs(exp_val1 / denom1 - exp_val3 / denom2)
                        / 2
                    )

                    return result

                # 计算二重积分并累加到 int3
                result, _ = integrate.dblquad(
                    func2xy,
                    0,
                    np.pi,
                    lambda x: 0,
                    lambda x: np.pi / 2,
                )
                int3 += result

        # print("int1 = ", int1)
        # print("int2 = ", int2)
        # print("int3 = ", int3)
        sigma = 3 / 4 * sigma0 * (int1 - int2 - int3) / np.pi
        return sigma

    def pure_diffusion_surface_scatter(self, xn, yn):
        self.p = 0
        sigma = self.specular_reflection_scattering(xn, yn)
        return sigma

    def film_expression(self, yn):
        def func0(theta):
            return 2 * np.pi * np.cos(theta) ** 2 * np.sin(theta)

        int1, _ = integrate.quad(func0, 0, np.pi)

        def func1xy(fai, theta):
            return (
                2
                * (1 - self.p)
                * (
                    np.cos(theta) ** 2
                    * np.sin(theta)
                    * (
                        (
                            np.exp(
                                self.kapa_b
                                * (yn - 1)
                                / (2 * np.sin(theta) * np.sin(fai))
                            )
                            + np.exp(
                                self.kapa_b
                                * (-yn - 1)
                                / (2 * np.sin(theta) * np.sin(fai))
                            )
                        )
                        / (
                            1
                            - self.p
                            * np.exp(-self.kapa_b / (np.sin(theta) * np.cos(fai)))
                        )
                    )
                )
            )

        int2, _ = integrate.dblquad(func1xy, 0, np.pi, lambda x: 0, lambda x: np.pi / 2)

        sigma = 3 / 4 * sigma0 * (int1 - int2) / np.pi
        return sigma

    def reset_params(self, ka, kb, p):
        self.kapa_a = ka
        self.kapa_b = kb
        self.p = p


def compute_conductivity_of_square_conductor(ka, kb, p):
    conducitivity = 0
    count = 0
    srfs_model = SRFS(p, ka, kb)
    srfs_model.reset_params(ka, kb, p)
    for xn in np.linspace(-1, 1, 10):
        for yn in np.linspace(-1, 1, 10):
            count = count + 1
            conducitivity = conducitivity + srfs_model.pure_diffusion_surface_scatter(
                xn, yn
            )
    conducitivity = conducitivity / count
    return conducitivity


inputp = 0
ka = 0.5
kb = ka
model = SRFS(inputp, ka, kb)
conducitivity_1 = []
# for kappa in np.linspace(0, 20, 41):
#     print(kappa)
#     conducitivity_1.append(conductivity_of_square_conductor(model, kappa, kappa, 0))

# with open("output.txt", "w", encoding="utf-8") as file:
#     # 遍历列表并将每个元素写入文件，每个元素占一行
#     for item in conducitivity_1:
#         file.write(str(item) + "\n")  # 添加换行符

# plt.plot(np.linspace(0, 20, 41), conducitivity_1)
# plt.xlabel(r"$\kappa_a = \kappa_b$")
# plt.ylabel(r"$Conducitivity$")
# plt.legend()
# conducitivity = model.pure_diffusion_surface_scatter(0, 0)

# print("原点电导率", conducitivity)
