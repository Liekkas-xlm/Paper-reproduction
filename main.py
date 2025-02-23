import multiprocessing
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import IntegrationWarning

# 常量定义
q = -1.602 * 10 ** (-19)  # 电子电量
h = 6.62607015 * 10 ** (-34)  # 普朗克常数
m0 = 9.11 * 10 ** (-31)  # 电子质量
m_eff = m0  # 先用电子质量代替
sigma0 = 59  # 体电导率
lambda0 = 40  # 铜的电子自由程


def safe_integration(func, a, b, c, d, params):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=IntegrationWarning)
        result, error = integrate.dblquad(
            func, a, b, lambda x: c, lambda x: d, epsabs=5e-6, epsrel=5e-6
        )
        if w:
            print("params = ", params)
            print("IntegrationWarning occurred, error = ", error)
    return result, error


# 导线的双曲模型
class COSH:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.rou_b = 32.05
        self.rou_q = 82
        self.lambda_d = 3.75

    def compute_conductivity(self, x, y):
        rou = self.rou_b + self.rou_q * (
            np.cosh(x / self.lambda_d) / np.cosh(self.w / self.lambda_d / 2)
            + np.cosh(y / self.lambda_d) / np.cosh(self.h / self.lambda_d / 2)
        )
        sigma = 1 / rou * 1000
        return sigma


# 矩形导线的SRFS模型
class SRFS:
    def __init__(self, p, a, b):
        self.reset_params(p, a, b)

    def reset_params(self, p, a, b):
        self.p = p
        self.kapa_a = a
        self.kapa_b = b

    def specular_reflection_scattering(self, xn, yn):

        def func1xy(fai, theta):
            epsilon = 1e-10
            sin_theta = np.sin(theta)
            sin_fai = np.sin(fai)
            cos_fai = np.cos(fai)

            denominator_a = np.maximum(sin_theta * cos_fai, epsilon)
            denominator_b = np.maximum(sin_theta * sin_fai, epsilon)

            exp_kapa_a = np.exp(-self.kapa_a / (2 * denominator_a))
            exp_kapa_a_xn = exp_kapa_a ** (1 - xn)
            exp_kapa_a_neg_xn = exp_kapa_a ** (1 + xn)

            exp_kapa_b = np.exp(-self.kapa_b / (2 * denominator_b))
            exp_kapa_b_yn = exp_kapa_b ** (1 - yn)
            exp_kapa_b_neg_yn = exp_kapa_b ** (1 + yn)

            denominator_term_a = 1 - self.p * exp_kapa_a**2
            denominator_term_b = 1 - self.p * exp_kapa_b**2

            exp_xn_divided_by_ka = exp_kapa_a_xn / denominator_term_a
            exp_neg_xn_divided_by_ka = exp_kapa_a_neg_xn / denominator_term_a
            exp_yn_divided_by_kb = exp_kapa_b_yn / denominator_term_b
            exp_neg_yn_divided_by_kb = exp_kapa_b_neg_yn / denominator_term_b

            result = (sin_theta - sin_theta**3) * (
                exp_xn_divided_by_ka
                + exp_neg_xn_divided_by_ka
                + exp_yn_divided_by_kb
                + exp_neg_yn_divided_by_kb
                + (
                    abs(exp_xn_divided_by_ka - exp_yn_divided_by_kb)
                    + abs(exp_xn_divided_by_ka - exp_neg_yn_divided_by_kb)
                    + abs(exp_neg_xn_divided_by_ka - exp_yn_divided_by_kb)
                    + abs(exp_neg_xn_divided_by_ka - exp_neg_yn_divided_by_kb)
                )
                / 2
            )
            return result

        int2, _ = safe_integration(func1xy, 0, np.pi, 0, np.pi / 2, [xn, yn])
        sigma = 1 - (3 / 4 * (1 - self.p) * int2) / np.pi
        return sigma

    def pure_diffusion_surface_scatter(self, xn, yn):
        p_temp = self.p
        self.p = 0
        result = self.specular_reflection_scattering(xn, yn)
        self.p = p_temp
        return result

    def expression_for_thin_films(self, yn):
        int1 = np.pi * 4 / 3

        def func1xy(fai, theta):
            epsilon = 1e-10
            sin_theta = np.sin(theta)
            sin_fai = np.sin(fai)

            denominator = np.maximum(sin_theta * sin_fai, epsilon)

            exp_kapa_b = np.exp(-self.kapa_b / (2 * denominator))
            exp_kapa_b_yn = exp_kapa_b ** (1 - yn)
            exp_kapa_b_neg_yn = exp_kapa_b ** (1 + yn)

            denominator_term = 1 - self.p * exp_kapa_b**2

            result = (
                2
                * (sin_theta - sin_theta**3)
                * (exp_kapa_b_yn + exp_kapa_b_neg_yn)
                / denominator_term
            )
            return result

        int2, _ = safe_integration(func1xy, 0, np.pi, 0, np.pi / 2, [yn])
        sigma = 3 / 4 * (int1 - (1 - self.p) * int2) / np.pi
        return sigma


"""计算正方形导线的平均电导率"""


def compute_av_conductivity(param):
    ka, kb, p = param
    count = 0
    conductivity_sum = 0.0
    srfs_model = SRFS(p, ka, kb)
    for xn in np.linspace(-1, 1, 15):
        for yn in np.linspace(-1, 1, 15):
            count += 1
            conductivity_sum += srfs_model.specular_reflection_scattering(xn, yn)
    return conductivity_sum / count


"""计算薄膜电导率"""


def compute_film_av_conductivity(param):
    kb, p = param
    count = 0
    conductivity_sum = 0.0
    srfs_model = SRFS(p, 0, kb)
    for yn in np.linspace(-1, 1, 20):
        count += 1
        conductivity_sum += srfs_model.expression_for_thin_films(yn)
    return conductivity_sum / count


def compute_conductivity_of_square_conductor():
    p_list = [0.9]
    # kapa = np.linspace(0, 1, 3)  # 生成kapa值
    kapa = [0.2]
    for p in p_list:
        conductivity_list = []
        for ka in kapa:
            kb = ka
            param = [ka, kb, p]
            # print("运算 ka = ", ka)
            conductivity_list.append(compute_av_conductivity(param))
        # 可选：将结果保存到文件
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(
                "Conductivity versus κ for different specularity p for square wire in the proposed SRFS model\n"
            )
            for item in conductivity_list:
                file.write(str(item) + "\n")
        print("kapa对应的电导率 : ", conductivity_list)
    return kapa, conductivity_list, p_list


"""使用多进程求解电导率"""


def compute_conductivity_of_square_conductor_mp():
    p_list = [0, 0.2, 0.6, 0.9]
    kapa = np.linspace(0, 20, 41)  # 生成kapa值
    result_list = []
    for p in p_list:
        # 使用列表推导式生成所需的列表
        params_list = [(k, k, p) for k in kapa]
        # 进程数
        num_processes = len(kapa)  # 进程数量等于区间数量
        # 创建进程池
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用 map 方法并行执行数值积分
            results = pool.map(compute_av_conductivity, params_list)
        # 打印结果
        # print("所有进程完成,每个kapa的电导率:", results)

        # 可选：将结果保存到文件
        # with open("output.txt", "w", encoding="utf-8") as file:
        #     file.write(
        #         "Conductivity versus κ for different specularity p = "
        #         + str(p)
        #         + " for square wire in the proposed SRFS model\n"
        #     )
        #     for item in results:
        #         file.write(str(item) + "\n")
        result_list.append(results)
    return kapa, result_list, p_list


"""求解薄膜电导率"""


def compute_conductivity_of_film():
    p_list = [0]
    # kapa = np.linspace(0, 20, 41)  # 生成kapa值
    kapa = [20]
    for p in p_list:
        conductivity_list = []
        for kb in kapa:
            param = [kb, p]
            # print("运算 kb = ", kb)
            conductivity_list.append(compute_film_av_conductivity(param))
        print("kapa对应的电导率 : ", conductivity_list)
    return kapa, conductivity_list, p_list


def plot_conductivity_vs_kapa_square_wire(kapa, p_list, conductivitys, title_str):
    for conductivity, p in zip(conductivitys, p_list):
        plt.plot(kapa, conductivity, linewidth=2, label=f"p={p:.2f}")
    # 添加图例
    plt.legend()
    plt.xlabel(title_str)
    plt.ylabel("Conductivity(S/μm)")
    plt.title("Conductivity vs. kappa for different specularity p")


"""conductivity normalized to bulk conductivity) for different κ and p for a square wire comparing the results"""


def plot_proportion():
    p_list = [0.1, 0.5, 0.9]
    kapa = [0.2, 0.5, 1, 1.5, 2]  # 生成kapa值
    result_list = []
    for p in p_list:
        # 使用列表推导式生成所需的列表
        params_list = [(k, k, p) for k in kapa]
        # 进程数
        num_processes = len(kapa)  # 进程数量等于区间数量
        # 创建进程池
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用 map 方法并行执行数值积分
            results = pool.map(compute_av_conductivity, params_list)
        # 打印结果
        # print("所有进程完成,每个kapa的电导率:", results)

        # 可选：将结果保存到文件
        # with open("output.txt", "w", encoding="utf-8") as file:
        #     file.write(
        #         "Conductivity versus κ for different specularity p = "
        #         + str(p)
        #         + " for square wire in the proposed SRFS model\n"
        #     )
        #     for item in results:
        #         file.write(str(item) + "\n")
        result_list.append(results)
    markers = ["o", "*", "^"]
    plt.figure(1)

    for i in range(len(p_list)):
        plt.plot(
            kapa,
            result_list[i] / sigma0,
            marker=markers[i],
            linewidth=2,
            label=f"p={p_list[i]:.2f}",
        )

    # 设置x轴和y轴的范围
    plt.xlim(0, 2)
    plt.ylim(0, 1)
    # 添加图例
    plt.legend()
    plt.xlabel(r"$\kappa_a = \kappa_b$")
    plt.ylabel("Conductivity(S/μm)")
    plt.title("Conductivity vs. kappa for different specularity p")
    pass


"""计算电导率的空间依赖性"""


def compute_spatial_dependence_of_conductivity():
    # 计算p = 0.663时候的平均电导率
    p = 0.663
    a = 10  # 互连宽度
    b = 29  # 互连厚度
    kapa_a = a / lambda0
    kapa_b = b / lambda0

    # 计算原点电导率
    srfs_model = SRFS(p, kapa_a, kapa_b)

    cosh_model = COSH(10, 25)
    conductivity_cosh = cosh_model.compute_conductivity(0, 0)
    print("双曲模型的电导率 : ", conductivity_cosh)

    lumped_parameter = 1 / 32.05 * 1000
    conductivity = (
        compute_av_conductivity([kapa_a, kapa_b, p]) - 1
    ) * sigma0 + lumped_parameter
    print("p = 0.663对应的集总参数电导率 : ", conductivity)

    # 创建一个图形
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列的子图
    axs_index = 0

    # 计算固定 y 值,沿 x 方向的值
    for yn in [0, 7.25 * 2 / b]:
        conductivity_list = []
        xn_list = np.linspace(-1, 1, 101)
        for xn in xn_list:
            conductivity_list.append(srfs_model.specular_reflection_scattering(xn, yn))
        axs[0, axs_index].plot(
            xn_list * 5,
            [(x - 1) * sigma0 + lumped_parameter for x in conductivity_list],
            linewidth=2,
        )
        axs[0, axs_index].set_title(f"y = {yn} nm")
        axs[0, axs_index].set_xlabel("X(nm)")
        axs[0, axs_index].set_ylabel("Conductivity(S/μm)")
        # 设置x轴和y轴的范围
        axs[0, axs_index].set_xlim([-5, 5])
        axs[0, axs_index].set_ylim([6, 14])
        axs_index = axs_index + 1

    axs_index = 0
    # 计算固定 x 值,沿 y 方向的值
    for xn in [0, 2.5 * 2 / a]:
        conductivity_list = []
        yn_list = np.linspace(-1, 1, 101)
        for yn in yn_list:
            conductivity_list.append(srfs_model.specular_reflection_scattering(xn, yn))
        axs[1, axs_index].plot(
            yn_list * b / 2,
            [(x - 1) * sigma0 + lumped_parameter for x in conductivity_list],
            linewidth=2,
        )
        axs[1, axs_index].set_title(f"x = {xn} nm")
        axs[1, axs_index].set_xlabel("Y(nm)")
        axs[1, axs_index].set_ylabel("Conductivity(S/μm)")
        # 设置x轴和y轴的范围
        axs[1, axs_index].set_xlim([-20, 20])
        axs[1, axs_index].set_ylim([7, 14])
        axs_index = axs_index + 1


"""计算归一化电导率分布"""


def compute_normalized_conductivity():
    # 计算 p = 0.663 时候的平均电导率
    p_list = [0, 0.663, 0.9999]
    a = 10  # 互连宽度
    b = 29  # 互连厚度
    kapa_a = a / lambda0
    kapa_b = b / lambda0

    # 计算原点电导率
    srfs_model = SRFS(0, kapa_a, kapa_b)
    # 计算固定 x 值,沿 x 方向的值
    plt.figure()
    for p in p_list:
        srfs_model.reset_params(p, kapa_a, kapa_b)
        conductivity_list = []
        yn_list = np.linspace(-1, 1, 101)
        for yn in yn_list:
            conductivity_list.append(srfs_model.specular_reflection_scattering(0, yn))
        min_delta = min(conductivity_list)
        max_delta = max(conductivity_list)
        l = max_delta - min_delta
        plt.plot(
            yn_list * b / 2,
            [(c - min_delta) / (max_delta - min_delta) for c in conductivity_list],
            linewidth=2,
            label=f"p={p:.2f}",
        )

    # 设置x轴和y轴的范围
    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(0, 1)
    plt.xlabel("y(nm)")
    plt.ylabel(f"$(δ-δ_min)/(δ_max-δ_min)$")
    pass


"""计算不同kapa和p值下的正方形导线的δ/δ0,绘制空间分布图"""


def plot_3d_spatial_distribution_of_conductivity():
    p_list = [0, 0.9]
    k_list = [0.2, 1, 5]
    w_list = [8, 40, 200]  # 矩形导线的尺寸
    # 计算原点电导率
    srfs_model = SRFS(0, 1, 1)
    # 创建一个新的图形
    fig = plt.figure(figsize=(10, 8))
    point_nums = 101
    for i in range(3):  # 遍历kapa
        interval = np.linspace(-w_list[i] / 2, w_list[i] / 2, point_nums)
        X, Y = np.meshgrid(interval, interval)
        conductivity_result = np.zeros_like(X)  # 初始化结果数组
        for j in range(2):  # 遍历p
            ax = fig.add_subplot(3, 2, i * 2 + j + 1, projection="3d")
            srfs_model.reset_params(p_list[j], k_list[i], k_list[i])
            for m in range(X.shape[0]):
                for n in range(X.shape[0]):
                    conductivity_result[m, n] = (
                        srfs_model.specular_reflection_scattering(
                            interval[m] / w_list[i] * 2, interval[n] / w_list[i] * 2
                        )
                    )
        ax.plot_surface(X, Y, conductivity_result, cmap="viridis")
        ax.set_title(f"p={p_list[j]}, k={k_list[i]}")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_zlabel(r"$\sigma/\sigma_0$")
        ax.set_zlim(0, np.max(conductivity_result) * 1.1)  # 设置Z轴的范围
    pass


def main():
    # 记录开始时间
    start_time = time.time()
    # kapa, conductivity_list, p = compute_conductivity_of_square_conductor()        #计算正方形导线的电导率
    # kapa, conductivity_list, p = compute_conductivity_of_square_conductor()
    # plot_proportion()
    # compute_spatial_dependence_of_conductivity()
    # compute_normalized_conductivity()
    plot_3d_spatial_distribution_of_conductivity()
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = (end_time - start_time) / 60  # 将秒换算成分钟
    print(f"程序运行时间：{elapsed_time:.6f} 分钟")

    squre_title = r"$\kappa_a = \kappa_b$"
    film_title = r"$\kappa_b$"

    # 可选：绘制结果
    # plot_conductivity_vs_kapa_square_wire(kapa, p, conductivity_list)
    plt.show()


if __name__ == "__main__":
    main()
