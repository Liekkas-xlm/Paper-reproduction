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
sigma0 = 1  # 体电导率


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
        sigma = sigma0 * (1 - (3 / 4 * (1 - self.p) * int2) / np.pi)
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
        sigma = 3 / 4 * sigma0 * (int1 - (1 - self.p) * int2) / np.pi
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
            result_list[i],
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


def main():
    # 记录开始时间
    start_time = time.time()
    # kapa, conductivity_list, p = compute_conductivity_of_square_conductor()        #计算正方形导线的电导率
    # kapa, conductivity_list, p = compute_conductivity_of_square_conductor()
    # plot_proportion()
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
