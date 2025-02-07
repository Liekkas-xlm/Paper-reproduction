import multiprocessing
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

# 常量定义
q = -1.602 * 10 ** (-19)  # 电子电量
h = 6.62607015 * 10 ** (-34)  # 普朗克常数
m0 = 9.11 * 10 ** (-31)  # 电子质量
m_eff = m0  # 先用电子质量代替
sigma0 = 59  # 体电导率


# 矩形导线的SRFS模型
class SRFS:
    def __init__(self, p, a, b):
        self.reset_params(p, a, b)

    def reset_params(self, p, a, b):
        self.p = p
        self.kapa_a = a
        self.kapa_b = b

    def specular_reflection_scattering(self, xn, yn):
        int1 = np.pi * 4 / 3

        def func1xy(fai, theta):
            epsilon = 1e-10
            sin_theta = np.sin(theta)
            sin_fai = np.sin(fai)
            cos_fai = np.cos(fai)

            denominator_a = np.maximum(sin_theta * cos_fai, epsilon)
            denominator_b = np.maximum(sin_theta * sin_fai, epsilon)

            exp_kapa_b = np.exp(-self.kapa_b / (2 * denominator_b))
            exp_kapa_b_yn = exp_kapa_b ** (1 - yn)
            exp_kapa_b_neg_yn = exp_kapa_b ** (1 + yn)

            exp_kapa_a = np.exp(-self.kapa_a / (2 * denominator_a))
            exp_kapa_a_xn = exp_kapa_a ** (1 - xn)
            exp_kapa_a_neg_xn = exp_kapa_a ** (1 + xn)
            # exp_kapa_b_yn = np.exp(
            #     np.clip(self.kapa_b * (yn - 1) / (2 * denominator_b), -700, 700)
            # )
            # exp_kapa_b_neg_yn = np.exp(
            #     np.clip(self.kapa_b * (-yn - 1) / (2 * denominator_b), -700, 700)
            # )
            # exp_kapa_a_xn = np.exp(
            #     np.clip(self.kapa_a * (xn - 1) / (2 * denominator_a), -700, 700)
            # )
            # exp_kapa_a_neg_xn = np.exp(
            #     np.clip(self.kapa_a * (-xn - 1) / (2 * denominator_a), -700, 700)
            # )

            denominator_term_a = 1 - self.p * exp_kapa_a**2
            denominator_term_b = 1 - self.p * exp_kapa_b**2

            result = (1 - self.p) * (
                (sin_theta - sin_theta**3)
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
        # 记录开始时间
        # start_time = time.time()
        int3 = 0
        for n in [1 + yn, 1 - yn]:
            for d in [1 + xn, 1 - xn]:

                def func2xy(fai, theta):
                    # 避免分母为零
                    sin_theta = np.sin(theta)
                    sin_fai = np.sin(fai)
                    cos_fai = np.cos(fai)

                    if sin_theta == 0 or sin_fai == 0 or cos_fai == 0:
                        return 0.0

                    # exp1 = -self.kapa_a * d / (2 * np.sin(theta) * np.cos(fai))
                    exp2 = -self.kapa_a / (sin_theta * cos_fai)
                    # exp3 = -self.kapa_b * n / (2 * np.sin(theta) * np.sin(fai))
                    exp4 = -self.kapa_b / (sin_fai * sin_fai)

                    exp_val2 = np.exp(np.clip(exp2, -700, 700))
                    exp_val1 = exp_val2 ** (d / 2)
                    exp_val4 = np.exp(np.clip(exp4, -700, 700))
                    exp_val3 = exp_val4 ** (n / 2)

                    denom1 = 1 - self.p * exp_val2
                    denom2 = 1 - self.p * exp_val4

                    if denom1 <= 0 or denom2 <= 0:
                        return 0.0

                    result = (
                        (sin_theta - sin_theta**3)
                        * abs(exp_val1 / denom1 - exp_val3 / denom2)
                        / 2
                    )
                    return result

                result, _ = integrate.dblquad(
                    func2xy,
                    0,
                    np.pi,
                    lambda x: 0,
                    lambda x: np.pi / 2,
                )
                int3 += result
        # 记录结束时间
        # end_time = time.time()

        # 计算运行时间
        # elapsed_time = end_time - start_time  # 将秒换算成分钟
        # print(f"积分3运行时间:{elapsed_time:.6f} s")
        # print("int1 = ", int1)
        # print("int2 = ", int2)
        # print("int3 = ", int3)
        sigma = 3 / 4 * sigma0 * (int1 - int2 - int3) / np.pi
        return sigma

    def pure_diffusion_surface_scatter(self, xn, yn):
        self.p = 0
        return self.specular_reflection_scattering(xn, yn)


def compute_av_conductivity(param):
    ka, kb, p = param
    count = 0
    conductivity_sum = 0.0
    srfs_model = SRFS(p, ka, kb)
    for xn in [0]:  # np.linspace(-1, 1, 10):
        for yn in [0]:  # np.linspace(-1, 1, 10):
            count += 1
            conductivity_sum += srfs_model.pure_diffusion_surface_scatter(xn, yn)
    return conductivity_sum / count


def compute_conductivity_of_square_conductor():
    # kapa = np.linspace(0, 1, 3)  # 生成kapa值
    kapa = [20]
    conductivity_list = []
    p = 0
    for ka in kapa:
        kb = ka
        param = [ka, kb, p]
        print("运算 ka = ", ka)
        conductivity_list.append(compute_av_conductivity(param))
    # 可选：将结果保存到文件
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(
            "Conductivity versus κ for different specularity p for square wire in the proposed SRFS model\n"
        )
        for item in conductivity_list:
            file.write(str(item) + "\n")
    print("kapa对应的电导率 : ", conductivity_list)
    return kapa, conductivity_list


"""使用多进程求解电导率"""


def compute_conductivity_of_square_conductor_mp():
    p_list = [0]
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
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(
                "Conductivity versus κ for different specularity p = "
                + str(p)
                + " for square wire in the proposed SRFS model\n"
            )
            for item in results:
                file.write(str(item) + "\n")
        result_list.append(results)
    return kapa, result_list, p_list


def plot_conductivity_vs_kapa_square_wire(kapa, p_list, conductivitys):

    for conductivity, p in zip(conductivitys, p_list):
        plt.plot(kapa, conductivity, linewidth=2, label=f"p={p:.2f}")
    # 添加图例
    plt.legend()
    plt.xlabel(r"$\kappa_a = \kappa_b$")
    plt.ylabel("Conductivity(S/μm)")
    plt.title("Conductivity vs. kappa for different specularity p")


def main():
    # 记录开始时间
    start_time = time.time()
    kapa, conductivity_list, p = compute_conductivity_of_square_conductor_mp()
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = (end_time - start_time) / 60  # 将秒换算成分钟
    print(f"程序运行时间：{elapsed_time:.6f} 分钟")

    # 可选：绘制结果
    plot_conductivity_vs_kapa_square_wire(kapa, p, conductivity_list)
    plt.show()


if __name__ == "__main__":

    main()
