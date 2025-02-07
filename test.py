import multiprocessing
import numpy as np
from scipy.integrate import quad


# 定义数值积分的函数
def integrate_function(params):
    # 解析参数
    a, b = params  # 积分区间 [a, b]

    # 定义被积函数（示例：f(x) = x^2）
    def integrand(x):
        return x**2

    # 执行数值积分
    result, error = quad(integrand, a, b)
    return result


# 主函数
def main():
    # 定义积分区间列表（每个进程处理一个区间）
    intervals = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]  # 示例区间
    num_processes = len(intervals)  # 进程数量等于区间数量

    # 创建进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 map 方法并行执行数值积分
        results = pool.map(integrate_function, intervals)

    # 打印结果
    print("所有进程完成，每个区间的积分结果：", results)
    print("总积分结果：", sum(results))


# 运行主函数
if __name__ == "__main__":
    main()
