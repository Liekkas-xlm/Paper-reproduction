import math
import numpy as np
from scipy import integrate

q = -1.602 * 10 ** (-19)  # 电子电量
h = 6.62607015 * 10 ** (-34)  # 普朗克常数
m0 = 9.11 * 10 ** (-31)  # 电子质量
m_eff = m0  # 先用电子质量代替

"""矩形导线的SRFS模型"""


class SRFS:
    def __init__(self, p, a, b):
        self.p = p
        self.kapa_a = a
        self.kapa_b = b
        pass

    def pure_diffusion_surface_scatter(self, xn, yn):
        # sigma0 = (8 * np.pi / 3) * (q**2 * m_eff**2 * v_tilde**2 / h**3) * λ0
        def func0(theta):
            return 2 * np.pi * math.cos(theta) ** 2 * math.sin(theta)

        int1 = integrate.quad(func0, 0, np.pi)

        def func1xy(theta, fai):
            return np.cos(theta) ** 2 * np.sin(theta) * np.exp(-self.kapa_b / (np.sin(theta) * np.sin(fai)) * np.cosh(-self.kapa_b * yn / (2 * np.sin(theta) * np.sin(fai)))) + np.exp(
                -self.kapa_a / (np.sin(theta) * np.sin(fai)) * np.cosh(-self.kapa_a * yn / (2 * np.sin(theta) * np.sin(fai)))
            )

        int2 = integrate.dblquad(func1xy, 0, np.pi, lambda x: 0, lambda x: np.pi / 2)

        def func2xy(theta, fai, n, d):
            np.cos(theta) ** 2 * np.sin(theta) * abs(
                np.exp(-self.kapa_a * d / (np.sin(theta) * np.sin(fai)) * np.cosh(-self.kapa_a * d / (2 * np.sin(theta) * np.sin(fai))))
                - np.exp(-self.kapa_b * n / (np.sin(theta) * np.sin(fai)) * np.cosh(-self.kapa_a * d / (2 * np.sin(theta) * np.sin(fai))))
            )

        int3 = 0

        for n in [1 + yn, 1 - yn]:
            for d in [1 + xn, 1 - xn]:
                int3 = int3 + integrate.dblquad(func2xy, 0, np.pi, lambda x: 0, lambda x: np.pi / 2, args=(n, d))
        sigma0 = 59
        sigma = 3 / 4 * sigma0 * (int1 - 2 * int2 - 1 / 2 * int3)
        return sigma


inputp = 0
ka = range(0, 20, 0.1)
kb = ka
model = SRFS(inputp, ka, kb)
conducitivity = model.pure_diffusion_surface_scatter()
