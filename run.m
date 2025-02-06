% 常量定义
q = -1.602e-19; % 电子电量
h = 6.62607015e-34; % 普朗克常数
m0 = 9.11e-31; % 电子质量
m_eff = m0; % 先用电子质量代替



% 输入参数
inputp = 0;
ka = 20;
kb = ka;

% 创建模型实例
model = SRFS(inputp, ka, kb);

% 计算原点电导率
conducitivity = model.pure_diffusion_surface_scatter(0, 1);

% 输出结果
fprintf('原点电导率: %f\n', conducitivity);