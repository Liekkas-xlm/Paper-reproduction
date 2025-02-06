% 矩形导线的SRFS模型
classdef SRFS
    properties
        p
        kapa_a
        kapa_b
    end
    
    methods
        function obj = SRFS(p, a, b)
            obj.p = p;
            obj.kapa_a = a;
            obj.kapa_b = b;
        end
        
        function sigma = pure_diffusion_surface_scatter(obj, xn, yn)
            % 计算第一个积分
            int1 = integral(@(theta) 2 * pi * cos(theta).^2 .* sin(theta), 0, pi);
            
            % 计算第二个积分
            int2 = integral2(@(theta, fai) cos(theta).^2 .* sin(theta) .* ...
                exp(-obj.kapa_b ./ (sin(theta) .* sin(fai)) .* ...
                cosh(-obj.kapa_b * yn ./ (2 * sin(theta) .* sin(fai)))) + ...
                exp(-obj.kapa_a ./ (sin(theta) .* sin(fai)) .* ...
                cosh(-obj.kapa_a * yn ./ (2 * sin(theta) .* sin(fai)))), ...
                0, pi, 0, pi/2);
            
            % 计算第三个积分
            int3 = 0;
            n_values = [1 + yn, 1 - yn];
            d_values = [1 + xn, 1 - xn];
            
            for n = n_values
                for d = d_values
                    int3 = int3 + integral2(@(theta, fai) cos(theta).^2 .* sin(theta) .* ...
                        abs(exp(-obj.kapa_a * d ./ (sin(theta) .* sin(fai)) .* ...
                        cosh(-obj.kapa_a * d ./ (2 * sin(theta) .* sin(fai)))) - ...
                        exp(-obj.kapa_b * n ./ (sin(theta) .* sin(fai)) .* ...
                        cosh(-obj.kapa_a * d ./ (2 * sin(theta) .* sin(fai))))), ...
                        0, pi, 0, pi/2);
                end
            end
            
            sigma0 = 59; % 假设的初始值
            sigma = 3/4 * sigma0 * (int1 - 2 * int2 - 1/2 * int3);
        end
    end
end