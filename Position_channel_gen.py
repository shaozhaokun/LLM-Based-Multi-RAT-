import numpy as np
from scipy.special import j1  # Bessel 函数，用于计算辐射方向图

# RAT 索引规则（根据论文模型）:
# M1=2 个 6G BSs: RAT 0, 1 (索引范围 [0, M1-1])
# M2=2 个 Wi-Fi BSs: RAT 2, 3 (索引范围 [M1, M1+M2-1])
# M3=卫星 BSs: RAT 4..(M-1) (索引范围 [M1+M2, M-1])
# 总共 M = M1 + M2 + M3，其中 M3 由 RAT_num - (M1+M2) 推断

class RATDistanceCalculator:
    def __init__(self, urllc_num, embb_num, RAT_num, time_):
        """
        初始化RAT距离计算器。
        
        参数:
        - urllc_num: URLLC用户数量
        - embb_num: eMBB用户数量
        - RAT_num: RAT数量（应该是 5：2个6G + 2个Wi-Fi + 1个卫星）
        - time_: 随机种子
        """
        self.URLLC_num = urllc_num
        self.eMBB_num = embb_num
        self.num_users = urllc_num + embb_num
        self.RAT_num = RAT_num
        
        # 根据论文模型定义索引范围
        self.M1 = 2  # 6G BSs 数量
        self.M2 = 2  # Wi-Fi BSs 数量
        self.M3 = RAT_num - (self.M1 + self.M2)  # 卫星 BSs 数量（自动推断）
        assert self.M3 > 0, f"RAT_num={RAT_num} 太小：需要至少 {self.M1 + self.M2 + 1} 才能包含卫星"
         
        self.seed = time_
        np.random.seed(time_)
        
        # ----------------------------
        # 卫星信道参数（用于 m ∈ M3）- Ku/Ka 波段配置
        # ----------------------------
        self.c_light = 3e8            # 光速 [m/s]
        # Ku/Ka 波段频率设置：
        # - Ku 波段：上行 ~14 GHz，下行 ~12 GHz（Starlink 早期用，路径损耗相对较小）
        # - Ka 波段：上行 ~30 GHz，下行 ~20 GHz（Starlink/OneWeb 主流，路径损耗更大）
        # 建议：先用 Ku 波段（14 GHz），路径损耗是 2 GHz 的 49 倍，比 Ka 波段的 225 倍更容易补偿
        # 如果需要 Ka 波段，可以改为 30e9
        self.f_up_sat = np.full(self.M3, 14e9, dtype=float)   # [Hz] Ku 波段上行 14 GHz
        # 天线增益（线性值）- Ku/Ka 波段需要更大的天线增益来补偿更高的路径损耗：
        # 频率从 2 GHz 增加到 14 GHz（Ku 波段），FSPL 增加 (14/2)² = 49 倍
        # 虽然频率降低了，但 Ku 波段仍然需要很大的增益来补偿路径损耗
        # 注意：保持较大的增益值，因为即使 Ku 波段路径损耗比 Ka 波段小，仍然很大
        # - G_m（卫星天线增益）：Ku 波段大型天线可达 40-50 dBi
        # - g（用户终端天线增益）：Ku 波段固定终端可达 30-40 dBi
        self.G_sat = np.full(self.M3, 1000000.0, dtype=float)  # G_m（线性，60 dBi，保持较大值）
        self.g_sat = np.full(self.M3, 100000.0, dtype=float)   # g_{k,m}^{up,phi}（线性，50 dBi，保持较大值）
        # 注意：新模型（基于 Friis 传输方程）不再使用雨衰参数
        # 以下参数保留是为了兼容性，但新模型中不会被使用
        # 如果需要考虑雨衰，可以在新模型的基础上额外添加雨衰项
        self.rain_rho = 1e-5           # 雨衰参数 ρ（已废弃，新模型不使用）
        self.rain_eta = 1.1           # 雨衰指数 η（已废弃，新模型不使用）
        self.rain_intensity = 0.5     # 降雨强度 I [mm/h]（已废弃，新模型不使用）

        # 设置 RAT 位置（3D 坐标系统，单位：米）
        # 地面 BSs (z=0)，卫星 BS (z 很高)
        terrestrial_positions = np.array(
            [
                [-250.0,  250.0,    0.0],  # RAT 0: 6G BS 1
                [ 250.0,  250.0,    0.0],  # RAT 1: 6G BS 2
                [-250.0, -250.0,    0.0],  # RAT 2: Wi-Fi BS 1
                [ 250.0, -250.0,    0.0],  # RAT 3: Wi-Fi BS 2
            ],
            dtype=float,
        )

        # 默认给卫星放两个不同的投影位置（高度 550 km）
        # - 若 M3=1，只取第一颗
        # - 若 M3>2，可继续按需扩展这个列表或改成规则生成
        sat_height = 550000.0
        sat_positions_pool = np.array(
            [
                [-300.0, 0.0, sat_height],        # SAT 1
                [300.0, 0.0, sat_height],   # SAT 2（与 SAT1 拉开水平距离）
            ],
            dtype=float,
        )
        sat_positions = sat_positions_pool[: self.M3, :]

        self.RAT_positions = np.vstack([terrestrial_positions, sat_positions])
        
        # 可选：如果你想用不同的布局，可以取消注释下面的配置
        # 方案2：6G 和 Wi-Fi 更分散
        # self.RAT_positions = np.array([
        #     [-500.0,  500.0,    0.0],  # RAT 0: 6G BS 1
        #     [ 500.0,  500.0,    0.0],  # RAT 1: 6G BS 2
        #     [-500.0, -500.0,    0.0],  # RAT 2: Wi-Fi BS 1
        #     [ 500.0, -500.0,    0.0],  # RAT 3: Wi-Fi BS 2
        #     [   0.0,    0.0, 550000.0],  # RAT 4: 卫星 BS
        # ], dtype=float)

    def generate_user_positions(self):
        """
        生成随机的用户位置（3D坐标系统），并按照 K1/K2/K3 三类用户划分：
        - K1: city，只能访问地面 BSs（6G+WiFi），z ≈ 0
        - K2: ocean，只能访问卫星 BSs，z ≈ 0（海平面）
        - K3: hybrid，可访问地面和卫星，z ≈ 0

        当前实现仅在索引上区分 K1/K2/K3，空间位置都在同一个方形区域内；
        如需空间上区分三个区域，可以进一步为三类用户设置不同的 (x, y) 范围。

        返回:
        - 用户位置的NumPy数组，形状为 (num_users, 3)，单位：米
        """
        K = self.num_users

        # 简单地按 1/3 划分 K1/K2/K3
        K1 = K // 3
        K2 = K // 3
        K3 = K - K1 - K2

        # 在指定范围内生成用户位置（x, y 在 [-1000, 1000] 米，z ≈ 0 表示地面/海平面）
        user_positions_xy = np.random.uniform(-1000, 1000, (K, 2))
        user_positions_z = np.zeros((K, 1))  # 所有用户都在地面/海平面（z=0）
        user_positions = np.hstack([user_positions_xy, user_positions_z])

        # 记录三类用户的数量和索引，用于后续接入/关联约束
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.idx_K1 = np.arange(0, K1)
        self.idx_K2 = np.arange(K1, K1 + K2)
        self.idx_K3 = np.arange(K1 + K2, K)

        return user_positions



    def calculate_DistancesAndChennel(self, user_positions):
        """
        计算用户到各RAT的距离（3D坐标系统）。
        
        参数:
        - user_positions: 用户的位置，形状为 (num_users, 3)，单位：米
        
        返回值:
        - dk_m: 用户到各RAT的距离矩阵，形状为 (num_users, RAT_num)，单位：米
        - channel: 信道增益矩阵，形状为 (num_users, RAT_num)
        """
        num_users = user_positions.shape[0]
        
        # 确保用户位置是 3D 的（如果是 2D，补 z=0）
        if user_positions.shape[1] == 2:
            user_positions = np.hstack([user_positions, np.zeros((num_users, 1))])
        
        dk_m = np.zeros((num_users, self.RAT_num))  # 初始化距离矩阵
        pathloss_exp = 3  # 地面链路路径损耗指数（原代码 rho=3）

        # 计算每个用户到各RAT的3D欧几里得距离
        for i in range(num_users):
            for j in range(self.RAT_num):
                dk_m[i, j] = np.linalg.norm(user_positions[i] - self.RAT_positions[j])  # 3D距离

        dk_m_urllc = dk_m[:self.URLLC_num, :]
        dk_m_eMBB = dk_m[self.URLLC_num:self.URLLC_num + self.eMBB_num, :]

        # 添加小尺度衰落（每个 RAT 各自独立 CN(0,1)）
        small_scale_fading = (
            np.random.randn(self.URLLC_num + self.eMBB_num, self.RAT_num)
            + 1j * np.random.randn(self.URLLC_num + self.eMBB_num, self.RAT_num)
        ) / np.sqrt(2)

        # ----------------------------
        # 地面链路（m ∈ M1 ∪ M2）：沿用原来的路径损耗模型 |h| ∝ d^{-α/2}
        # ----------------------------
        channel_ = small_scale_fading / (dk_m ** (pathloss_exp / 2))

        # ----------------------------
        # 卫星链路（m ∈ M3）：按新模型（基于 Friis 传输方程）计算
        # h_i^{u2s} = A_i^{trans} * A_s^{rec} * Γ(θ_{is}) * (C / (4π L_i^{u2s} F))^2
        # 其中：
        # - A_i^{trans}: 用户发射天线增益（g_sat）
        # - A_s^{rec}: 卫星接收天线增益（G_sat）
        # - Γ(θ_{is}): 辐射方向图（与仰角相关）
        # - L_i^{u2s}: 距离（dk_m）
        # - F: 频率（GHz）
        # - C: 光速
        # ----------------------------
        sat_start = self.M1 + self.M2
        sat_indices = list(range(sat_start, self.RAT_num))
        eps = 1e-12
        
        for s, sat_idx in enumerate(sat_indices):
            # 获取卫星位置和参数
            sat_pos = self.RAT_positions[sat_idx, :]  # (x, y, z)
            d_sat = dk_m[:, sat_idx]  # 斜距（3D距离）
            f_up = self.f_up_sat[s] if s < self.f_up_sat.shape[0] else self.f_up_sat[-1]
            f_up_ghz = f_up / 1e9  # 转换为 GHz
            A_trans = self.g_sat[s] if s < self.g_sat.shape[0] else self.g_sat[-1]  # 用户天线增益
            A_rec = self.G_sat[s] if s < self.G_sat.shape[0] else self.G_sat[-1]   # 卫星天线增益
            
            # 计算仰角 θ_{is}（卫星与用户连线与水平面的夹角）
            # 仰角 = arctan((h_sat - h_user) / d_horizontal)
            h_sat = sat_pos[2]  # 卫星高度 [m]
            h_user = user_positions[:, 2]  # 用户高度 [m]（通常为 0，地面）
            delta_h = h_sat - h_user  # 高度差
            
            # 水平距离
            d_horizontal = np.sqrt(
                (sat_pos[0] - user_positions[:, 0]) ** 2 + 
                (sat_pos[1] - user_positions[:, 1]) ** 2
            )
            # 仰角（弧度）：θ = arctan(delta_h / d_horizontal)
            # 注意：当 d_horizontal = 0 时（用户正对卫星下方），θ = 90°
            theta_rad = np.arctan2(delta_h, d_horizontal + eps)  # 使用 arctan2 避免除零
            theta_deg = np.rad2deg(theta_rad)  # 转换为度
            
            # 计算辐射方向图 Γ(θ)（Equation 6）
            # Γ(θ) = 1, if θ = 90°
            #      = 4 |J_1(20πcosθ) / (20πcosθ)|^2, if 0° ≤ θ < 90°
            Gamma = np.ones_like(theta_deg)
            mask_not_90 = np.abs(theta_deg - 90.0) > 1e-6  # 不是 90° 的情况
            if np.any(mask_not_90):
                theta_mask = theta_rad[mask_not_90]
                cos_theta = np.cos(theta_mask)
                arg = 20.0 * np.pi * cos_theta
                # 避免除零：当 arg = 0 时，J_1(0)/0 的极限是 0.5
                j1_val = j1(arg)
                ratio = np.where(np.abs(arg) > eps, j1_val / arg, 0.5)
                Gamma[mask_not_90] = 4.0 * np.abs(ratio) ** 2
            
            # 计算信道增益（基于 Friis 传输方程）
            # h = A_trans * A_rec * Γ(θ) * (C / (4π L F))^2
            # 注意：论文中 F 的单位是 GHz，但 Friis 公式中频率单位应该是 Hz
            # 所以需要转换：F_Hz = F_GHz * 1e9
            C = self.c_light  # 光速 [m/s] = 3e8
            L = d_sat  # 距离 [m]
            F_Hz = f_up  # 频率 [Hz]，从 GHz 转换：f_up 已经是 Hz
            
            # 自由空间路径损耗因子：(C / (4π L F))^2
            # C [m/s], L [m], F [Hz]
            fspl_factor = (C / (4.0 * np.pi * L * F_Hz + eps)) ** 2
            
            # 最终信道幅度（线性值）
            sat_mag = A_trans * A_rec * Gamma * fspl_factor
            
            # 乘以小尺度衰落
            channel_[:, sat_idx] = small_scale_fading[:, sat_idx] * sat_mag

        # 按业务类型拆分（保持原有接口）
        URLLC_h = channel_[:self.URLLC_num, :]
        eMBB_h = channel_[self.URLLC_num:self.URLLC_num + self.eMBB_num, :]

        # 根据 RAT 类型设置不同的增益（单位：线性）
        # RAT 0, 1: 6G BSs (M1) - 较高增益
        # RAT 2, 3: Wi-Fi BSs (M2) - 中等增益
        # RAT 4, 5: 卫星 BS (M3) - 需要很大的增益来补偿路径损耗
        # 注意：这些增益值可以根据你的实际模型调整
        # 增益向量长度需与 RAT_num 一致：地面 4 个 + M3 个卫星
        gain_terrestrial = np.array([10.0, 10.0, 5.0, 5.0], dtype=float)
        # 卫星增益需要很大来补偿巨大的路径损耗（550km 距离 + Ku/Ka 波段频率）
        # 频率从 2 GHz 增加到 14 GHz（Ku 波段），路径损耗增加 49 倍
        # 虽然频率降低了，但仍需要保持较大的增益值来补偿
        # 这里设置一个较大的增益值来补偿 Ku 波段的路径损耗
        gain_sat = np.full(self.M3, 10000.0, dtype=float)  # 40 dBi（保持较大值）
        gain_per_rat = np.concatenate([gain_terrestrial, gain_sat])
        
        # 扩展到所有用户
        Gain = np.tile(gain_per_rat, (self.eMBB_num + self.URLLC_num, 1))

        channel = channel_ * np.sqrt(Gain)


        
        
        return dk_m,channel