import numpy as np


BW_EPS_HZ = 1.0  # 带宽阈值：小于该值的带宽按 0 处理，避免数值噪声导致 C/W 溢出
MAX_SE_TARGET = 80.0  # 2**60 ≈ 1e18，超过该谱效率时直接视为“cap 需要无穷大功率”


def _water_filling_no_cap(W_row, h_row, N0, P_k):
    """
    对给定的一组通道做「不带上限 P_c」的经典 water-filling，使用闭式解（论文 Equation 30）。
    只考虑非负功率和总功率约束。

    实现方式：
      1) 先假设所有通道都激活，用闭式解算水位 μ；
      2) 检查哪些通道 p_m < 0（即 μ <= N0/|h_m|^2），这些通道应该被关闭；
      3) 排除这些通道，重新用闭式解算 μ；
      4) 重复直到所有激活通道都满足 p_m >= 0。
    """
    abs_h2 = np.abs(h_row) ** 2
    RAT_num = W_row.shape[0]

    if np.all(W_row <= 0) or P_k <= 0:
        return np.zeros_like(W_row, dtype=float)

    # 初始：所有通道都参与
    active_mask = (W_row > BW_EPS_HZ) & (abs_h2 > 0)
    p_result = np.zeros_like(W_row, dtype=float)

    # 迭代：逐步排除 p_m < 0 的通道
    for _ in range(RAT_num):  # 最多迭代 RAT_num 次
        if not np.any(active_mask) or P_k <= 0:
            break

        W_active = W_row[active_mask]
        h_active = h_row[active_mask]
        abs_h2_active = abs_h2[active_mask]

        # 闭式解：μ = (P_k + Σ(W_m N0 / |h_m|^2)) / Σ(W_m)
        # 论文 Equation (30)
        sum_W_N0_over_h2 = np.sum(W_active * N0 / np.maximum(abs_h2_active, 1e-20))
        sum_W = np.sum(W_active)
        
        if sum_W <= 0:
            break

        mu = (P_k + sum_W_N0_over_h2) / sum_W

        # 计算每个激活通道的功率：p_m = W_m * (μ - N0 / |h_m|^2)
        # 论文 Equation (23)
        p_active = W_active * (mu - N0 / np.maximum(abs_h2_active, 1e-20))

        # 检查哪些通道 p_m < 0（即 μ <= N0 / |h_m|^2），这些通道应该被关闭
        negative_mask_active = p_active < -1e-12  # 允许一点数值误差

        if not np.any(negative_mask_active):
            # 所有激活通道都满足 p_m >= 0，这是最终结果
            p_result[active_mask] = np.maximum(p_active, 0.0)  # 确保非负
            break

        # 排除 p_m < 0 的通道，重新计算
        # 把 negative_mask_active 映射回原始的 active_mask 索引
        active_indices = np.where(active_mask)[0]
        negative_indices = active_indices[negative_mask_active]
        active_mask[negative_indices] = False

    return p_result


def _capped_water_filling_single_user(W_row, h_row, N0, P_k, C_vec):
    """
    对单个 eMBB 用户在多个 RAT 之间做「先普通 water-filling，再考虑 P_c 上限」的截断水填充。

    实现方式：
      1）先在所有未冻结通道上做一次不带上限的 water-filling；
      2）如果有通道 p_m > P_m^c，则将这些通道的功率固定为 P_m^c，
          减去对应功率后，在剩余通道上重新做 water-filling；
      3）重复直到所有通道都不超过各自的 P_c。

    这等价于你论文里的「先做 WF，再 clip + 重新分配」的迭代版本。
    """
    RAT_num = W_row.shape[0]

    # 如果该用户在所有 RAT 上都没有带宽，功率全为 0
    if np.all(W_row <= 0) or P_k <= 0:
        return np.zeros_like(W_row)

    abs_h2 = np.abs(h_row) ** 2

    # 计算每个 RAT 的截断功率 P_m^c，使得 R_m(P_m^c) = C_m
    # R_m(p) = W_m * log2(1 + p*|h_m|^2 / (W_m*N0))
    # => P_m^c = (W_m * N0 / |h_m|^2) * (2^{C_m / W_m} - 1)
    P_c = np.zeros_like(W_row, dtype=float)
    for m in range(RAT_num):
        if W_row[m] > BW_EPS_HZ and abs_h2[m] > 0 and C_vec is not None: 
            C_m = C_vec[m]
            # 如果该 RAT 没有回传约束（例如 C_m 为 inf 或 nan），视为无截断, 比如卫星
            if np.isinf(C_m) or np.isnan(C_m):
                P_c[m] = np.inf
            else:
                # se_target = C/W；当 W 很小或 C 很大时可能非常大，直接算 2**se_target 会溢出。
                # 溢出时等价于“需要无穷大功率才能达到 cap”，因此将 P_c 置为 inf（即不对该通道做 clip）。
                se_target = C_m / W_row[m]
                if se_target >= MAX_SE_TARGET:
                    P_c[m] = np.inf
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        snr_target = np.exp2(se_target) - 1.0
                    if not np.isfinite(snr_target):
                        P_c[m] = np.inf
                    else:
                        P_c[m] = (W_row[m] * N0 / abs_h2[m]) * snr_target
        else:
            P_c[m] = 0.0

    # 若所有通道都有有限上限，且 P_c 总和仍 <= P_k，则直接饱和
    finite_P_c = np.where(np.isfinite(P_c), P_c, 0.0)
    if np.all(np.isfinite(P_c)) and finite_P_c.sum() <= P_k:
        return finite_P_c

    # 迭代：先在所有通道上做普通 water-filling，再对超出 P_c 的通道进行 clip，
    #       然后在剩余通道上对剩余功率继续 water-filling，直到无通道超过 P_c。
    p_final = np.zeros_like(W_row, dtype=float)
    remaining_power = P_k
    active_mask = (W_row > BW_EPS_HZ)  # 当前参与 water-filling 的通道

    # 如果存在 P_c = 0 的通道，直接保持为 0
    for _ in range(RAT_num):  # 最多迭代 RAT_num 次就会收敛
        if remaining_power <= 0 or not np.any(active_mask):
            break

        # 只对 active 通道做普通 water-filling
        W_active = W_row[active_mask]
        h_active = h_row[active_mask]
        p_active = _water_filling_no_cap(W_active, h_active, N0, remaining_power)

        # 放回完整向量的位置
        p_temp = np.zeros_like(W_row, dtype=float)
        p_temp[active_mask] = p_active

        # 检查是否有通道超过了各自的 P_c
        over_mask = p_temp > P_c + 1e-12  # 允许一点数值误差
        if not np.any(over_mask):
            # 没有超过上限的，先更新 p_final
            p_final += p_temp
            remaining_power = P_k - p_final.sum()
            
            # 如果还有剩余功率（说明某些通道太差，water-filling 没有分配功率），
            # 不应该 break，而是继续到最后的剩余功率分配逻辑（第 148-153 行）
            # 但需要防止无限循环：如果 p_temp.sum() 很小，说明所有通道都太差，无法再分配
            if remaining_power > 1e-10:
                p_temp_sum = p_temp.sum()
                if p_temp_sum < 1e-12:
                    # water-filling 返回的功率几乎为 0，说明所有通道都太差，无法分配
                    break
                # 否则继续循环，会在最后的剩余功率分配逻辑中处理
            else:
                # 功率已用完，这是最终结果
                break

        # 对超过上限的通道进行 clip：固定为 P_c
        clip_power = np.minimum(p_temp[over_mask], P_c[over_mask])
        p_final[over_mask] += clip_power

        # 减去已固定的功率
        used_power = clip_power.sum()
        remaining_power = P_k - p_final.sum()
        if remaining_power <= 0:
            break

        # 这些已被 clip 的通道不再参与后续 water-filling
        active_mask[over_mask] = False

    # 如果还有剩余功率（例如只有卫星无上限的情况），在剩余 active 通道上再做一次普通 water-filling
    if remaining_power > 0 and np.any(active_mask):
        W_active = W_row[active_mask]
        h_active = h_row[active_mask]
        p_active = _water_filling_no_cap(W_active, h_active, N0, remaining_power)
        p_final[active_mask] += p_active

    return p_final


def water_filling_power_allocation(embb_band_matrix_up, eMBB_h_up, N0, P_k, C_vec=None):
    """
    计算 eMBB 的功率分配。

    - 若 C_vec 为 None，则退化为传统 water-filling（与原代码一致）；
    - 若 C_vec 为长度为 RAT_num 的向量，则对每个 RAT 使用固定 C 做“截断水填充”。

    参数:
        embb_band_matrix: np.ndarray，形状为 (NIND, eMBB_num, RAT_num)，带宽分配矩阵
        eMBB_h:           np.ndarray，形状同上，信道增益
        N0:               float，噪声功率谱密度
        P_k:              float，每个 eMBB 用户的总功率预算
        C_vec:            array-like, 形状为 (RAT_num,)，每个 RAT 的回传容量 C_m（单位需与速率一致）

    返回:
        embb_power_matrix: np.ndarray，形状同 embb_band_matrix，对应功率分配矩阵
    """
    NIND, eMBB_num, RAT_num = embb_band_matrix_up.shape

    # 如果没有提供 C_vec，使用原始 water-filling（无线单跳）
    if C_vec is None:
        embb_band_sumaxis_2 = np.sum(embb_band_matrix_up, axis=2, keepdims=True)
        embb_band_sumaxis = np.sum(
            embb_band_matrix_up * N0 / np.abs(eMBB_h_up) ** 2,
            axis=2,
            keepdims=True,
        )
        miu_ = (P_k + embb_band_sumaxis) / embb_band_sumaxis_2
        miu = np.tile(miu_, (1, 1, RAT_num))
        embb_power_matrix = embb_band_matrix_up * (miu - N0 / np.abs(eMBB_h) ** 2)
        return embb_power_matrix

    # 有固定的每 RAT 回传容量 C_vec，执行截断水填充（按用户、按 RAT）
    C_vec = np.asarray(C_vec, dtype=float).reshape(-1)
    assert C_vec.shape[0] == RAT_num, "C_vec 的长度必须等于 RAT_num"

    embb_power_matrix = np.zeros_like(embb_band_matrix_up, dtype=float)
    for i in range(NIND):
        for k in range(eMBB_num):
            W_row = embb_band_matrix_up[i, k, :]
            h_row = eMBB_h_up[i, k, :]
            p_row = _capped_water_filling_single_user(W_row, h_row, N0, P_k, C_vec)
            embb_power_matrix[i, k, :] = p_row

    return embb_power_matrix


def satellite_downlink_power_allocation(
    embb_band_matrix_down,
    sat_to_gateway_h,
    N0,
    P_sat_total,
    cap_rate_matrix=None,
):
    """
    卫星第二跳（卫星->gateway）eMBB 下行功率分配：
    - 每颗卫星有自己的总功率预算 P_sat_total（标量，表示“每颗卫星”相同；如需不同，可传入长度为 M3 的向量并自行扩展）
    - 在该卫星上，被分配了下行带宽的 eMBB 用户共同分享该卫星的总功率
    - 支持 capped water-filling：cap_rate_matrix 作为“速率上限”（例如第一跳速率 r^{(1)}），避免第二跳过强造成浪费

    参数:
        embb_band_matrix_down: np.ndarray, (NIND, eMBB_num, M3)，每个用户在每颗卫星上的下行带宽 B_{k,m}^{down,e}
        sat_to_gateway_h:      np.ndarray, (M3,)，每颗卫星到 gateway 的复信道（对用户无关，但对卫星不同）
        N0:                    float, 噪声功率谱密度
        P_sat_total:           float, 每颗卫星的 eMBB 下行总功率预算（W）
        cap_rate_matrix:       np.ndarray, (NIND, eMBB_num, M3)，速率上限（bps）。若为 None，则不加 cap

    返回:
        power_down: np.ndarray, (NIND, eMBB_num, M3)，每个用户在每颗卫星上的下行功率分配 L_{k,m}^{down,e}
    """
    embb_band_matrix_down = np.asarray(embb_band_matrix_down, dtype=float)
    sat_to_gateway_h = np.asarray(sat_to_gateway_h)

    NIND, eMBB_num, M3 = embb_band_matrix_down.shape
    assert sat_to_gateway_h.shape[0] == M3, "sat_to_gateway_h 的长度必须等于 M3"

    if cap_rate_matrix is not None:
        cap_rate_matrix = np.asarray(cap_rate_matrix, dtype=float)
        assert cap_rate_matrix.shape == (NIND, eMBB_num, M3), "cap_rate_matrix 形状必须是 (NIND, eMBB_num, M3)"

    power_down = np.zeros_like(embb_band_matrix_down, dtype=float)

    # 对每个样本、每颗卫星：把“用户”当成 water-filling 的“通道”
    for i in range(NIND):
        for s in range(M3):
            B_row = embb_band_matrix_down[i, :, s].copy()  # (eMBB_num,)
            # 将数值噪声的“极小带宽”直接置零，避免 cap/W 溢出与不稳定
            B_row[B_row <= BW_EPS_HZ] = 0.0
            if np.all(B_row <= 0) or P_sat_total <= 0:
                continue

            # 卫星->gateway 信道对所有用户相同：构造长度为 eMBB_num 的 h_row
            h_row = np.full((eMBB_num,), sat_to_gateway_h[s], dtype=complex)

            if cap_rate_matrix is None:
                # 无 cap：退化为普通 WF
                p_row = _water_filling_no_cap(B_row, h_row, N0, P_sat_total)
            else:
                C_row = cap_rate_matrix[i, :, s]
                # 有 cap：调用“单用户多通道”的 capped WF，把用户当通道即可
                p_row = _capped_water_filling_single_user(B_row, h_row, N0, P_sat_total, C_row)

            power_down[i, :, s] = p_row

    return power_down


if __name__ == "__main__":
    """
    简单 main 测试：
    - 1 个 eMBB 用户，3 条 RAT：前两条是地面（有回传容量约束），第 3 条是卫星（无回传约束）；
    - total power = P_k，在三个 RAT 上做截断水填充。
    """
    np.random.seed(0)

    NIND = 1          # 样本个数
    eMBB_num = 1      # eMBB 用户数
    RAT_num = 3       # RAT 数 = 2 地面 + 1 卫星

    # 带宽分配：假设地面 1/2 各 10 MHz，卫星 5 MHz（只是示例）
    W_ground1 = 10e6
    W_ground2 = 10e6
    W_sat = 5e6
    embb_band_matrix = np.array([[[W_ground1, W_ground2, W_sat]]])  # (1,1,3)

    # 信道增益：示例复数增益
    eMBB_h = np.array([[[0.8 + 0.1j, 0.5 + 0.2j, 0.3 + 0.4j]]])  # (1,1,3)

    # 噪声谱密度 & 总功率
    noise_spectral_density_dbm_hz = -174
    N0 = 10 ** (noise_spectral_density_dbm_hz / 10) * 1e-3
    P_k = 0.2  # 总功率

    # 回传容量：前两条地面链路有限 C，卫星无约束（np.inf）
    C_ground1 = 4e6   
    C_ground2 = 2e6    
    C_sat = np.inf      # 卫星：无回传容量约束
    C_vec = np.array([C_ground1, C_ground2, C_sat])

    power = water_filling_power_allocation(
        embb_band_matrix=embb_band_matrix,
        eMBB_h=eMBB_h,
        N0=N0,
        P_k=P_k,
        C_vec=C_vec,
    )

    print("embb_band_matrix (W):", embb_band_matrix)
    print("eMBB_h:", eMBB_h)
    print("C_vec:", C_vec)
    print("分配得到的功率矩阵 power:", power)
    print("总功率和 =", power.sum())
