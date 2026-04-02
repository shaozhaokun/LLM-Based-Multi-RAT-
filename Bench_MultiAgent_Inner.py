import numpy as np
import random
import pandas as pd

from Scheduling import queue_delay_calculation
from WF import water_filling_power_allocation, satellite_downlink_power_allocation


class MyproblemInner:
    """
    Multi-agent 的 inner（自包含卫星通信模型版本）：
    - outer 固定 association（outer_ass 作为 0/1 mask）
    - inner 优化 allocation（bandwidth）

    兼容外层调用接口：
      run_origin(idv, best_value_matrix, trans_delay_matrix, queue_delay_matrix)
      run(idv, best_value_matrix, trans_delay_matrix, queue_delay_matrix, population=None)
    """

    def __init__(
        self,
        URLLC_num,
        eMBB_num,
        RAT_num,
        seed,
        outer_ass,
        population_inner,
        ch,
        num_list=None,
        RAT_list=None,
        RAT_num_cure=None,
        inner_generation: int = 100,
        inner_population_size: int = 10,
    ):
        self.URLLC_num = int(URLLC_num)
        self.eMBB_num = int(eMBB_num)
        self.K_total = self.URLLC_num + self.eMBB_num

        if RAT_list is None:
            raise ValueError("RAT_list 不能为空（例如 [M1,M2,M3,M3]）。")
        self.RAT_list = np.asarray(RAT_list).astype(int)

        self.RAT_num = int(np.sum(self.RAT_list))                 # 总RAT数 = uplink + downlink sat
        self.RAT_num_up = int(np.sum(self.RAT_list) - self.RAT_list[3])
        self.RAT_num_down = int(self.RAT_list[3])
        self.RAT_num_sat = int(self.RAT_list[2])
        self.RAT_num_cure = int(RAT_num_cure) if RAT_num_cure is not None else int(RAT_num)
        self.RAT_num_terrestrial = int(self.RAT_num_cure - self.RAT_list[2])

        self.seed = int(seed)
        np.random.seed(self.seed)

        self.ch = ch
        if self.ch is None:
            raise ValueError("ch(channel) 不能为空。")

        self.population_size = int(inner_population_size)
        self.generation = int(inner_generation)

        self.chromosome_length = (self.K_total) * self.RAT_num

        # ---- warm-start seed (from outer) ----
        # outer 会把“上一次该 outer 个体对应的 inner 最优解”传进来（1D 向量）
        # 我们用它作为种群第 0 个个体（mask + clip），避免每次从零开始。
        self.population_seed = None
        if population_inner is not None:
            try:
                seed_vec = np.asarray(population_inner, dtype=float).reshape(-1)
                if seed_vec.size == self.chromosome_length:
                    self.population_seed = seed_vec
            except Exception:
                self.population_seed = None

        self.outer_ass_ = np.asarray(outer_ass).astype(int).reshape(-1)
        if self.outer_ass_.size != self.chromosome_length:
            raise ValueError(f"outer_ass 长度不对：期望 {self.chromosome_length}, 实际 {self.outer_ass_.size}")
        self.outer_ass = self.outer_ass_.reshape(1, self.chromosome_length)

        # ---------- bandwidth caps (same as Bench_All_DE) ----------
        self.W_6g = 50 * 1e6
        self.W_wifi = 10 * 1e6

        self.W_6g_ = 3* 1e5
        self.W_wifi_ = 2* 1e5

        self.W_sat_eMBB_up = 3 * 1e5
        self.W_sat_URLLC_up = 3 * 1e5
        self.W_sat_URLLC_down = 3 * 1e5
        self.W_sat_eMBB_down = 3 * 1e5


        self.W_sat_eMBB_up_ = 8 * 1e4
        self.W_sat_URLLC_up_ = 8 * 1e4
        self.W_sat_URLLC_down_ = 8 * 1e4
        self.W_sat_eMBB_down_ = 8 * 1e4

        # URLLC downlink power per satellite
        self.L_sat_URLLC_down = 100.0
        # eMBB downlink total power per satellite
        self.L_sat_eMBB_down_total = 20.0

        # backhaul capacities (uplink RAT only)
        C_6g = 4e7
        C_wifi = 2e7
        C_sat = np.inf
        self.C_vec = np.array(
            [C_6g] * int(self.RAT_list[0]) +
            [C_wifi] * int(self.RAT_list[1]) +
            [C_sat] * int(self.RAT_list[2]),
            dtype=float,
        )

        # ---------- bounds (lb/ub) ----------
        lb_band_URLLC = [0] * self.URLLC_num * self.RAT_num
        ub_band_URLLC = (
            ([self.W_6g_] * int(self.RAT_list[0]) +
             [self.W_wifi_] * int(self.RAT_list[1]) +
             [self.W_sat_URLLC_up_] * int(self.RAT_list[2]) +
             [self.W_sat_URLLC_down_] * int(self.RAT_list[2]))
            * self.URLLC_num
        )

        lb_band_eMBB = [0] * self.eMBB_num * self.RAT_num
        ub_band_eMBB = (
            ([self.W_6g_] * int(self.RAT_list[0]) +
             [self.W_wifi_] * int(self.RAT_list[1]) +
             [self.W_sat_eMBB_up_] * int(self.RAT_list[2]) +
             [self.W_sat_eMBB_down_] * int(self.RAT_list[2]))
            * self.eMBB_num
        )

        self.lb = np.array(lb_band_eMBB + lb_band_URLLC).reshape(1, self.chromosome_length)
        self.ub = np.array(ub_band_eMBB + ub_band_URLLC).reshape(1, self.chromosome_length)

        # ---------- preload tasks (new naming style) ----------
        urllc_df = pd.read_csv(f"Data/urllc_tasks_{self.URLLC_num}.csv")
        embb_df = pd.read_csv(f"Data/embb_tasks_{self.eMBB_num}.csv")

        self.urllc_data_size = urllc_df["Data Size (bits)"].to_numpy()
        self.embb_data_size = embb_df["Data Size (bits)"].to_numpy()
        self.urllc_cpu_cycles = urllc_df["CPU Cycles"].to_numpy()
        self.embb_cpu_cycles = embb_df["CPU Cycles"].to_numpy()
        self.urllc_deadline = urllc_df["Deadline (s)"].to_numpy()
        self.embb_deadline = embb_df["Deadline (s)"].to_numpy()

    def initialize_population_origin(self):
        population_ = (
            (np.dot(np.ones((self.population_size, 1)), (self.ub - self.lb)) * np.random.rand(self.population_size, self.chromosome_length)
             + np.dot(np.ones((self.population_size, 1)), self.lb))
            * (np.dot(np.ones((self.population_size, 1)), self.outer_ass))
        )
        # inject warm-start individual at index 0 (mask + clip)
        if self.population_seed is not None and self.population_size > 0:
            seed = self.population_seed.reshape(1, -1)
            seed = seed * self.outer_ass
            seed = np.minimum(np.maximum(seed, self.lb), self.ub)
            population_[0, :] = seed.reshape(-1)
        return population_

    def mutate(self, population, F=0.01):
        F_matrix_ = F * np.ones((self.population_size, 1))
        F_matrix = F_matrix_ @ np.ones((1, self.chromosome_length))

        a = np.random.permutation(self.population_size)
        b = np.random.permutation(self.population_size)
        c = np.random.permutation(self.population_size)

        population_a = population[a, :]
        population_b = population[b, :]
        population_c = population[c, :]

        donor_matrix_ = population_a + F_matrix * (population_b - population_c)

        H_matrix = donor_matrix_ < np.ones((self.population_size, 1)) @ self.ub
        donor_matrix_1 = donor_matrix_ + 2 * (np.ones((self.population_size, 1)) @ self.ub - donor_matrix_) * (1 - H_matrix)

        Q_matrix = donor_matrix_1 > np.ones((self.population_size, 1)) @ self.lb
        donor_matrix = donor_matrix_1 + 2 * (np.ones((self.population_size, 1)) @ self.lb - donor_matrix_1) * (1 - Q_matrix)

        return donor_matrix

    def crossover(self, population_, mutant_population, CR=0.7):
        trial_population = np.copy(population_)
        CR_matrix_ = CR * np.ones((self.population_size, 1))
        CR_matrix = CR_matrix_ @ np.ones((1, self.chromosome_length))
        C_matrix = np.random.rand(self.population_size, self.chromosome_length) < CR_matrix
        trial_matrix = mutant_population * C_matrix + trial_population * (1 - C_matrix)
        return trial_matrix

    def select(
        self,
        fitness_new,
        population_new,
        CV_pha_new,
        cost_urllc_new,
        trans_new,
        queue_new,
        best_fitness,
        best_population,
        best_CV_pha,
        cost_urllc_pop,
        trans_pop,
        queue_pop,
    ):
        for i in range(self.population_size):
            if CV_pha_new[i] < best_CV_pha[i]:
                best_population[i] = population_new[i]
                best_fitness[i] = fitness_new[i]
                best_CV_pha[i] = CV_pha_new[i]
                cost_urllc_pop[i] = cost_urllc_new[i]
                trans_pop[i] = trans_new[i]
                queue_pop[i] = queue_new[i]
            elif CV_pha_new[i] == best_CV_pha[i] and fitness_new[i] < best_fitness[i]:
                best_population[i] = population_new[i]
                best_fitness[i] = fitness_new[i]
                best_CV_pha[i] = CV_pha_new[i]
                cost_urllc_pop[i] = cost_urllc_new[i]
                trans_pop[i] = trans_new[i]
                queue_pop[i] = queue_new[i]

        return best_fitness, best_population, best_CV_pha, cost_urllc_pop, trans_pop, queue_pop

    def evalVars(self, X):
        Vars = X
        NIND = Vars.shape[0]

        matrix = Vars.reshape(NIND, self.K_total, self.RAT_num)
        W_matrix_up = matrix[:, :, : self.RAT_num_up]
        W_matrix_down = matrix[:, :, self.RAT_num_up :]

        urllc_band_reshape_up = W_matrix_up[:, : self.URLLC_num, :]
        embb_band_reshape_up = W_matrix_up[:, self.URLLC_num :, :]
        urllc_band_reshape_down = W_matrix_down[:, : self.URLLC_num, :]
        embb_band_reshape_down = W_matrix_down[:, self.URLLC_num :, :]

        embb_band_matrix_up = embb_band_reshape_up
        urllc_band_matrix_up = urllc_band_reshape_up
        embb_band_matrix_down = embb_band_reshape_down
        urllc_band_matrix_down = urllc_band_reshape_down

        binary_matrix_embb_up = (embb_band_matrix_up != 0).astype(int)
        binary_matrix_urllc_up = (urllc_band_matrix_up != 0).astype(int)

        cpu_rate_urllc = 5 * 1e9
        cpu_rate_embb = 7 * 1e9
        noise_spectral_density_dbm_hz = -174
        N0 = 10 ** (noise_spectral_density_dbm_hz / 10) * 1e-3
        P_k = 0.2

        urllc_data = np.tile(self.urllc_data_size.reshape(-1, 1), (NIND, 1, 1))
        embb_data = np.tile(self.embb_data_size.reshape(-1, 1), (NIND, 1, 1))

        channel = self.ch
        channel_up = channel[:, : self.RAT_num_up]
        channel_down = channel[:, self.RAT_num_up :]

        URLLC_h_up = np.tile(channel_up[: self.URLLC_num, :], (NIND, 1, 1))
        eMBB_h_up = np.tile(channel_up[self.URLLC_num : self.URLLC_num + self.eMBB_num, :], (NIND, 1, 1))

        # ---------- uplink rates ----------
        embb_power_matrix_up = water_filling_power_allocation(
            embb_band_matrix_up=embb_band_matrix_up,
            eMBB_h_up=eMBB_h_up,
            N0=N0,
            P_k=P_k,
            C_vec=self.C_vec,
        )

        eps = 1e-10

        denom_urllc = N0 * urllc_band_matrix_up
        mask_urllc = urllc_band_matrix_up > 0
        snr_urllc = np.zeros_like(denom_urllc, dtype=float)
        np.divide((np.abs(URLLC_h_up) ** 2) * 0.2, denom_urllc, out=snr_urllc, where=mask_urllc)
        rk_m_urllc = np.zeros_like(urllc_band_matrix_up, dtype=float)
        rk_m_urllc[mask_urllc] = urllc_band_matrix_up[mask_urllc] * np.log2(1.0 + snr_urllc[mask_urllc])
        rk_m_urllc_sum = np.sum(rk_m_urllc, axis=2, keepdims=1)

        denom_embb = N0 * embb_band_matrix_up
        mask_embb = embb_band_matrix_up > 0
        snr_embb = np.zeros_like(denom_embb, dtype=float)
        np.divide((np.abs(eMBB_h_up) ** 2) * embb_power_matrix_up, denom_embb, out=snr_embb, where=mask_embb)
        r_first_embb = np.zeros_like(embb_band_matrix_up, dtype=float)
        r_first_embb[mask_embb] = embb_band_matrix_up[mask_embb] * np.log2(1.0 + snr_embb[mask_embb])

        # ---------- eMBB second hop ----------
        M3 = self.RAT_num_sat
        sat_to_gateway_channel_full = channel[:, self.RAT_num_cure : self.RAT_num_cure + M3]
        sat_to_gateway_channel = sat_to_gateway_channel_full[0, :]
        sat_to_gateway_gain2 = np.abs(sat_to_gateway_channel) ** 2

        C_vec_up = self.C_vec[: self.RAT_num_up]
        r_second_embb = np.zeros_like(r_first_embb)
        r_second_embb[:, :, : self.RAT_num_terrestrial] = C_vec_up[: self.RAT_num_terrestrial]

        C_sat_down = r_first_embb[:, :, self.RAT_num_terrestrial :]
        embb_sat_power_matrix_down = satellite_downlink_power_allocation(
            embb_band_matrix_down=embb_band_matrix_down,
            sat_to_gateway_h=sat_to_gateway_channel,
            N0=N0,
            P_sat_total=self.L_sat_eMBB_down_total,
            cap_rate_matrix=C_sat_down,
        )

        h2_sat = sat_to_gateway_gain2.reshape(1, 1, M3)
        denom_sat = N0 * embb_band_matrix_down
        mask_sat = embb_band_matrix_down > 0
        r_second_embb_sat_snr = np.zeros_like(denom_sat, dtype=float)
        np.divide(embb_sat_power_matrix_down * h2_sat, denom_sat, out=r_second_embb_sat_snr, where=mask_sat)
        r_second_embb_sat = np.zeros_like(embb_band_matrix_down, dtype=float)
        r_second_embb_sat[mask_sat] = embb_band_matrix_down[mask_sat] * np.log2(1.0 + r_second_embb_sat_snr[mask_sat])

        r_second_embb[:, :, self.RAT_num_terrestrial :] = r_second_embb_sat
        r_second_embb = r_second_embb * binary_matrix_embb_up
        R_e2e_embb = np.minimum(r_first_embb, r_second_embb)
        R_e2e_sum = np.sum(R_e2e_embb, axis=2, keepdims=True)
        Communication_delay_eMBB_e2e = embb_data / (R_e2e_sum + eps)
        Communication_delay_eMBB_e2e = Communication_delay_eMBB_e2e.reshape(NIND, -1)

        # ---------- URLLC downlink scheduling (FIFO with full downlink bandwidth/power) ----------
        r_first_urllc_transmission_time = urllc_data / (rk_m_urllc_sum + eps)
        r_first_urllc_transmission_time = r_first_urllc_transmission_time.reshape(NIND, -1)
        rk_m_urllc_down_sum = np.zeros((NIND, self.URLLC_num))

        sat_to_gateway_channel = channel[:, self.RAT_num_cure : self.RAT_num]
        sat_to_gateway_channel = sat_to_gateway_channel[0, :]
        sat_to_gateway_h = np.tile(sat_to_gateway_channel, (NIND, self.URLLC_num, 1))

        terrestrial_band = urllc_band_matrix_up[:, :, : self.RAT_num_terrestrial]
        satellite_up_band = urllc_band_matrix_up[:, :, self.RAT_num_terrestrial :]

        terrestrial_mask = np.any(terrestrial_band > eps, axis=2)
        satellite_mask = np.any(satellite_up_band > eps, axis=2)

        terrestrial_rat_indices = np.argmax(terrestrial_band > eps, axis=2)
        terrestrial_rat_indices = np.where(terrestrial_mask, terrestrial_rat_indices, 0)
        C_backhaul_matrix = self.C_vec[terrestrial_rat_indices]
        rk_m_urllc_down_sum = np.where(terrestrial_mask, C_backhaul_matrix, 0)

        satellite_up_indices = np.argmax(satellite_up_band > eps, axis=2)
        satellite_up_indices = np.where(satellite_mask, satellite_up_indices, 0)

        B_down_u = self.W_sat_URLLC_down
        L_down_u = self.L_sat_URLLC_down
        denominator_down = (N0 * B_down_u) + eps

        i_indices = np.arange(NIND)[:, np.newaxis]
        k_indices = np.arange(self.URLLC_num)[np.newaxis, :]
        sat_indices = np.clip(satellite_up_indices, 0, M3 - 1)
        h_sat_gw_selected = sat_to_gateway_h[i_indices, k_indices, sat_indices]
        rk_sat_down = B_down_u * np.log2(1 + (L_down_u * np.abs(h_sat_gw_selected) ** 2) / denominator_down)
        rk_sat_down = np.where(np.isnan(rk_sat_down), 0, rk_sat_down)
        satellite_only_mask = satellite_mask & (~terrestrial_mask)
        rk_m_urllc_down_sum = np.where(satellite_only_mask, rk_sat_down, rk_m_urllc_down_sum)

        arrival_time_urllc_downlink = r_first_urllc_transmission_time
        max_delay_urllc = 2

        r_second_urllc_transmission_time = np.zeros((NIND, self.URLLC_num))
        downlink_queue_delay_urllc = np.zeros((NIND, self.URLLC_num))

        for i in range(NIND):
            arrival_times = arrival_time_urllc_downlink[i, :]
            downlink_rates = rk_m_urllc_down_sum[i, :]
            data_sizes = urllc_data[i, :, 0]
            task_indices = np.argsort(arrival_times)
            current_time = 0.0
            for idx in task_indices:
                arrival_time = arrival_times[idx]
                data_size = data_sizes[idx]
                downlink_rate = downlink_rates[idx]
                if downlink_rate < eps:
                    r_second_urllc_transmission_time[i, idx] = max_delay_urllc
                    downlink_queue_delay_urllc[i, idx] = max_delay_urllc
                    continue
                transmission_time = data_size / (downlink_rate + eps)
                if current_time < arrival_time:
                    queue_delay = 0.0
                    current_time = arrival_time + transmission_time
                else:
                    queue_delay = current_time - arrival_time
                    current_time = current_time + transmission_time
                r_second_urllc_transmission_time[i, idx] = transmission_time
                downlink_queue_delay_urllc[i, idx] = queue_delay

        Communication_delay_urllc_e2e = r_first_urllc_transmission_time + downlink_queue_delay_urllc + r_second_urllc_transmission_time

        # ---------- computation times + deadlines ----------
        cpu_cycles_urllc_expanded = np.tile(self.urllc_cpu_cycles, (Vars.shape[0], 1))
        computation_time_urllc = cpu_cycles_urllc_expanded / cpu_rate_urllc
        cpu_cycles_embb_expanded = np.tile(self.embb_cpu_cycles, (Vars.shape[0], 1))
        computation_time_embb = cpu_cycles_embb_expanded / cpu_rate_embb

        deadline_urllc = np.tile(self.urllc_deadline, (Vars.shape[0], 1))
        deadline_embb = np.tile(self.embb_deadline, (Vars.shape[0], 1))

        # ---------- constraints (CV) ----------
        M1 = int(self.RAT_list[0])
        M2 = int(self.RAT_list[1])
        M3 = int(self.RAT_list[2])
        CV_terms = []

        for m in range(M1):
            rat_sum = np.sum(embb_band_matrix_up[:, :, [m]], axis=1) + np.sum(urllc_band_matrix_up[:, :, [m]], axis=1)
            CV_terms.append(rat_sum - self.W_6g)
        for m in range(M1, M1 + M2):
            rat_sum = np.sum(embb_band_matrix_up[:, :, [m]], axis=1) + np.sum(urllc_band_matrix_up[:, :, [m]], axis=1)
            CV_terms.append(rat_sum - self.W_wifi)

        sat_up_start = M1 + M2
        for s in range(M3):
            m = sat_up_start + s
            urllc_sum = np.sum(urllc_band_matrix_up[:, :, [m]], axis=1)
            embb_sum = np.sum(embb_band_matrix_up[:, :, [m]], axis=1)
            CV_terms.append(urllc_sum - self.W_sat_URLLC_up)
            CV_terms.append(embb_sum - self.W_sat_eMBB_up)

        for s in range(M3):
            embb_down_sum = np.sum(embb_band_matrix_down[:, :, [s]], axis=1)
            CV_terms.append(embb_down_sum - self.W_sat_eMBB_down)

        CV = np.hstack(CV_terms)
        pha = np.where(CV < 0, 0, CV)
        CV_pha = np.sum(pha, axis=1)

        max_delay_eMBB = 2
        queue_time_embb, queue_time_urllc, trans_time_eMBB, trans_time_URLLC, total_delay_eMBB, total_delay_URLLC = queue_delay_calculation(
            Communication_delay_eMBB_e2e,
            Communication_delay_urllc_e2e,
            computation_time_embb,
            computation_time_urllc,
            deadline_embb,
            deadline_urllc,
            max_delay_urllc,
            max_delay_eMBB,
        )

        cost_embb = np.sum(total_delay_eMBB, axis=1, keepdims=True)
        cost_urllc = np.sum(total_delay_URLLC, axis=1, keepdims=True)

        trans_delay_eMBB = np.sum(trans_time_eMBB, axis=1, keepdims=True)
        trans_delay_URLLC = np.sum(trans_time_URLLC, axis=1, keepdims=True)

        average_Transrate_embb = np.mean(total_delay_eMBB, axis=1, keepdims=True)
        average_Outagerate_urllc = cost_urllc / (self.URLLC_num * max_delay_urllc)

        trans_delay = np.hstack((trans_delay_URLLC, trans_delay_eMBB))
        queue_delay = np.hstack((average_Transrate_embb, average_Outagerate_urllc))

        cost = cost_embb + cost_urllc
        f = cost

        return f, cost_urllc, CV, CV_pha, trans_delay, queue_delay

    def select_based_on_fitness(self, best_population, best_fitness, best_CV_pha, best_cost_urllc, best_trans, best_queue):
        best_CV_pha = best_CV_pha.flatten()
        selected_indices = np.argsort(best_CV_pha)[: self.population_size]
        selected_population = best_population[selected_indices]
        selected_fitness = best_fitness[selected_indices]
        selected_CV_pha = best_CV_pha[selected_indices]
        selected_cost_urllc = best_cost_urllc[selected_indices]
        selected_trans_delay = best_trans[selected_indices]
        selected_queue_delay = best_queue[selected_indices]

        zero_CV_indices = np.where(selected_CV_pha == 0)[0]
        if len(zero_CV_indices) > 0:
            zero_CV_fitness = selected_fitness[zero_CV_indices]
            sorted_indices = np.argsort(zero_CV_fitness.flatten())
            zero_CV_indices_sorted = zero_CV_indices[sorted_indices]
            selected_population[zero_CV_indices] = selected_population[zero_CV_indices_sorted]
            selected_fitness[zero_CV_indices] = selected_fitness[zero_CV_indices_sorted]
            selected_CV_pha[zero_CV_indices] = selected_CV_pha[zero_CV_indices_sorted]
            selected_cost_urllc[zero_CV_indices] = selected_cost_urllc[zero_CV_indices_sorted]
            selected_trans_delay[zero_CV_indices] = selected_trans_delay[zero_CV_indices_sorted]
            selected_queue_delay[zero_CV_indices] = selected_queue_delay[zero_CV_indices_sorted]

        return selected_fitness, selected_population, selected_CV_pha, selected_cost_urllc, selected_trans_delay, selected_queue_delay

    def run_origin(self, idv=None, best_value_matrix=None, trans_delay_matrix=None, queue_delay_matrix=None):
        population_inter = self.initialize_population_origin()
        population = population_inter * (np.dot(np.ones((self.population_size, 1)), self.outer_ass))

        fitness_best = 1e15
        CV_best = 1e30
        population_best = population[0]
        cost_urllc_best = 0.0

        fitness_generation_full = np.zeros((self.generation, 1))

        for gen in range(self.generation):
            donor_population = self.mutate(population)
            trial_population = self.crossover(population, donor_population)

            fitness_pop, cost_urllc_pop, CV_pop, CV_pha_pop, trans_delay_pop, queue_delay_pop = self.evalVars(population)
            fitness_trial, cost_urllc_trail, CV_trial, CV_pha_trial, trans_delay_trial, queue_delay_trial = self.evalVars(trial_population)

            best_fitness, best_population, best_CV_pha, best_cost_urllc, best_trans, best_queue = self.select(
                fitness_trial,
                trial_population,
                CV_pha_trial,
                cost_urllc_trail,
                trans_delay_trial,
                queue_delay_trial,
                fitness_pop,
                population,
                CV_pha_pop,
                cost_urllc_pop,
                trans_delay_pop,
                queue_delay_pop,
            )

            best_fitness, best_population, best_CV_pha, best_cost_urllc, best_trans, best_queue = self.select_based_on_fitness(
                best_population, best_fitness, best_CV_pha, best_cost_urllc, best_trans, best_queue
            )

            population = best_population

            if best_CV_pha[0] < CV_best:
                fitness_best = best_fitness[0]
                population_best = best_population[0]
                CV_best = best_CV_pha[0]
                cost_urllc_best = best_cost_urllc[0]
                trans_best = best_trans[0]
                queue_best = best_queue[0]
            elif best_CV_pha[0] == CV_best and best_fitness[0] < fitness_best:
                fitness_best = best_fitness[0]
                population_best = best_population[0]
                CV_best = best_CV_pha[0]
                cost_urllc_best = best_cost_urllc[0]
                trans_best = best_trans[0]
                queue_best = best_queue[0]

            # 记录可行解曲线：不可行则记一个“大惩罚”，避免出现 0 这种“看起来更优”的假象
            fitness_generation_full[gen] = float(best_fitness[0]) if float(np.asarray(CV_best).item()) == 0.0 else 1e15

            # 可选打印（保持简洁）
            # print(f"Innergeneration{gen}|| CV{float(np.asarray(CV_best).item())} || Cost{float(np.asarray(fitness_best).item())} ||")

        # fill multi-agent matrices (if provided): use recorded curve + final best metrics repeated
        if idv is not None and best_value_matrix is not None:
            best_value_matrix[idv, : self.generation] = fitness_generation_full.reshape(-1)[: self.generation]

        try:
            _, _, _, _, trans_delay, queue_delay = self.evalVars(np.asarray(population_best).reshape(1, -1))
            trans_delay = np.asarray(trans_delay).reshape(-1)
            queue_delay = np.asarray(queue_delay).reshape(-1)
            if idv is not None and trans_delay_matrix is not None:
                trans_delay_matrix[0, idv, : self.generation] = float(trans_delay[0])
                trans_delay_matrix[1, idv, : self.generation] = float(trans_delay[1])
            if idv is not None and queue_delay_matrix is not None:
                queue_delay_matrix[0, idv, : self.generation] = float(queue_delay[0])
                queue_delay_matrix[1, idv, : self.generation] = float(queue_delay[1])
        except Exception:
            pass

        return (
            population_best,
            fitness_best,
            CV_best,
            best_value_matrix,
            cost_urllc_best,
            trans_delay_matrix,
            queue_delay_matrix,
        )

    def run(self, idv, best_value_matrix, trans_delay_matrix, queue_delay_matrix, population=None):
        # outer 可能传 population=None，这里直接走 run_origin
        return self.run_origin(idv, best_value_matrix, trans_delay_matrix, queue_delay_matrix)

