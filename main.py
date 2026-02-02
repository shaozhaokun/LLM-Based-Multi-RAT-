import numpy as np
import random  
import pandas as pd
from Scheduling import queue_delay_calculation

from Position_channel_gen import RATDistanceCalculator
from WF import water_filling_power_allocation



class MyproblemInner:
    def __init__(self, URLLC_num, eMBB_num, RAT_num_cure, seed, outer_ass,ch,num_list,RAT_list):
        self.URLLC_num = URLLC_num
        self.eMBB_num = eMBB_num
        self.RAT_num_cure = RAT_num_cure
        self.RAT_num = sum(RAT_list)                 # 总的RAT数量      2,2,2,2    -> 8
        self.RAT_num_up = sum(RAT_list) - RAT_list[3]  # 上行的RAT数量  2,2,2      -> 6
        self.RAT_num_down = RAT_list[3]                 # 下行的RAT数量       2    ->2
        self.RAT_num_sat = RAT_list[2]
        self.RAT_num_terrestrial = self.RAT_num_cure - RAT_list[2]

        self.seed = seed
        self.outer_ass_ = outer_ass  #  (,D) 
        self.ch = ch   # channel ((eMBB+URLLC),RAT_num)
        self.population_size = 10  # inner individual
        self.generation = 100        # inner generation
        self.embb = eMBB_num * RAT_num
        self.num_list = num_list   # [k1_u,k2_u,k3_u,k1_e,k2_e,k3_e]
        self.RAT_list = RAT_list   # [6G_BSs_num,Wi-Fi_BSs_num,Satellite_BSs_num]




        self.chromosome_length = (self.eMBB_num +self.URLLC_num) * self.RAT_num     # （K_u + K_e）* M     
        self.outer_ass = self.outer_ass_.reshape(1,self.chromosome_length)
        self.outer_ass_reshape = self.outer_ass.reshape(-1,self.RAT_num) 

        self.W_6g = 50 * 1e6     # 50 MHz
        self.W_wifi = 10 * 1e6     # 10 MHz
        self.W_sat_Up = 30 * 1e6     # 30 MHz   
        self.W_sat_Down = 30 * 1e6     # 30 MHz 卫星上行和下行的带宽是分开的


        
        self.W_6g_ = 4 * 1e5   
        self.W_wifi_ = 1.5 * 1e5    

        self.W_sat_eMBB_up = 3* 1e5    
        self.W_sat_URLLC_up = 3* 1e5    
        self.W_sat_URLLC_down = 3* 1e5     # urllc 和 embb 进行分开分配
        self.W_sat_eMBB_down = 3* 1e5
        
        # URLLC下行功率 (每个卫星BS的URLLC下行传输功率，单位：W)
        self.L_sat_URLLC_down = 100.0  # 1 W，可根据实际模型调整    
        # eMBB 下行总功率（每颗卫星的总功率预算，单位：W）
        self.L_sat_eMBB_down_total = 20.0

        # 根据 RAT 索引设置回传容量 C_vec（单位：bit/s）
        # RAT 0, 1: 6G BSs (M1)
        # RAT 2, 3: Wi-Fi BSs (M2)
        # RAT 4, 5: 卫星 BSs (M3) - 无回传约束
        C_6g = 4e7  # 6G 回传容量，单位：bit/s（可根据实际模型调整）
        C_wifi = 2e6  # Wi-Fi 回传容量，单位：bit/s
        C_sat = np.inf  # 卫星：无回传容量约束
        self.C_vec = np.array([C_6g, C_6g, C_wifi, C_wifi, C_sat, C_sat])  # 对应 RAT 0,1,2,3,4,5

        lb_band_URLLC = [0]* self.URLLC_num * self.RAT_num
        ub_band_URLLC = ([self.W_6g_] * self.RAT_list[0] + [self.W_wifi_] * self.RAT_list[1] +
                   [self.W_sat_URLLC_up] * self.RAT_list[2]+[self.W_sat_URLLC_down] * self.RAT_list[2]) * self.URLLC_num

        # 下行用全部带宽的

        lb_band_eMBB = [0] *self.eMBB_num * self.RAT_num                     #K_e * M
        ub_band_eMBB = ([self.W_6g_] * self.RAT_list[0] + [self.W_wifi_] * self.RAT_list[1] +
                      [self.W_sat_eMBB_up] * self.RAT_list[2]+[self.W_sat_eMBB_down] * self.RAT_list[2]) * self.eMBB_num

        self.lb =     np.array( lb_band_eMBB + lb_band_URLLC).reshape(1,self.chromosome_length)
        self.ub =    np.array(ub_band_eMBB + ub_band_URLLC ).reshape(1,self.chromosome_length)  # 1 X D
        
        # 预加载任务数据
        urllc_df = pd.read_csv("Data/urllc_tasks_{}.csv".format(self.URLLC_num))
        embb_df = pd.read_csv("Data/embb_tasks_{}.csv".format(self.eMBB_num))
        
        # 存储为实例变量，供evalVars使用
        self.urllc_data_size = urllc_df['Data Size (bits)'].to_numpy()  # (URLLC_num,)
        self.embb_data_size = embb_df['Data Size (bits)'].to_numpy()    # (eMBB_num,)
        self.urllc_cpu_cycles = urllc_df['CPU Cycles'].to_numpy()       # (URLLC_num,)
        self.embb_cpu_cycles = embb_df['CPU Cycles'].to_numpy()         # (eMBB_num,)
        self.urllc_deadline = urllc_df['Deadline (s)'].to_numpy()        # (URLLC_num,)
        self.embb_deadline = embb_df['Deadline (s)'].to_numpy()         # (eMBB_num,)

    
    def initialize_population_origin(self):
    

        population_ = (np.dot(np.ones((self.population_size,1)) , (self.ub-self.lb)) * np.random.rand(self.population_size,
                                         self.chromosome_length) + np.dot(np.ones((self.population_size,1)),self.lb)) * (np.dot(np.ones((self.population_size,1)) , self.outer_ass ))

        return population_
    
    

    def mutate(self, population, F=0.8):
        F_matrix_ = F * np.ones((self.population_size,1))    # N_2 x 1
        F_matrix =  F_matrix_ @ np.ones((1, self.chromosome_length))  # N_2 x D
        

        # 生成随机排列  reshuffling
        a =  np.random.permutation(self.population_size)
        b =  np.random.permutation(self.population_size)
        c =  np.random.permutation(self.population_size)
        mutant_population = np.zeros_like(population)

        population_a = population[a,:]
        population_b = population[b,:]
        population_c = population[c,:]


        # donor matrix 
        donor_matrix_ = population_a + F_matrix * (population_b - population_c)  # N_2 x D


        # boundary checking
        H_matrix = donor_matrix_ < np.ones((self.population_size,1)) @ self.ub 
       
            # 反射法
        donor_matrix_1 = donor_matrix_  + 2*(np.ones((self.population_size,1)) @ self.ub - donor_matrix_) * (1-H_matrix)  
        
        Q_matrix = donor_matrix_1 > np.ones((self.population_size,1)) @ self.lb 
        donor_matrix = donor_matrix_1 + 2*(np.ones((self.population_size,1)) @ self.lb - donor_matrix_1) * (1-Q_matrix)

            # 非反射法，直接返回到边界值

        # donor_matrix = donor_matrix_ * H_matrix + np.ones((self.population_size,1)) @ self.ub * (1-H_matrix)
        # donor_matrix = donor_matrix_ * Q_matrix + np.ones((self.population_size,1)) @ self.lb * (1-Q_matrix)

        return donor_matrix

    
    def crossover(self, population_, mutant_population, CR=0.7):
        
        trial_population = np.copy(population_)  


        CR_matrix_ = CR * np.ones((self.population_size,1))    # N_2 x 1
        CR_matrix = CR_matrix_ @ np.ones((1,self.chromosome_length)) 

        C_matrix = np.random.rand(self.population_size,self.chromosome_length) < CR_matrix

        trial_matrix = mutant_population * C_matrix + trial_population * (1-C_matrix)


        return trial_matrix
    

    

    def select(self, fitness_new, population_new,CV_pha_new,cost_urllc_new,trans_new,queue_new,best_fitness,best_population,best_CV_pha,cost_urllc_pop,trans_pop,queue_pop):


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



        return best_fitness, best_population,  best_CV_pha,cost_urllc_pop, trans_pop, queue_pop


    def evalVars(self, X):

        Vars = X  # 获取决策变量矩阵
        NIND = Vars.shape[0]
        
        matrix = Vars.reshape(NIND,self.eMBB_num+self.URLLC_num,self.RAT_num) # NIND x (eMBB_num+URLLC_num) x RAT_num
        W_matrix_up = matrix[:,:,:self.RAT_num_up] # NIND x (eMBB_num+URLLC_num) x RAT_num_up
        W_matrix_down = matrix[:,:,self.RAT_num_up:] # NIND x (eMBB_num+URLLC_num) x RAT_num_down
        
       
    
        urllc_band_reshape_up = W_matrix_up[:,:self.URLLC_num,:]  # NIND x URLLC_num x RAT_num_up
        embb_band_reshape_up  = W_matrix_up[:,self.URLLC_num:,:]  # NIND x eMBB_num x RAT_num_up
        urllc_band_reshape_down = W_matrix_down[:,:self.URLLC_num,:]  # NIND x URLLC_num x RAT_num_down
        embb_band_reshape_down = W_matrix_down[:,self.URLLC_num:,:]  # NIND x eMBB_num x RAT_num_down


        embb_band_matrix_up = embb_band_reshape_up       
        urllc_band_matrix_up = urllc_band_reshape_up    
        embb_band_matrix_down = embb_band_reshape_down     
        urllc_band_matrix_down = urllc_band_reshape_down    

        binary_matrix_embb_up =  (embb_band_matrix_up != 0).astype(int)
        binary_matrix_urllc_up =  (urllc_band_matrix_up != 0).astype(int)
        binary_matrix_embb_down =  (embb_band_matrix_down != 0).astype(int)
        binary_matrix_urllc_down =  (urllc_band_matrix_down != 0).astype(int)


        
        cpu_rate_urllc = 5*1e9
        cpu_rate_embb = 7*1e9
        rho = 3
        noise_spectral_density_dbm_hz = -174
        N0 = 10 ** (noise_spectral_density_dbm_hz / 10) * 1e-3
        P_k = 0.2 # wt

        channel = self.ch
        channel_up = channel[:,:self.RAT_num_up]
        channel_down = channel[:,self.RAT_num_up:]

        URLLC_h_up = channel_up[:self.URLLC_num,:]
        eMBB_h_up =  channel_up[self.URLLC_num:self.URLLC_num+self.eMBB_num,:]
        URLLC_h_down = channel_down[:self.URLLC_num,:]
        eMBB_h_down =  channel_down[self.URLLC_num:self.URLLC_num+self.eMBB_num,:]


        URLLC_h_up = np.tile(URLLC_h_up, (NIND, 1, 1))
        eMBB_h_up = np.tile(eMBB_h_up, (NIND, 1, 1))
        URLLC_h_down = np.tile(URLLC_h_down, (NIND, 1, 1))
        eMBB_h_down = np.tile(eMBB_h_down, (NIND, 1, 1))




        # *************** uplink ***************

        # 计算 eMBB 功率分配情况（water-filling）在uplink上，提取到独立模块 WF.py
        embb_power_matrix_up = water_filling_power_allocation(
            embb_band_matrix_up=embb_band_matrix_up,
            eMBB_h_up=eMBB_h_up,
            N0=N0,
             P_k=P_k,
            C_vec=self.C_vec,
        )

        eps = 1e-10
        denominator_urllc = ( N0 * urllc_band_matrix_up) + eps
        rk_m_urllc_ = urllc_band_matrix_up * np.log2(1 + (abs(URLLC_h_up)**2 * 0.2) / denominator_urllc)  # NIND x embb_num x rat num
        rk_m_urllc = np.where(np.isnan(rk_m_urllc_), 0, rk_m_urllc_)  # 把nan值变成0
        rk_m_urllc_sum = np.sum(rk_m_urllc,axis=2,keepdims=1) # (NIND,1,1)


        denominator_embb = ( N0 * embb_band_matrix_up) + eps
        rk_m_embb_ = embb_band_matrix_up  * np.log2(1 + (abs(eMBB_h_up)**2 * embb_power_matrix_up) / denominator_embb)

        rk_m_embb = np.where(np.isnan(rk_m_embb_), 0, rk_m_embb_)  # 把nan值变成0

        # SE check
        SE_urllc_ = np.sum(np.log2(1 + (abs(URLLC_h_up)**2 * 0.2) / denominator_urllc)*binary_matrix_urllc_up,axis=2)
        SE_urllc_test = np.mean(np.log2(1 + (abs(URLLC_h_up)**2 * 0.2) / denominator_urllc)*binary_matrix_urllc_up,axis=1)
        SE_urllc = np.mean(SE_urllc_,axis=1)

        SE_eMBB_ = np.sum(np.log2(1 + (abs(eMBB_h_up)**2 * embb_power_matrix_up) / denominator_embb)*binary_matrix_embb_up,axis=2)
        SE_eMBB_test = np.mean(np.log2(1 + (abs(eMBB_h_up)**2 * embb_power_matrix_up) / denominator_embb)*binary_matrix_embb_up,axis=1)
        SE_eMBB = np.mean(SE_eMBB_,axis=1)

                                                       


        # 使用预加载的任务数据（已在__init__中读取）
        urllc_data = self.urllc_data_size.reshape(-1,1)
        urllc_data = np.tile(urllc_data,(NIND,1,1)) #(NIND,URLLC_num,1)

        embb_data = self.embb_data_size.reshape(-1,1)
        embb_data = np.tile(embb_data,(NIND,1,1)) #(NIND,eMBB_num,1)

 

        # 计算URLLC的传输时间（单跳等效）
        transmission_time_urllc = urllc_data/(rk_m_urllc_sum+eps)          # (NIND,self.urllc_num,1)
        transmission_time_urllc = transmission_time_urllc.reshape(NIND,-1) # (NIND,self.urllc_num)

        # ========= 计算 eMBB 的两跳有效速率 R_{k,m}^{[e2e]} 并得到传输时间 =========
        # 第一跳速率：rk_m_embb 已经计算好 (NIND, eMBB_num, RAT_num_up)
        # 第二跳速率：
        # - terrestrial 路径：由回传容量 C_vec 决定（固定）
        # - satellite 路径：在每颗卫星的总功率约束下，对接入该卫星的 eMBB 用户做功率分配，再计算下行 data rate

        # 提取卫星->gateway 下行信道（在 Position_channel_gen 中已追加到 channel 的最后 M3 列）
        M3 = self.RAT_num_sat  # 卫星数量
        sat_to_gateway_channel_full = channel[:, self.RAT_num_cure:self.RAT_num_cure + M3]  # (num_users, M3)
        # 卫星->gateway 信道只与卫星和gateway位置有关，与用户无关；每一行相同，取第一行为 (M3,)
        sat_to_gateway_channel = sat_to_gateway_channel_full[0, :]  # (M3,)
        sat_to_gateway_gain2 = np.abs(sat_to_gateway_channel) ** 2  # |h_{sat->gw}|^2

        # 构造第二跳速率矩阵（与 rk_m_embb 同形状）
        C_vec_up = self.C_vec[:self.RAT_num_up]  # 对应上行 RAT 的回传容量，形状 (RAT_num_up,)
        r_second_embb = np.zeros_like(rk_m_embb)
        

        for m_idx in range(self.RAT_num_up):
            C_m = C_vec_up[m_idx]
            # terrestrial 上行 RAT：第二跳为有线回传，速率固定为 C_m
            if np.isfinite(C_m):
                r_second_embb[:, :, m_idx] = C_m
            else:
                # satellite 上行 RAT：索引从 self.RAT_num_terrestrial 开始
                sat_idx = m_idx - self.RAT_num_terrestrial  # 对应的卫星索引 0..M3-1
                if sat_idx < 0 or sat_idx >= M3:
                    continue

                h2_sat_gw = sat_to_gateway_gain2[sat_idx]  # 该卫星的 |h_{sat->gw}|^2，标量
                # 对应卫星下行的带宽分配：NIND x eMBB_num
                B_down_mat = embb_band_matrix_down[:, :, sat_idx]  # (NIND, eMBB_num)

                # 每个个体样本 i 在该卫星上的总带宽：shape (NIND, 1)
                sum_B = np.sum(B_down_mat, axis=1, keepdims=True)

                # 等功率谱密度下的 SNR（对该卫星的所有用户相同）：shape (NIND, 1)
                snr_const = (self.L_sat_eMBB_down_total * h2_sat_gw) / (sum_B * N0 + eps)
                # 没有任何用户使用该卫星时（sum_B=0），SNR 置为 0
                snr_const = np.where(sum_B > eps, snr_const, 0.0)

                # 下行速率：B_k * log2(1 + SNR_const)，广播到 (NIND, eMBB_num)
                r_second_embb[:, :, m_idx] = B_down_mat * np.log2(1.0 + snr_const)

        # 两跳有效速率 R_{k,m}^{[e2e]} = min(r^{(1)}, r^{(2)})
        R_e2e_embb = np.minimum(rk_m_embb, r_second_embb)

        # 基于 Lemma 1：对于给定的 {R_{k,m}^{[e2e]}}, eMBB 的最优传输时间为
        #   τ_k = α_k / Σ_m R_{k,m}^{[e2e]}
        # 代码层面：直接用 Σ_m R_{k,m}^{[e2e]} 替换原来的一跳 Σ_m r_{k,m}
        R_e2e_sum = np.sum(R_e2e_embb, axis=2, keepdims=True)  # NIND x eMBB_num x 1
        transmission_time_embb = embb_data / (R_e2e_sum + eps)   # NIND x eMBB_num x 1
        transmission_time_embb = transmission_time_embb.reshape(NIND,-1)               # NIND x embb_num


       
        # 计算每个任务的计算时间（ms）
        # 使用预加载的数据
        cpu_cycles_urllc_expanded = np.tile(self.urllc_cpu_cycles, (Vars.shape[0], 1))  # (NIND, URLLC_num)
        computation_time_urllc = cpu_cycles_urllc_expanded / cpu_rate_urllc  # (NIND, URLLC_num)

        cpu_cycles_embb_expanded = np.tile(self.embb_cpu_cycles, (Vars.shape[0], 1))  # (NIND, eMBB_num)
        computation_time_embb = cpu_cycles_embb_expanded / cpu_rate_embb  # (NIND, eMBB_num)

        # URLLC 和 eMBB 的 deadline 时间
        deadline_urllc = np.tile(self.urllc_deadline, (Vars.shape[0], 1))   # (NIND, URLLC_num)
        deadline_embb = np.tile(self.embb_deadline, (Vars.shape[0], 1))   # (NIND, eMBB_num)





        # *************** downlink ***************

        # URLLC 是 用全部URLLC download 带宽和power进行传输 ，但是需要进行scheduling，因为可能会有多个URLLC任务到达，到达时间不同
        
        # ========== 步骤1: 计算URLLC下行传输速率 ==========
        # URLLC下行传输分为两种情况：
        # 1. Terrestrial BSs (6G和Wi-Fi): 使用有线回传，下行速率 = C_m (回传容量)
        # 2. Satellite BSs: 使用无线链路，下行速率 = B_m^{down,u} * log2(1 + L_m^{down,u} * |h_{sat->gw}|^2 / (B_m^{down,u} * N0))
        
        # 首先，需要识别每个URLLC任务关联到哪些RAT（通过上行关联判断）
        # urllc_band_matrix_up: NIND x URLLC_num x RAT_num_up
        # RAT索引：0,1=6G; 2,3=WiFi; 4,5=Satellite上行
        
        # 初始化下行速率矩阵（每个任务一个速率值）
        rk_m_urllc_down_sum = np.zeros((NIND, self.URLLC_num))
        
        # 提取卫星到gateway的信道（在channel矩阵的后M3列，对所有用户都相同）
        # channel矩阵结构: (num_users, RAT_num + M3)
        # 前RAT_num列：用户到各RAT的信道
        # 后M3列：卫星到gateway的信道（对所有用户都相同，因为gateway位置固定）
        M3 = self.RAT_list[2]  # 卫星BS数量
        sat_to_gateway_channel = channel[:, self.RAT_num_cure:self.RAT_num]  # (num_users, M3)
        # 由于对所有用户都相同，取第一行即可
        sat_to_gateway_channel = sat_to_gateway_channel[0, :]  # (M3,)
        # 扩展为 (NIND, URLLC_num, M3) 的形状
        sat_to_gateway_h = np.tile(sat_to_gateway_channel, (NIND, self.URLLC_num, 1))  # (NIND, URLLC_num, M3)
        
        # 向量化处理：对于每个URLLC任务，判断其关联的RAT类型并计算下行速率
        # 分离terrestrial和satellite的关联
        terrestrial_band = urllc_band_matrix_up[:, :, :self.RAT_num_terrestrial]  # NIND x URLLC_num x 4 (6G和WiFi)
        satellite_up_band = urllc_band_matrix_up[:, :, self.RAT_num_terrestrial:]  # NIND x URLLC_num x 2 (卫星上行)
        
        # 检查每个任务是否关联到terrestrial RATs
        terrestrial_mask = np.any(terrestrial_band > eps, axis=2)  # NIND x URLLC_num
        # 检查每个任务是否关联到satellite RATs
        satellite_mask = np.any(satellite_up_band > eps, axis=2)  # NIND x URLLC_num
        
        # 对于terrestrial关联的任务，使用有线回传容量
        # 找到每个任务关联的第一个terrestrial RAT
        terrestrial_rat_indices = np.argmax(terrestrial_band > eps, axis=2)  # NIND x URLLC_num
        # 对于没有关联terrestrial的任务，索引可能不正确，需要mask处理
        terrestrial_rat_indices = np.where(terrestrial_mask, terrestrial_rat_indices, 0)
        # 获取对应的回传容量
        C_backhaul_matrix = self.C_vec[terrestrial_rat_indices]  # NIND x URLLC_num
        # 只对关联terrestrial的任务应用回传容量
        rk_m_urllc_down_sum = np.where(terrestrial_mask, C_backhaul_matrix, 0)
        
        # 对于satellite关联的任务，使用无线链路计算
        # 找到每个任务关联的第一个satellite RAT（在卫星上行中的索引）
        satellite_up_indices = np.argmax(satellite_up_band > eps, axis=2)  # NIND x URLLC_num
        # 对于没有关联satellite的任务，索引可能不正确，需要mask处理
        satellite_up_indices = np.where(satellite_mask, satellite_up_indices, 0)
        
        # 获取该卫星BS的URLLC下行带宽和功率
        B_down_u = self.W_sat_URLLC_down  # 全部URLLC下行带宽
        L_down_u = self.L_sat_URLLC_down  # 全部URLLC下行功率
        denominator_down = (N0 * B_down_u) + eps
        
        # 使用卫星到gateway的信道计算下行速率
        # 需要根据satellite_up_indices选择对应的卫星到gateway信道
        # 创建索引数组来选择正确的卫星信道
        i_indices = np.arange(NIND)[:, np.newaxis]  # NIND x 1
        k_indices = np.arange(self.URLLC_num)[np.newaxis, :]  # 1 x URLLC_num
        sat_indices = np.clip(satellite_up_indices, 0, M3 - 1)  # 确保索引在有效范围内
        
        # 选择对应的卫星到gateway信道
        h_sat_gw_selected = sat_to_gateway_h[i_indices, k_indices, sat_indices]  # NIND x URLLC_num
        
        # 计算卫星下行速率
        rk_sat_down = B_down_u * np.log2(1 + (L_down_u * abs(h_sat_gw_selected)**2) / denominator_down)
        rk_sat_down = np.where(np.isnan(rk_sat_down), 0, rk_sat_down)
        
        # 对于关联satellite但不关联terrestrial的任务，使用卫星下行速率
        # 如果同时关联terrestrial和satellite，优先使用terrestrial（有线回传更稳定）
        satellite_only_mask = satellite_mask & (~terrestrial_mask)
        rk_m_urllc_down_sum = np.where(satellite_only_mask, rk_sat_down, rk_m_urllc_down_sum)
        
        # ========== 步骤2: 确定任务到达时间 ==========
        # 到达时间 = 上行传输完成时间（计算在gateway，所以到达时间就是上行传输完成时间）
        arrival_time_urllc_downlink = transmission_time_urllc  # NIND x URLLC_num
        
        # ========== 步骤3: 实现下行FIFO调度 ==========
        # 对每个个体（NIND），按到达时间对URLLC任务进行排序，然后依次传输
        # 每个任务使用全部下行带宽和功率，所以需要串行传输
        
        # 定义最大延迟（用于处理失败情况）
        max_delay_urllc = 2  # 与后面的定义保持一致
        
        downlink_transmission_time_urllc = np.zeros((NIND, self.URLLC_num))
        downlink_queue_delay_urllc = np.zeros((NIND, self.URLLC_num))
        
        for i in range(NIND):
            # 获取该个体的任务信息
            arrival_times = arrival_time_urllc_downlink[i, :]  # URLLC_num
            downlink_rates = rk_m_urllc_down_sum[i, :]  # URLLC_num
            data_sizes = urllc_data[i, :, 0]  # URLLC_num
            
            # 创建任务索引列表，按到达时间排序
            task_indices = np.argsort(arrival_times)
            
            # FIFO调度：按到达时间顺序依次传输
            current_time = 0.0  # 当前时间
            
            for idx in task_indices:
                arrival_time = arrival_times[idx]
                data_size = data_sizes[idx]
                downlink_rate = downlink_rates[idx]
                
                # 如果任务没有关联卫星BS（下行速率为0），跳过或标记为失败
                if downlink_rate < eps:
                    downlink_transmission_time_urllc[i, idx] = max_delay_urllc
                    downlink_queue_delay_urllc[i, idx] = max_delay_urllc
                    continue
                
                # 计算下行传输时间
                transmission_time = data_size / (downlink_rate + eps)
                
                # 计算队列延迟（等待时间）
                # 如果当前时间 < 到达时间，任务到达后立即传输（无队列延迟）
                # 如果当前时间 >= 到达时间，任务需要等待（队列延迟 = current_time - arrival_time）
                if current_time < arrival_time:
                    queue_delay = 0.0
                    current_time = arrival_time + transmission_time
                else:
                    queue_delay = current_time - arrival_time
                    current_time = current_time + transmission_time
                
                downlink_transmission_time_urllc[i, idx] = transmission_time
                downlink_queue_delay_urllc[i, idx] = queue_delay
        
        # URLLC端到端总延迟 = 上行传输时间 + 下行队列延迟 + 下行传输时间
        # 注意：计算在gateway，所以没有单独的计算时间
        total_delay_urllc_e2e = transmission_time_urllc + downlink_queue_delay_urllc + downlink_transmission_time_urllc



      #该做embb下行传输速率计算



                
        # 带宽约束项 check
        RAT_5G = np.sum(embb_band_matrix_up[:,:,[0]],axis=1) + np.sum(urllc_band_matrix_up[:,:,[0]],axis=1)
        RAT_4G = np.sum(embb_band_matrix_up[:,:,[1]],axis=1) + np.sum(urllc_band_matrix_up[:,:,[1]],axis=1)


        
        # 采用可行性法则处理约束
        CV = np.hstack(
            [
              RAT_5G - self.W_5g,
              RAT_4G - self.W_4g,
            ])
        
        
        
        pha = CV
        pha = np.where(CV < 0, 0, CV)

        CV_pha = np.sum(pha,axis=1)   # NIND x 1

        # ------------------------------------------------------------------------------------------------------------
        max_delay_urllc = 2
        max_delay_eMBB  = 2
        queue_time_embb,queue_time_urllc,trans_time_eMBB,trans_time_URLLC,total_delay_eMBB,total_delay_URLLC= queue_delay_calculation (transmission_time_embb,transmission_time_urllc,
                                                                    computation_time_embb,computation_time_urllc,
                                                                    deadline_embb,deadline_urllc,
                                                                    max_delay_urllc,max_delay_eMBB)
        

        cost_embb = np.sum(total_delay_eMBB, axis=1, keepdims=True)
        cost_urllc = np.sum(total_delay_URLLC, axis=1, keepdims=True)

        trans_delay_eMBB = np.sum(trans_time_eMBB, axis=1, keepdims=True)
        trans_delay_URLLC = np.sum(trans_time_URLLC, axis=1, keepdims=True)

                # eMBB average rate and URLLC outage rate
        average_Transrate_embb = np.mean(total_delay_eMBB, axis=1, keepdims=True)
        average_Outagerate_urllc = cost_urllc/(self.URLLC_num*max_delay_urllc)

        trans_delay = np.hstack((trans_delay_URLLC,trans_delay_eMBB)) 
        queue_delay = np.hstack((average_Transrate_embb,average_Outagerate_urllc)) 
        Cost = np.hstack((cost_urllc,cost_embb))
        Rate = np.hstack((average_Transrate_embb,average_Outagerate_urllc))


        cost = cost_embb + cost_urllc
        
        f = cost

        return f, cost_urllc, CV, CV_pha,trans_delay ,queue_delay
    
    
    def select_based_on_fitness(self,  best_population, best_fitness, best_CV_pha,best_cost_urllc,best_trans,best_queue):
        # 基于适应度选择种群，
        # 排序
        best_CV_pha = best_CV_pha.flatten()
        selected_indices = np.argsort(best_CV_pha)[:self.population_size]
        selected_population = best_population[selected_indices]
        selected_fitness = best_fitness[selected_indices]  # 获取对应的适应度
        selected_CV_pha = best_CV_pha[selected_indices]  # 获取对应的适应度
        selected_cost_urllc  = best_cost_urllc[selected_indices]

        selected_trans_delay = best_trans[selected_indices]
        selected_queue_delay = best_queue[selected_indices]



        # 对于selected_CV_pha[j]是0的个体，根据selected_fitness再排序
        zero_CV_indices = np.where(selected_CV_pha == 0)[0]
        if len(zero_CV_indices) > 0:
            # 获取CV_pha为0的个体的fitness和索引
            zero_CV_fitness = selected_fitness[zero_CV_indices]
            # 对这些个体的fitness进行排序
            sorted_indices = np.argsort(zero_CV_fitness.flatten())
            # 重新排列这些个体
            zero_CV_indices_sorted = zero_CV_indices[sorted_indices]
            # 更新selected数组
            selected_population[zero_CV_indices] = selected_population[zero_CV_indices_sorted]
            selected_fitness[zero_CV_indices] = selected_fitness[zero_CV_indices_sorted]
            selected_CV_pha[zero_CV_indices] = selected_CV_pha[zero_CV_indices_sorted]
            selected_cost_urllc[zero_CV_indices]  = selected_cost_urllc[zero_CV_indices_sorted]

            selected_trans_delay[zero_CV_indices] = selected_trans_delay[zero_CV_indices_sorted]
            selected_queue_delay[zero_CV_indices] = selected_queue_delay[zero_CV_indices_sorted]



        return selected_fitness,selected_population,selected_CV_pha,selected_cost_urllc,selected_trans_delay,selected_queue_delay
    

    def apply_bounds(self, individual):
    # 应用边界约束 - 使用反射法，并处理连续反射问题
        for idx in range(len(individual)):
            loop_count = 0  # 添加一个循环计数器
            max_attempts = 10  # 设置最大尝试次数
            while True:
                if individual[idx] < self.lb[idx]:
                    individual[idx] = self.lb[idx] + (self.lb[idx] - individual[idx])
                elif individual[idx] > self.ub[idx]:
                    individual[idx] = self.ub[idx] - (individual[idx] - self.ub[idx])
                else:
                    break
                loop_count += 1
                if loop_count >= max_attempts:
                    # 如果达到最大尝试次数，直接设置为最近的边界值
                    if individual[idx] < self.lb[idx]:
                        individual[idx] = self.lb[idx]
                    elif individual[idx] > self.ub[idx]:
                        individual[idx] = self.ub[idx]
                    break
        return individual
        
    
    def run_origin(self):

        population_inter = self.initialize_population_origin()            #  Big W_n  eq(51)  # NIND x D
        population = population_inter * np.dot(np.ones((self.population_size,1)),self.outer_ass)  # NIND x D

        fitness_best = 1000000
        CV_best = 1000000000000000
        population_best = population[0]
        cost_urllc_best = 0

        for _ in range(self.generation):  # Number of generations
            population_generation = np.zeros((self.generation,self.chromosome_length))
            fitness_generation = np.zeros((self.generation,1))
            CV_generation = np.zeros((self.generation,1))
            cost_urllc_generation = np.zeros((self.generation,1))



            donor_population = self.mutate(population)
            trial_population = self.crossover(population, donor_population)
            fitness_pop, cost_urllc_pop,CV_pop, CV_pha_pop,trans_delay_pop,queue_delay_pop = self.evalVars(population)
            fitness_trial, cost_urllc_trail,CV_trial, CV_pha_trial,trans_delay_trial,queue_delay_trial = self.evalVars(trial_population)
            #  选出两个种群的最优解
            best_fitness,best_population,best_CV_pha,best_cost_urllc,best_trans,best_queue  = self.select(fitness_trial,trial_population,CV_pha_trial,cost_urllc_trail,trans_delay_trial,queue_delay_trial,
                                                                   fitness_pop,population,CV_pha_pop,cost_urllc_pop,trans_delay_pop,queue_delay_pop)
    



            # 可以根据更新后的适应度进行选择， 进行排序
            best_fitness , best_population, best_CV_pha,best_cost_urllc,best_trans, best_queue = self.select_based_on_fitness(best_population, best_fitness,best_CV_pha,best_cost_urllc,best_trans,best_queue) 
            # population_local,selected_fitness = self.select_based_on_fitness(population, fitness) 

            population_generation[_] =  best_population[0]   #记录了每一代的最优解
            fitness_generation[_] = best_fitness[0]
            CV_generation[_] = best_CV_pha[0]
            cost_urllc_generation[_] = best_cost_urllc[0]



            if best_CV_pha[0] < CV_best:  
                fitness_best = best_fitness[0]     # 选出了所有代中的最优解
                population_best = best_population[0]
                CV_best = best_CV_pha[0]
                cost_urllc_best = best_cost_urllc[0]

                trans_best = best_trans[0]
                queue_best = best_queue[0]
                

            elif best_CV_pha[0] == CV_best and best_fitness[0] < fitness_best:
                fitness_best = best_fitness[0]     # 选出了所有代中的最优解
                population_best = best_population[0]
                CV_best = best_CV_pha[0]
                cost_urllc_best = best_cost_urllc[0]

                trans_best = best_trans[0]
                queue_best = best_queue[0]



            
            print('Innergeneration{}|| CV{} ||  Cost{}  ||'.format(_,CV_best,fitness_best))

        return population_best,fitness_best,CV_best,cost_urllc_best       # population_best: (chormlength,1) fitness_best: : (NIND,1) CV_best: value
    
    


if __name__=="__main__":

    k1_u = 4
    k2_u = 4
    k3_u = 4
    k1_e = 4
    k2_e = 4
    k3_e = 4
    k_embb = k1_e + k2_e + k3_e 
    k_urllc = k1_u + k2_u + k3_u 
    num_list =[k1_u,k2_u,k3_u,k1_e,k2_e,k3_e]
    
    RAT_num = 6
    SixG_BSs_num = 2
    WiFi_BSs_num = 2
    Satellite_BSs_num = 2
    RAT_num = SixG_BSs_num + WiFi_BSs_num + Satellite_BSs_num
    RAT_list = np.array([SixG_BSs_num,WiFi_BSs_num,Satellite_BSs_num,Satellite_BSs_num])




    seed = np.random.seed(42)
    # outer= np.ones((k_urllc+k_embb,RAT_num))   # LLM (GPT),association  
    outer = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],     # k1_u
                     [0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,1],     # k2_u
                     [0,1,0,0,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],     # k3_u
                     [1,0,1,0,0,0],[1,0,0,1,0,0],[1,0,1,0,0,0],[0,1,0,1,0,0],     # k1_e
                     [0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,1,0],     # k2_e
                     [0,1,0,1,1,0],[1,0,1,0,1,0],[1,0,1,0,0,1],[0,1,1,0,1,0]])     # k3_e
    

    outer = np.concatenate([outer, outer[:, 4:6]], axis=1)

    calculator = RATDistanceCalculator(urllc_num = k_urllc, embb_num = k_embb,RAT_num = RAT_num,time_ = seed )
    user_positions = calculator.generate_user_positions()    # （24，3）个用户的位置
    dk_m,channel = calculator.calculate_DistancesAndChennel(user_positions) # （24，6）个用户到各RAT的距离和信道增益
    # ch = np.ones((k_embb+k_urllc,RAT_num))

    Inner = MyproblemInner(k_urllc,k_embb,RAT_num,seed,outer,channel,num_list,RAT_list)
    population_best,fitness_best,CV_best,cost_urllc_best  =  Inner.run_origin()

