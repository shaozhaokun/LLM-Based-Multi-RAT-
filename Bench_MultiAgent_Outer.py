import numpy as np
from Bench_MultiAgent_Inner import MyproblemInner
from Position_channel_gen import RATDistanceCalculator
from Task_gen import TaskGenerator
import random
import time
import os


#

class MyProblem:
    def __init__(self, URLLC_num, eMBB_num, RAT_num, seed, gen, channel, num_list=None, RAT_list=None, RAT_num_cure=None, embb_k: int = 2):
        self.URLLC_num = URLLC_num
        self.eMBB_num = eMBB_num
        # RAT_num: 兼容旧接口。卫星模型下推荐：
        # - RAT_num_cure = 上行RAT数量 = (6G + WiFi + Sat_up)
        # - RAT_list = [M1, M2, M3, M3]，总RAT数 = sum(RAT_list) = RAT_num_cure + Sat_down
        self.RAT_num = RAT_num
        if RAT_list is None:
            raise ValueError("需要传入 RAT_list（例如 [M1,M2,M3,M3]）以启用卫星模型。")
        self.RAT_list = np.asarray(RAT_list).astype(int)
        self.RAT_num_total = int(np.sum(self.RAT_list))
        self.RAT_num_up = int(np.sum(self.RAT_list[:3]))
        self.RAT_num_down = int(self.RAT_list[3])
        self.RAT_num_cure = int(RAT_num_cure) if RAT_num_cure is not None else int(self.RAT_num_up)
        self.num_list = num_list  # [k1_u,k2_u,k3_u,k1_e,k2_e,k3_e]；None 则默认全是 K3
        self.embb_k = int(embb_k)
        self.seed = seed
        self.gen = gen
        self.chennel = channel
        self.population_size = 5   # outer individual 
        self.outer_genneration = 10  # outer generation 
        self.inner_generation = 100
        # outer 染色体：association 掩码（与 inner allocation 向量同维度）
        self.chromosome_length = (URLLC_num + eMBB_num) * self.RAT_num_total
        
       
        self.W_5g_ = 7 * 1e5            # 200 eMBB 200 URLLC real
        self.W_4g_ = 1.5 * 1e5    
        self.W_6g_ = 2 * 1e6    
        self.W_wifi_ = 3 * 1e5 


        # self.W_5g_ = 4.5 * 1e5        # 350 eMBB 50 URLLC
        # self.W_4g_ = 1 * 1e5    
        # self.W_6g_ = 2.2 * 1e6    
        # self.W_wifi_ = 2.2 * 1e5


        # self.W_5g_ = 4.5 * 1e5        # 300 eMBB 100 URLLC
        # self.W_4g_ = 1.3 * 1e5    
        # self.W_6g_ = 2.2 * 1e6    
        # self.W_wifi_ = 2.2 * 1e5

        # self.W_5g_ = 7.5 * 1e5       # 100eMBB 300URLLC
        # self.W_4g_ = 1.5 * 1e5    
        # self.W_6g_ = 4.5 * 1e6    
        # self.W_wifi_ = 3.2 * 1e5  

        # self.W_5g_ = 9.3 * 1e5   
        # self.W_4g_ = 1.6 * 1e5     # 350URLLC 50eMBB
        # self.W_6g_ = 3 * 1e6    
        # self.W_wifi_ = 3 * 1e5


        # self.W_5g_ = 5 * 1e5    # 100 MHz
        # self.W_4g_ = 1 * 1e5    # 20 MHz
        # self.W_6g_ = 2 * 1e6    # 20 GHz
        # self.W_wifi_ = 2 * 1e5  # 160 MHz

        
       
        np.random.seed(seed)

        # 定义上下界（outer 是 0/1，不依赖 lb/ub；保留字段以兼容旧代码）
        self.lb = [0] * self.chromosome_length
        self.ub = [1] * self.chromosome_length

    def _allowed_uplink_rats_for_user(self, user_global_idx: int) -> np.ndarray:
        """根据 K1/K2/K3 规则返回该用户允许的 uplink RAT 索引集合（范围 0..RAT_num_up-1）。"""
        M1 = int(self.RAT_list[0])
        M2 = int(self.RAT_list[1])
        M3 = int(self.RAT_list[2])
        terr = np.arange(0, M1 + M2, dtype=int)
        sat = np.arange(M1 + M2, M1 + M2 + M3, dtype=int)

        if self.num_list is None:
            return np.arange(0, self.RAT_num_up, dtype=int)  # 默认全 K3

        k1_u, k2_u, k3_u, k1_e, k2_e, k3_e = [int(x) for x in self.num_list]
        K_u = k1_u + k2_u + k3_u
        if user_global_idx < K_u:
            # URLLC 用户
            u = user_global_idx
            if u < k1_u:
                return terr
            if u < k1_u + k2_u:
                return sat
            return np.arange(0, self.RAT_num_up, dtype=int)
        else:
            # eMBB 用户（用 e 的相对索引判断区域）
            e = user_global_idx - K_u
            if e < k1_e:
                return terr
            if e < k1_e + k2_e:
                return sat
            return np.arange(0, self.RAT_num_up, dtype=int)

    def _repair_outer_assoc(self, outer_vec: np.ndarray) -> np.ndarray:
        """
        修复 outer association：
        - URLLC uplink one-hot（只在允许集合内）
        - eMBB uplink 选择 self.embb_k 个（不足则全选），只在允许集合内
        - downlink 卫星列由 uplink 卫星列派生（复制）
        """
        outer = np.asarray(outer_vec).reshape(self.URLLC_num + self.eMBB_num, self.RAT_num_total)
        outer = (outer > 0.5).astype(int)

        M1 = int(self.RAT_list[0])
        M2 = int(self.RAT_list[1])
        M3 = int(self.RAT_list[2])
        terr_count = M1 + M2

        # 先清空 downlink，再由 uplink 卫星列派生
        outer[:, self.RAT_num_up:] = 0

        # 最小改动修复：只在“违反规则”时修改；并且用确定性策略（最小索引）避免额外随机扰动
        for u in range(self.URLLC_num + self.eMBB_num):
            allowed = self._allowed_uplink_rats_for_user(u)
            allowed = np.asarray(allowed, dtype=int)
            allowed = allowed[(allowed >= 0) & (allowed < self.RAT_num_up)]
            if allowed.size == 0:
                # 理论上不会发生（至少应允许一个 RAT），这里兜底：全部清零
                outer[u, : self.RAT_num_up] = 0
                continue

            # 先清除不允许的 uplink 连接（这是“最小改动”的必要步骤）
            allowed_mask = np.zeros((self.RAT_num_up,), dtype=bool)
            allowed_mask[allowed] = True
            row_up = outer[u, : self.RAT_num_up].copy()
            row_up[~allowed_mask] = 0

            if u < self.URLLC_num:
                # URLLC: one-hot
                ones = np.where(row_up > 0)[0]
                if ones.size == 1:
                    pick = int(ones[0])  # already valid
                elif ones.size == 0:
                    pick = int(np.min(allowed))  # minimal change (deterministic fill)
                else:
                    pick = int(np.min(ones))  # deterministic keep one
                row_up[:] = 0
                row_up[pick] = 1

                outer[u, : self.RAT_num_up] = row_up
                # derive satellite downlink if uplink is satellite
                if pick >= terr_count:
                    outer[u, self.RAT_num_up + (pick - terr_count)] = 1
            else:
                # eMBB: keep exactly k (=embb_k) within allowed, if possible
                ones = np.where(row_up > 0)[0]
                k_eff = int(min(max(1, self.embb_k), allowed.size))

                if ones.size == k_eff:
                    picks = np.sort(ones)  # already valid (deterministic order)
                elif ones.size > k_eff:
                    picks = np.sort(ones)[:k_eff]  # deterministic drop extras
                else:
                    # add smallest not-yet-selected allowed indices
                    remain = np.setdiff1d(np.sort(allowed), np.sort(ones), assume_unique=False)
                    need = k_eff - ones.size
                    add = remain[:need] if need > 0 else np.array([], dtype=int)
                    picks = np.concatenate([np.sort(ones), add]).astype(int)

                row_up[:] = 0
                row_up[picks] = 1
                outer[u, : self.RAT_num_up] = row_up

                # derive downlink satellite
                sat_picks = picks[picks >= terr_count]
                for p in sat_picks:
                    outer[u, self.RAT_num_up + (int(p) - terr_count)] = 1

        return outer.reshape(-1)

 
    def initialize_population(self):
        # 初始化 outer association 掩码（与 inner allocation 同维度），并做一次 repair
        K = self.URLLC_num + self.eMBB_num
        pop = np.zeros((self.population_size, K, self.RAT_num_total), dtype=int)

        M1 = int(self.RAT_list[0])
        M2 = int(self.RAT_list[1])
        terr_count = M1 + M2

        for i in range(self.population_size):
            # URLLC：uplink one-hot
            for u in range(self.URLLC_num):
                allowed = self._allowed_uplink_rats_for_user(u)
                pick = int(np.random.choice(allowed))
                pop[i, u, pick] = 1
                if pick >= terr_count:
                    pop[i, u, self.RAT_num_up + (pick - terr_count)] = 1

            # eMBB：uplink 选 k 个
            for e in range(self.eMBB_num):
                row = self.URLLC_num + e
                allowed = self._allowed_uplink_rats_for_user(row)
                k = min(max(1, self.embb_k), allowed.size)
                picks = np.random.choice(allowed, size=k, replace=False)
                pop[i, row, picks] = 1
                sat_picks = picks[picks >= terr_count]
                for p in sat_picks:
                    pop[i, row, self.RAT_num_up + (int(p) - terr_count)] = 1

        pop_flat = pop.reshape(self.population_size, -1)
        for i in range(self.population_size):
            pop_flat[i] = self._repair_outer_assoc(pop_flat[i])
        return pop_flat

   



    def fitness_function_origin(self, population,population_inner,best_value,Cost_eMBB_value,best_trans_value,best_queue_value):
        # 获取种群大小
        NIND = population.shape[0]
        
        # 卫星模型：population 本身就是 outer_ass 掩码（K_total * RAT_num_total），先做 repair
        population_outer = population.copy()
        for j in range(NIND):
            population_outer[j] = self._repair_outer_assoc(population_outer[j])
        

        # 初始化适应度数组
        fitness_inner = np.zeros((NIND, 1))
        population_inner = np.zeros((NIND, self.chromosome_length))
        cost_urllc_inner  = np.zeros((NIND, 1))
        CV_inner = np.zeros((NIND, 1))
        best_value_matrix = np.zeros((NIND,self.inner_generation))
        trans_delay_matrix = np.zeros((2,NIND,self.inner_generation))  # 1 --> URLLC 2--> eMBB
        queue_delay_matrix = np.zeros((2,NIND,self.inner_generation))

        # 对每个个体评估其适应度
        for j in range(NIND):
            problem_inner = MyproblemInner(
                URLLC_num=self.URLLC_num,
                eMBB_num=self.eMBB_num,
                RAT_num=self.RAT_num_cure,
                seed=self.seed,
                outer_ass=population_outer[j],
                population_inner=population_inner[j],
                ch=self.chennel,
                num_list=self.num_list,
                RAT_list=self.RAT_list,
                RAT_num_cure=self.RAT_num_cure,
                inner_generation=self.inner_generation,
                inner_population_size=10,
            )
            # 调用 run 方法来计算适应度
            inner_population_best,inner_fitness_best,CV_best,best_value_matrix,cost_urllc_best,trans_delay_matrix,queue_delay_matrix = problem_inner.run_origin(j,best_value_matrix,trans_delay_matrix,queue_delay_matrix)

            fitness_inner[j] = inner_fitness_best
            population_inner[j] = inner_population_best
            CV_inner[j] = CV_best
            cost_urllc_inner[j] = cost_urllc_best

        
        
        column_min_values = np.min(best_value_matrix, axis=0)
        column_min_indices = np.argmin(best_value_matrix, axis=0)  # 获取最小值的索引
        col_indices = np.arange(best_value_matrix.shape[1])

        best_trans_value[0,0:self.inner_generation] = trans_delay_matrix[0,column_min_indices,col_indices]  # (100,)
        best_trans_value[1,0:self.inner_generation]  = trans_delay_matrix[1,column_min_indices,col_indices]  # (100,)
        best_queue_value[0,0:self.inner_generation] = queue_delay_matrix[0,column_min_indices,col_indices]
        best_queue_value[1,0:self.inner_generation] = queue_delay_matrix[1,column_min_indices,col_indices]



        # 更新 best_value 以保证其始终包含历史最小值
        if len(best_value) == 0:  # 如果 best_value 是空的，直接添加
            best_value.extend(column_min_values)
            Cost_eMBB_value.extend(best_queue_value[1,0:self.inner_generation])
        else:
            # 比较并更新 best_value 中的每个元素
            last_values = best_value[-1]  # 获取最后一组元素
            new_values = np.minimum(last_values, column_min_values)
            best_value.extend(new_values)  # 追加新的最小值



        return fitness_inner,population_inner,CV_inner,best_value,Cost_eMBB_value,cost_urllc_inner,best_trans_value,best_queue_value        # fitness_inner: (NIND_outer,1) population_inner: (NIND_outer,1) CV_inner: (NIND_outer,1)
    



    def fitness_function(self, population,population_inner,best_value,Cost_eMBB_value,outer_gen,best_trans_value,best_queue_value):
        # 获取种群大小
        NIND = population.shape[0]
        
        # 卫星模型：population 本身就是 outer_ass 掩码（K_total * RAT_num_total），先做 repair
        population_outer = population.copy()
        for j in range(NIND):
            population_outer[j] = self._repair_outer_assoc(population_outer[j])



        # 初始化适应度数组
        fitness_inner = np.zeros((NIND, 1))
        cost_urllc_inner  = np.zeros((NIND, 1))
        # population_inner = np.zeros((NIND, (self.RAT_num*self.eMBB_num)*3+self.URLLC_num+1))

        best_value_matrix = np.zeros((NIND,self.inner_generation))                     # [[20,32,10,30....]      
        trans_delay_matrix = np.zeros((2,NIND,self.inner_generation))  # 1 --> URLLC 2--> eMBB
        queue_delay_matrix = np.zeros((2,NIND,self.inner_generation))
        CV_inner = np.zeros((NIND, 1))  
                                                      #  .........
                                                    # [0.1,0.2,0.3,0.4....]]

        # 对每个个体评估其适应度
        for j in range(NIND//2):
            print('outer_individual---{}'.format(j))
            
            problem_inner = MyproblemInner(
                URLLC_num=self.URLLC_num,
                eMBB_num=self.eMBB_num,
                RAT_num=self.RAT_num_cure,
                seed=self.seed,
                outer_ass=population_outer[j],
                population_inner=population_inner[j],
                ch=self.chennel,
                num_list=self.num_list,
                RAT_list=self.RAT_list,
                RAT_num_cure=self.RAT_num_cure,
                inner_generation=self.inner_generation,
                inner_population_size=10,
            )
            # 调用 run 方法来计算适应度
            inner_population_best,inner_fitness_best,CV_best,best_value_matrix,cost_urllc_best,trans_delay_matrix,queue_delay_matrix = problem_inner.run(
                j, best_value_matrix, trans_delay_matrix, queue_delay_matrix, population=None
            )

            fitness_inner[j] = inner_fitness_best
            population_inner[j] = inner_population_best
            CV_inner[j] = CV_best
            cost_urllc_inner[j] = cost_urllc_best


        
        for j in range(NIND//2,NIND):
            print('outer_individual---{}'.format(j))
            problem_inner = MyproblemInner(
                URLLC_num=self.URLLC_num,
                eMBB_num=self.eMBB_num,
                RAT_num=self.RAT_num_cure,
                seed=self.seed,
                outer_ass=population_outer[j],
                population_inner=population_inner[j],
                ch=self.chennel,
                num_list=self.num_list,
                RAT_list=self.RAT_list,
                RAT_num_cure=self.RAT_num_cure,
                inner_generation=self.inner_generation,
                inner_population_size=10,
            )
            # 调用 run 方法来计算适应度
            inner_population_best,inner_fitness_best,CV_best,best_value_matrix,cost_urllc_best,trans_delay_matrix,queue_delay_matrix = problem_inner.run(
                j, best_value_matrix, trans_delay_matrix, queue_delay_matrix, population=None
            )

            fitness_inner[j] = inner_fitness_best
            population_inner[j] = inner_population_best
            CV_inner[j] = CV_best
            cost_urllc_inner[j] = cost_urllc_best



            

        column_min_values = np.min(best_value_matrix, axis=0)
        column_min_indices = np.argmin(best_value_matrix, axis=0)  # 获取最小值的索引
        col_indices = np.arange(best_value_matrix.shape[1])
        best_trans_value[0,outer_gen*self.inner_generation:(outer_gen+1)*self.inner_generation] = trans_delay_matrix[0,column_min_indices,col_indices]  # (100,)
        best_trans_value[1,outer_gen*self.inner_generation:(outer_gen+1)*self.inner_generation]  = trans_delay_matrix[1,column_min_indices,col_indices]  # (100,)
        best_queue_value[0,outer_gen*self.inner_generation:(outer_gen+1)*self.inner_generation] = queue_delay_matrix[0,column_min_indices,col_indices]
        best_queue_value[1,outer_gen*self.inner_generation:(outer_gen+1)*self.inner_generation]= queue_delay_matrix[1,column_min_indices,col_indices]

        # 更新 best_value 以保证其始终包含历史最小值
        if len(best_value) == 0:  # 如果 best_value 是空的，直接添加
            best_value.extend(column_min_values)
        else:
            # 比较并更新 best_value 中的每个元素
            last_values1 = best_value[-1]  # 获取最后一组元素
            new_values1 = np.minimum(last_values1, column_min_values)
            best_value.extend(new_values1)  # 追加新的最小值
            last_values2 = Cost_eMBB_value[-1]
            new_values2 = np.minimum(last_values2, best_queue_value[1,outer_gen*self.inner_generation:(outer_gen+1)*self.inner_generation])
            Cost_eMBB_value.extend(new_values2)

        



        return fitness_inner,population_inner,CV_inner,best_value,Cost_eMBB_value,cost_urllc_inner,best_trans_value,best_queue_value

 
    def select_(self, fitness_new, population_outer_new,population_inner_new,CV_pha_new ,best_fitness,best_population_outer,best_population_inner,best_CV_pha):
        for i in range(self.population_size):
            if CV_pha_new[i] < best_CV_pha[i]:
                best_population_outer[i] = population_outer_new[i]
                best_population_inner[i] = population_inner_new[i]
                best_fitness[i] = fitness_new[i]
                best_CV_pha[i] = CV_pha_new[i]
            elif CV_pha_new[i] == best_CV_pha[i] and fitness_new[i] < best_fitness[i]:
                best_population_outer[i] = population_outer_new[i]
                best_population_inner[i] = population_inner_new[i]
                best_fitness[i] = fitness_new[i]
                best_CV_pha[i] = CV_pha_new[i]

        
        return best_fitness, best_population_outer, best_population_inner, best_CV_pha

    def select(self,selected_fitness,selected_population_outer,selected_population_inner,selected_CV_pha,selected_cost_urllc):
        best_fitness = selected_fitness[:self.population_size]
        best_population_outer = selected_population_outer[:self.population_size]
        best_population_inner = selected_population_inner[:self.population_size]
        best_CV_pha = selected_CV_pha[:self.population_size]
        best_cost_urllc = selected_cost_urllc[:self.population_size]

      
        return best_fitness , best_population_outer, best_population_inner,best_CV_pha,best_cost_urllc

        
    def crossover(self, population_outer, population_inner):
        """
        卫星模型 outer 的交叉：
        - 按“用户块”做交叉：以概率交换某个用户的整行 association（uplink + downlink 卫星派生后由 repair 保证一致）
        - 交叉后做 _repair_outer_assoc()，保证 URLLC one-hot / eMBB k-hot / 下行卫星派生
        """
        N = population_outer.shape[0]
        D = population_outer.shape[1]
        assert D == self.chromosome_length

        crossover_rate = 0.7
        gene_rate_user = 0.7  # per-user swap probability when crossover happens

        perm = np.random.permutation(N)
        half = N // 2
        p1s = perm[:half]
        p2s = perm[half:half + half]

        child_outer = population_outer.copy().astype(int)
        child_inner = population_inner.copy()

        K = self.URLLC_num + self.eMBB_num
        child_outer_3d = child_outer.reshape(N, K, self.RAT_num_total)

        for p1, p2 in zip(p1s, p2s):
            if np.random.rand() >= crossover_rate:
                continue
            for u in range(K):
                if np.random.rand() < gene_rate_user:
                    tmp = child_outer_3d[p1, u, :].copy()
                    child_outer_3d[p1, u, :] = child_outer_3d[p2, u, :]
                    child_outer_3d[p2, u, :] = tmp

        child_outer = child_outer_3d.reshape(N, -1)

        # repair outer + mask inner
        for i in range(N):
            child_outer[i] = self._repair_outer_assoc(child_outer[i])
        child_inner = child_inner * child_outer

        return child_outer, child_inner

    
    
    def mutate(self, population,population_inner):   
        """
        卫星模型 outer 的变异：
        - 按“用户块”做结构化变异：
          - URLLC：以小概率把 uplink one-hot 改到另一个 allowed RAT
          - eMBB：保持 uplink 恰好 k(=embb_k) 个连接，做“替换一个 RAT”式变异
        - 变异后调用 _repair_outer_assoc()（最小改动修复）
        - inner 只是占位：保持与 outer 同形状，并且被 outer 掩码约束为 0（不影响 inner 求解）
        """
        mutation_rate = 0.1

        new_outer = population.copy().astype(int)
        new_inner = population_inner.copy()

        N, D = new_outer.shape
        assert D == self.chromosome_length

        K = self.URLLC_num + self.eMBB_num
        outer3 = new_outer.reshape(N, K, self.RAT_num_total)

        M1 = int(self.RAT_list[0])
        M2 = int(self.RAT_list[1])
        terr_count = M1 + M2

        for i in range(N):
            # ---- URLLC mutate (change to another allowed RAT) ----
            for u in range(self.URLLC_num):
                if np.random.rand() >= mutation_rate:
                    continue
                allowed = np.asarray(self._allowed_uplink_rats_for_user(u), dtype=int)
                allowed = allowed[(allowed >= 0) & (allowed < self.RAT_num_up)]
                if allowed.size <= 1:
                    continue

                row_up = outer3[i, u, : self.RAT_num_up].copy()
                ones = np.where(row_up > 0)[0]
                cur = int(ones[0]) if ones.size == 1 else int(np.min(allowed))
                cand = allowed[allowed != cur]
                if cand.size == 0:
                    continue
                pick = int(np.random.choice(cand))

                # set uplink one-hot (do not touch others)
                outer3[i, u, : self.RAT_num_up] = 0
                outer3[i, u, pick] = 1

            # ---- eMBB mutate (swap one RAT, keep k) ----
            for e in range(self.eMBB_num):
                row = self.URLLC_num + e
                if np.random.rand() >= mutation_rate:
                    continue
                allowed = np.asarray(self._allowed_uplink_rats_for_user(row), dtype=int)
                allowed = allowed[(allowed >= 0) & (allowed < self.RAT_num_up)]
                if allowed.size <= 1:
                    continue
                k_eff = int(min(max(1, self.embb_k), allowed.size))

                row_up = outer3[i, row, : self.RAT_num_up].copy()
                ones = np.where(row_up > 0)[0]

                # if current is not exactly k, let repair handle; otherwise do structured swap
                if ones.size == k_eff:
                    remain = np.setdiff1d(allowed, ones, assume_unique=False)
                    if remain.size == 0:
                        continue
                    drop = int(np.random.choice(ones))
                    add = int(np.random.choice(remain))
                    row_up[drop] = 0
                    row_up[add] = 1
                    outer3[i, row, : self.RAT_num_up] = row_up

            # repair + derive downlink satellite deterministically/minimally
            new_outer[i] = self._repair_outer_assoc(outer3[i].reshape(-1))

        new_inner = new_inner * new_outer
        return new_outer, new_inner

  
  
    def combine(self, fitness_new, population_outer_new,population_inner_new,CV_pha_new ,cost_urllc_inner_new,best_fitness,best_population_outer,best_population_inner,best_CV_pha,best_cost_urllc):
        combined_fitness = np.concatenate((fitness_new, best_fitness))
        combined_population_outer = np.concatenate((population_outer_new, best_population_outer))
        combined_population_inner = np.concatenate((population_inner_new, best_population_inner))
        combined_CV = np.concatenate((CV_pha_new, best_CV_pha))
        combined_cost_urllc = np.concatenate((cost_urllc_inner_new, best_cost_urllc)) 

        return combined_fitness , combined_population_outer, combined_population_inner,combined_CV,combined_cost_urllc



    def select_based_on_fitness(self, best_population_outer,best_population_inner, best_fitness,best_CV_pha,best_cost_urllc):
        # 基于可行性法则选择种群，保持形状为
            # 根据CV排序
        best_CV_pha_ = best_CV_pha.flatten()
        # selected_indices = np.argsort(best_CV_pha)[:self.population_size]
        selected_indices = np.argsort(best_CV_pha_)
        selected_population_inner = best_population_inner[selected_indices]
        selected_population_outer = best_population_outer[selected_indices]
        selected_fitness = best_fitness[selected_indices]  # 获取对应的适应度
        selected_CV_pha = best_CV_pha[selected_indices]  # 获取对应的适应度
        selected_cost_urllc = best_cost_urllc[selected_indices]

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
            selected_population_inner[zero_CV_indices] = selected_population_inner[zero_CV_indices_sorted]
            selected_population_outer[zero_CV_indices] = selected_population_outer[zero_CV_indices_sorted]
            selected_fitness[zero_CV_indices] = selected_fitness[zero_CV_indices_sorted]
            selected_CV_pha[zero_CV_indices] = selected_CV_pha[zero_CV_indices_sorted]
            selected_cost_urllc[zero_CV_indices]  = selected_cost_urllc[zero_CV_indices_sorted]

        return selected_fitness,selected_population_outer,selected_population_inner,selected_CV_pha,selected_cost_urllc
        

    def run_main(self):
        
        start_time = time.time()  # 记录开始时间

        best_value = []
        Cost_eMBB_value = []
        best_trans_value = np.zeros((2,self.outer_genneration*self.inner_generation))
        best_queue_value = np.zeros((2,self.outer_genneration*self.inner_generation))
        population_outer = self.initialize_population()
        # inner 染色体与 outer 掩码同维度（allocation 向量），这里初始化为 0 占位即可
        population_inner = np.zeros((self.population_size, self.chromosome_length))
        fitness,population_inner,CV_inner,best_value,Cost_eMBB_value,cost_urllc_inner,best_trans_value,best_queue_value = self.fitness_function_origin(population_outer,population_inner,best_value,Cost_eMBB_value,best_trans_value,best_queue_value)
        best_fitness , best_population_outer, best_population_inner, best_CV_inner,best_cost_urllc_inner = fitness,population_outer,population_inner,CV_inner,cost_urllc_inner
        # best_CV_inner: [NIND,1]
        # 根据可行性法则排序过后
        selected_fitness,selected_population_outer,selected_population_inner,selected_CV_pha,selected_cost_urllc = self.select_based_on_fitness(best_population_outer,best_population_inner,best_fitness,best_CV_inner,best_cost_urllc_inner)
        print('\ngeneration0:{},outage{}\n'.format(selected_fitness[0][0],selected_cost_urllc[0][0]))
        # print(best_value)       

        # population_outer,population_inner = self.crossover(population_outer,population_inner)
        # population_outer_new,population_inner_new = self.mutate(population_outer,population_inner)
        population_outer,population_inner = self.crossover(best_population_outer.copy(),best_population_inner.copy()) #????
        population_outer_new,population_inner_new = self.mutate(population_outer,population_inner)
        

        for _ in range(1,self.outer_genneration):  # Number of generations

            population_outer_new[0:self.population_size//2],population_inner_new[0:self.population_size//2] = selected_population_outer[0:self.population_size//2],selected_population_inner[0:self.population_size//2]
            
            fitness_new,population_inner_new,CV_inner_new,best_value,Cost_eMBB_value,cost_urllc_inner_new,best_trans_value,best_queue_value = self.fitness_function(population_outer_new,population_inner_new,best_value,Cost_eMBB_value,_,best_trans_value,best_queue_value)
            
            combined_fitness,combined_population_outer,combined_population_inner,combined_CV_pha,combined_cost_urllc = self.combine(fitness_new, population_outer_new,population_inner_new,CV_inner_new,cost_urllc_inner_new, best_fitness
                                                                                                                ,best_population_outer,best_population_inner,best_CV_inner,best_cost_urllc_inner)
           
            # best_fitness , best_population_outer, best_population_inner,best_CV_inner= self.select(fitness_new, population_outer_new, population_inner_new,CV_inner_new
            #                                                 ,best_fitness , best_population_outer, best_population_inner,best_CV_inner)
            
            selected_fitness,selected_population_outer,selected_population_inner,selected_CV_pha,selected_cost_urllc = self.select_based_on_fitness(combined_population_outer,combined_population_inner,combined_fitness,combined_CV_pha,combined_cost_urllc)
            # 固定住固定种群大小
            best_fitness , best_population_outer, best_population_inner,best_CV_inner,best_cost_urllc= self.select(selected_fitness,selected_population_outer,selected_population_inner,selected_CV_pha,selected_cost_urllc)
            print('\ngeneration{}:{},outage{}'.format(_,best_fitness[0][0],best_cost_urllc[0][0]))

            # print('generation{}:{}'.format(_,best_fitness))
            population_outer,population_inner = self.crossover(best_population_outer.copy(),best_population_inner.copy())
            population_outer_new,population_inner_new = self.mutate(population_outer,population_inner)
            print(best_value)
        np.save('Result_MultiAgent/simulation{}_multi_agent_BestValue.npy'.format(self.seed),best_value)
        np.save('Result_MultiAgent/simulation{}_multi_agent_OutageValue.npy'.format(self.seed),best_queue_value)
        np.save('Result_MultiAgent/simulation{}_multi_agent_Transtime.npy'.format(self.seed),best_trans_value)
        np.save('Result_MultiAgent/simulation{}_multi_agent_CosteMBB.npy'.format(self.seed),Cost_eMBB_value)


        
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间

        print(f"代码运行时间：{elapsed_time}秒")
        
        pass





if __name__ == "__main__":

    # 用户数量（与 Bench_All_DE.py 的写法对齐）
    k1_u = 4
    k2_u = 4
    k3_u = 4
    k1_e = 4
    k2_e = 4
    k3_e = 4
    k_embb = k1_e + k2_e + k3_e
    k_urllc = k1_u + k2_u + k3_u
    num_list = [k1_u, k2_u, k3_u, k1_e, k2_e, k3_e]
    # 卫星模型 RAT 配置（与 Bench_All_DE/GA 一致）
    SixG_BSs_num = 2
    WiFi_BSs_num = 4
    Satellite_BSs_num = 1
    RAT_num_cure = SixG_BSs_num + WiFi_BSs_num + Satellite_BSs_num  # uplink RAT 数（用于 Position_channel_gen）
    RAT_list = np.array([SixG_BSs_num, WiFi_BSs_num, Satellite_BSs_num, Satellite_BSs_num])

    # num_list 已按 K1/K2/K3 分区统计给出

    # task generate（只生成一次；若文件已存在则不覆盖）
    urllc_task_path = f"Data/urllc_tasks_{k_urllc}.csv"
    embb_task_path = f"Data/embb_tasks_{k_embb}.csv"
    if (not os.path.exists(urllc_task_path)) or (not os.path.exists(embb_task_path)):
        task_generator = TaskGenerator(num_urllc_users=k_urllc, num_embb_users=k_embb)
        task_generator.save_tasks_to_csv()

    for time_ in range (10):
    
    # time_ = 99

        # position and channel generate
        calculator = RATDistanceCalculator(
            urllc_num=k_urllc,
            embb_num=k_embb,
            RAT_num=RAT_num_cure,
            time_=time_,
            RAT_list=RAT_list,
        )
        user_positions = calculator.generate_user_positions()
        dk_m,channel = calculator.calculate_DistancesAndChennel(user_positions)

        # Example usage
        problem = MyProblem(
            URLLC_num=k_urllc,
            eMBB_num=k_embb,
            RAT_num=RAT_num_cure,
            seed=time_,
            gen=0,
            channel=channel,
            num_list=num_list,
            RAT_list=RAT_list,
            RAT_num_cure=RAT_num_cure,
            embb_k=3,
        )
        final_population = problem.run_main()
