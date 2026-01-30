
import numpy as np
import pandas as pd

class TaskGenerator:
    def __init__(self, num_urllc_users, num_embb_users):
        self.k0 = 330  # cycles per byte
        self.num_urllc_users = num_urllc_users
        self.num_embb_users = num_embb_users

        # URLLC parameters
        self.urllc_task_arrival_prob = 1  # 0.6
        self.urllc_deadline_lower = 5 * 1e-5  # s (0.1ms) - 修正：至少要大于物理处理时间(~0.03ms)
        self.urllc_deadline_upper = 10* 1e-5  # s (0.4ms) - 适当放宽上限
        self.urllc_data_size_lower = 100  # bytes
        self.urllc_data_size_upper = 400  # bytes

        # eMBB parameters
        self.embb_task_arrival_prob = 0.4
        self.embb_deadline = 2000 * 1e-3  # s
        self.embb_data_size_lower = 75 * 1024  # bytes
        self.embb_data_size_upper = 125 * 1024  # bytes

    def generate_urllc_tasks(self):
        urllc_tasks = []
        for _ in range(self.num_urllc_users):
            data_size = np.random.randint(self.urllc_data_size_lower, self.urllc_data_size_upper + 1)
            deadline = np.random.uniform(self.urllc_deadline_lower, self.urllc_deadline_upper)
            cpu_cycles = self.k0 * data_size
            data_size_bits = data_size * 8
            urllc_tasks.append((data_size_bits, deadline, cpu_cycles))
        return urllc_tasks

    def generate_embb_tasks(self):
        embb_tasks = []
        for _ in range(self.num_embb_users):
            data_size = np.random.randint(self.embb_data_size_lower, self.embb_data_size_upper + 1)
            deadline = self.embb_deadline
            cpu_cycles = self.k0 * data_size
            data_size_bits = data_size * 8
            embb_tasks.append((data_size_bits, deadline, cpu_cycles))
        return embb_tasks

    def save_tasks_to_csv(self):
        urllc_tasks = self.generate_urllc_tasks()
        embb_tasks = self.generate_embb_tasks()

        urllc_df = pd.DataFrame(urllc_tasks, columns=["Data Size (bits)", "Deadline (s)", "CPU Cycles"])
        embb_df = pd.DataFrame(embb_tasks, columns=["Data Size (bits)", "Deadline (s)", "CPU Cycles"])

        urllc_df.to_csv("Data/urllc_tasks_{}_.csv".format(self.num_urllc_users), index=False)
        embb_df.to_csv("Data/embb_tasks_{}_.csv".format(self.num_embb_users), index=False)

        print("URLLC and eMBB tasks have been saved to 'urllc_tasks.csv' and 'embb_tasks.csv' respectively.")

# 使用示例
task_generator = TaskGenerator(num_urllc_users=12, num_embb_users=12)
task_generator.save_tasks_to_csv()