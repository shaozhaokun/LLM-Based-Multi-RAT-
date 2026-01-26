import numpy as np

class Task():
    def __init__(self, task_id,arrival_time,process_time,task_deadline):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.process_time =process_time
        self.task_deadline = task_deadline
        self.queue_delay=0
        self.total_delay=0

class Scheduling ():

    def __init__(self) :
        pass

    def Scheduler_eMBB(self,task_arrival_list):
        Q = task_arrival_list
        s =[]
        T =[]
        for task in Q:
            T_bar = [item for item in T if item.total_delay <= task.arrival_time]
            T_set=set(T_bar)
            T_tilde =[item for item in T if item not in T_set]
            if len(T_tilde)==0:
                s = s+T_bar
                T =T_tilde
                T.append(task) #######
                # T =sorted(T, key=lambda A: A.process_time)
                # cumulative_time=0
                # for task_T in T:
                task.queue_delay = 0 #################
                task.total_delay = task.arrival_time+ task.queue_delay+task.process_time
                    # cumulative_time += task_T.process_time
            else:
                id =len(T_bar) #########
                ID =T[id] ######
                s = s+T_bar
                # T = T_tilde.remove(ID)
                T = [x for x in T_tilde if x != ID]
                T.append(task) #######
                # T =T ####reshuffling
                T =sorted(T, key=lambda A: A.process_time)
                cumulative_time=0
                for task_T in T:
                    task_T.queue_delay = ID.total_delay - task.arrival_time + cumulative_time #################
                    task_T.total_delay = task.arrival_time+ task_T.queue_delay+task_T.process_time
                    cumulative_time += task_T.process_time
                T=[ID]+T
                # x=1
        s=s+T
        return s

    def Scheduler_URLLC(self,task_arrival_list):
        Q = task_arrival_list
        s =[]
        T =[]
        i=0
        for task in Q:
            i=i+1
            T_bar = [item for item in T if item.total_delay <= task.arrival_time]
            # T_set=set(T_bar)
            T_tilde =[item for item in T if item not in T_bar]
            if len(T_tilde)==0:
                # s.extend(T_bar)
                s = s+T_bar
                T.clear()
                T =T_tilde
                T.append(task) #######
                T =self.Moore_Hodgson(T,task.arrival_time)
                task.queue_delay = 0 #################
                task.total_delay = task.arrival_time+ task.queue_delay+task.process_time
                    # cumulative_time += task_T.process_time
            else:
                id =len(T_bar) #########
                ID =T[id] ######

                s = s+T_bar

                T.clear()
                T = [x for x in T_tilde if x != ID]

                T.append(task) #######

                # T =T ####reshuffling
                T =self.Moore_Hodgson(T,ID.total_delay)

                cumulative_time=0
                for task_T in T:
                    task_T.queue_delay = ID.total_delay - task.arrival_time + cumulative_time #################
                    task_T.total_delay = task.arrival_time+ task_T.queue_delay+task_T.process_time
                    cumulative_time += task_T.process_time
                T=[ID]+T

        s=s+T
        return s


    def Moore_Hodgson(self,task_list, time_instant):
        F=[]
        tau =0
        tau_prime = 0
        G = sorted(task_list, key=lambda T: T.task_deadline)
        for task in G:
            F.append(task)
            tau_prime=tau_prime+task.process_time
            tau = time_instant+tau_prime
            if tau > task.task_deadline:
                Fg= max(F, key=lambda A: A.process_time)
                # F =F.remove(Fg)
                F = [x for x in F if x != Fg]
                tau_prime = tau_prime-Fg.process_time
    
        return F
    def Scheduler_eMBB_FIFO(self,task_arrival_list):
        Q = task_arrival_list
        cumulative_time=0
        for task_T in Q:
            task_T.queue_delay = cumulative_time #################
            task_T.total_delay = task_T.arrival_time+ task_T.queue_delay+task_T.process_time
            cumulative_time += task_T.process_time
        return Q
        
    def Scheduler_URLLC_FIFO(self,task_arrival_list):
        Q = task_arrival_list
        s=[]
        cumulative_time=0
        for task in Q:
            task.queue_delay = cumulative_time #################
            task.total_delay = task.arrival_time+ task.queue_delay+task.process_time
            if task.total_delay<=task.task_deadline:
                s.append(task)
                cumulative_time += task.process_time
            else:
                cumulative_time =cumulative_time
        return s   
    


def queue_delay_calculation(transmission_delay_eMBB,transmission_delay_URLLC,Belta_eMBB,Belta_URLLC,
                                Deadline_eMBB, Deadline_URLLC,max_delay_urllc,max_delay_eMBB):
        NIND=transmission_delay_eMBB.shape[0]
        eMBB_num=transmission_delay_eMBB.shape[1]
        URLLC_num=transmission_delay_URLLC.shape[1]

        SD =Scheduling ()

        Queue_eMBB = np.zeros([NIND,eMBB_num])
        trans_delay_eMBB= np.zeros([NIND,eMBB_num])
        total_delay_eMBB= np.zeros([NIND,eMBB_num])
        for i in range(NIND):
            Q_eMBB =[]
            for j in range(eMBB_num):
                tt=Task(j+1,transmission_delay_eMBB[i,j],Belta_eMBB[i,j],Deadline_eMBB[i,j])
                Q_eMBB.append(tt)
            Q_eMBB = sorted(Q_eMBB, key=lambda A: A.arrival_time)
            queue_eMBB = SD.Scheduler_eMBB(Q_eMBB)
            for j in range(eMBB_num):
                Queue_eMBB[i,j] = queue_eMBB[j].queue_delay
                trans_delay_eMBB[i,j]= queue_eMBB[j].arrival_time
                if queue_eMBB[j].total_delay>queue_eMBB[j].task_deadline:
                    total_delay_eMBB[i,j] =max_delay_eMBB
                else:
                    total_delay_eMBB[i,j]= queue_eMBB[j].total_delay

        Queue_URLLC = np.zeros([NIND,URLLC_num])
        trans_delay_URLLC = np.zeros([NIND,URLLC_num])
        total_delay_URLLC = np.zeros([NIND,URLLC_num])
        for i in range(NIND):
            Q_URLLC =[]
            for j in range(URLLC_num):
                tt=Task(j+1,transmission_delay_URLLC[i,j],Belta_URLLC[i,j],Deadline_URLLC[i,j])
                Q_URLLC.append(tt)
            Q_URLLC = sorted(Q_URLLC, key=lambda A: A.arrival_time)
            queue_URLLC = SD.Scheduler_URLLC(Q_URLLC)
            for j in range(URLLC_num):
                # mm=len(queue_URLLC)
                if j< len(queue_URLLC):
                    Queue_URLLC[i,j] = queue_URLLC[j].queue_delay
                    trans_delay_URLLC[i,j]= queue_URLLC[j].arrival_time
                    # total_delay_URLLC[i,j]= queue_URLLC[j].total_delay   
                    total_delay_URLLC[i,j]= 0 
                    
                else:
                    Queue_URLLC[i,j] = max_delay_urllc  
                    trans_delay_URLLC[i,j]= max_delay_urllc
                    total_delay_URLLC[i,j]= max_delay_urllc  
        # x=1
        return Queue_eMBB, Queue_URLLC,trans_delay_eMBB,trans_delay_URLLC,total_delay_eMBB,total_delay_URLLC

def queue_delay_calculation_FIFO(transmission_delay_eMBB,transmission_delay_URLLC,Belta_eMBB,Belta_URLLC,
                                Deadline_eMBB, Deadline_URLLC,max_delay_urllc,max_delay_eMBB):
        NIND=transmission_delay_eMBB.shape[0]
        eMBB_num=transmission_delay_eMBB.shape[1]
        URLLC_num=transmission_delay_URLLC.shape[1]

        SD =Scheduling ()

        Queue_eMBB = np.zeros([NIND,eMBB_num])
        trans_delay_eMBB= np.zeros([NIND,eMBB_num])
        total_delay_eMBB= np.zeros([NIND,eMBB_num])
        for i in range(NIND):
            Q_eMBB =[]
            for j in range(eMBB_num):
                tt=Task(j+1,transmission_delay_eMBB[i,j],Belta_eMBB[i,j],Deadline_eMBB[i,j])
                Q_eMBB.append(tt)
            Q_eMBB = sorted(Q_eMBB, key=lambda A: A.arrival_time)
            queue_eMBB = SD.Scheduler_eMBB_FIFO(Q_eMBB)
            for j in range(eMBB_num):
                Queue_eMBB[i,j] = queue_eMBB[j].queue_delay
                trans_delay_eMBB[i,j]= queue_eMBB[j].arrival_time
                if queue_eMBB[j].total_delay>queue_eMBB[j].task_deadline:
                    total_delay_eMBB[i,j] =max_delay_eMBB
                else:
                    total_delay_eMBB[i,j]= queue_eMBB[j].total_delay

        Queue_URLLC = np.zeros([NIND,URLLC_num])
        trans_delay_URLLC = np.zeros([NIND,URLLC_num])
        total_delay_URLLC = np.zeros([NIND,URLLC_num])
        for i in range(NIND):
            Q_URLLC =[]
            for j in range(URLLC_num):
                tt=Task(j+1,transmission_delay_URLLC[i,j],Belta_URLLC[i,j],Deadline_URLLC[i,j])
                Q_URLLC.append(tt)
            Q_URLLC = sorted(Q_URLLC, key=lambda A: A.arrival_time)
            queue_URLLC = SD.Scheduler_URLLC_FIFO(Q_URLLC)
            for j in range(URLLC_num):
                # mm=len(queue_URLLC)
                if j< len(queue_URLLC):
                    Queue_URLLC[i,j] = queue_URLLC[j].queue_delay
                    trans_delay_URLLC[i,j]= queue_URLLC[j].arrival_time
                    # total_delay_URLLC[i,j]= queue_URLLC[j].total_delay   
                    total_delay_URLLC[i,j]= 0 
                    
                else:
                    Queue_URLLC[i,j] = max_delay_urllc  
                    trans_delay_URLLC[i,j]= max_delay_urllc
                    total_delay_URLLC[i,j]= max_delay_urllc  
        return Queue_eMBB, Queue_URLLC,trans_delay_eMBB,trans_delay_URLLC,total_delay_eMBB,total_delay_URLLC