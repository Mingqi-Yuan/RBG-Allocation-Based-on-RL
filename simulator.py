# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:32:10 2019

@author: tao
"""

import numpy as np
import ue_recv
import os
import pickle
import copy
import datetime

MEAN_UE_ARRIVAL_TIME = 100
CQI_REPORT_PERIOD = 5
RB_NUM = 50
RBG_NUM = int(np.ceil(RB_NUM / 3))
RBG_2_RB = tuple([tuple(range(rbg*3, min(RB_NUM, (rbg+1)*3))) for rbg in range(RBG_NUM)])
MIN_MCS = 1
MAX_MCS = 29
INIT_CQI = 4
OLLA_ENABLE = True
HARQ_NUM = 8
MAX_TX_TIME = 5
HARQ_FEEDBACK_PERIOD = 8
BONUS = 100

temp_1 = 0

def load_av_ue_info(file='ue_info.pkl'):
    assert(os.path.exists(file) == True)
    with open(file, 'rb') as f:
        ues_info = pickle.load(f)

    for ue_idx in range(len(ues_info)-1,-1,-1):
        if ues_info[ue_idx]['rsrp'] < -130:   # if ue_rsrp < -130, the transimission will fail
            ues_info.pop(ue_idx)

    return ues_info

def rbgs_2_rbs(rbgs):  # a mapping from rb to rbg
    rbs = list()
    for rbg in rbgs:
        rbs.extend(RBG_2_RB[rbg])
    return rbs

class HARQ:
    def __init__(self):
        self.tb_series = None
        self.alloc_rbg = list()
        self.mcs = None
        self.tb_size = 0
        self.eff_tb_size = 0
        self.tx_time = list()
        self.rx_time = list()
        self.ack = list()
        self.snr_bler = None

    def clear(self):
        self.__init__()

class UE:
    def __init__(self, ue_id, arrival_tti, av_ue_info, olla_enable):
        '''basic'''
        self.id = ue_id
        self.arrival_tti = arrival_tti
        self.tti = arrival_tti
        self.av_ue_info = copy.deepcopy(av_ue_info)
        self.av_ue_info_idx = 0
        self.rsrp = self.av_ue_info['rsrp']
        self.olla_enable = olla_enable
        self.olla_step = 0.01

        '''state'''
        self.init_buffer = self.av_ue_info['buffer']
        self.before_sched_buffer = self.init_buffer
        self.after_sched_buffer = self.before_sched_buffer
        self.before_sched_avg_thp = 0
        self.after_sched_avg_thp = self.before_sched_avg_thp
        self.olla_offset = 0

        '''cqi'''
        self.cqi_time = None
        self.rb_cqi = [INIT_CQI for i in range(RB_NUM)] # cqi to RBG
        self.wb_cqi = int(np.mean(self.rb_cqi)) # average cqi
        self.cqi_reported_indicator = 0

        '''harq'''
        self.harqs = [HARQ() for i in range(HARQ_NUM)]
        self.before_sched_idle_harq_num = HARQ_NUM
        self.after_sched_idle_harq_num = self.before_sched_idle_harq_num
        self.idle_harq_id = None

        '''schedule'''
        self.rbg_rate = [None for i in range(RBG_NUM)]
        self.rbg_prior = [None for i in range(RBG_NUM)]
        self.alloc_rbg = list()
        self.alloc_rb = list()
        self.alloc_rb_mcs = list()
        self.mcs = None
        self.tb_size = 0
        self.eff_tb_size = 0
        self.retxing_flag = None

        '''statistic'''
        self.itx_times = 0
        self.recv_ack_tbs_size = 0
        self.avg_rate = 0
        self.tbs = list()

    def clear_sched_info(self):
        self.rbg_rate = [None for i in range(RBG_NUM)]
        self.rbg_prior = [None for i in range(RBG_NUM)]
        self.alloc_rbg = list()
        self.alloc_rb = list()
        self.alloc_rb_mcs = list()
        self.mcs = None
        self.tb_size = 0
        self.eff_tb_size = 0
        self.retxing_flag = None

    def calc_rbg_rate_by_cqi(self, rb_cqi):
        rb_rate = [180 * ue_recv.get_se(int(cqi+self.olla_offset*self.olla_enable)) for cqi in rb_cqi]
        rbg_rate = [sum(rb_rate[rb] for rb in RBG_2_RB[rbg]) for rbg in range(RBG_NUM)]
        return rbg_rate

    def calc_rbg_prior(self):
        self.rbg_rate = self.calc_rbg_rate_by_cqi(self.rb_cqi)
        #self.rbg_prior = [rbg_rate / max(10, self.before_sched_avg_thp) for rbg_rate in self.rbg_rate]
        self.rbg_prior = [rbg_rate / max(1, self.avg_rate) for rbg_rate in self.rbg_rate]

    def update_sched_info(self, retx_flag, harq_id=None):
        if retx_flag:
            self.alloc_rbg = self.harqs[harq_id].alloc_rbg
            self.alloc_rb = rbgs_2_rbs(self.alloc_rbg)
            self.mcs = self.harqs[harq_id].mcs
            self.tb_size = self.harqs[harq_id].tb_size
            self.eff_tb_size = self.harqs[harq_id].eff_tb_size
            self.retxing_flag = True
            self.after_sched_buffer = self.before_sched_buffer
            self.after_sched_avg_thp = self.before_sched_avg_thp
        else:
            harq_id = self.idle_harq_id
            assert(harq_id is not None)

            self.after_sched_idle_harq_num = self.before_sched_idle_harq_num - 1

            self.alloc_rb = rbgs_2_rbs(self.alloc_rbg)
            self.alloc_rb_mcs = [self.rb_cqi[rb] for rb in self.alloc_rb]
            self.mcs = int(np.mean(self.alloc_rb_mcs))
            self.mcs = int(self.mcs + self.olla_offset * self.olla_enable)
            self.mcs = max(MIN_MCS, min(MAX_MCS, self.mcs))
            self.tb_size = int(180 * ue_recv.get_se(self.mcs) * len(self.alloc_rb))
            self.eff_tb_size = min(self.tb_size, self.before_sched_buffer)
            self.retxing_flag = False
            self.itx_times += 1

            self.harqs[harq_id].tb_series = self.itx_times
            self.harqs[harq_id].alloc_rbg = self.alloc_rbg
            self.harqs[harq_id].mcs = self.mcs
            self.harqs[harq_id].tb_size = self.tb_size
            self.harqs[harq_id].eff_tb_size = self.eff_tb_size
            self.harqs[harq_id].snr_bler = ue_recv.get_snr_bler(self.harqs[harq_id].mcs, self.harqs[harq_id].tb_size)

            self.after_sched_buffer = self.before_sched_buffer - self.eff_tb_size
            assert(self.after_sched_buffer >= 0)
            self.after_sched_avg_thp = 0.005 * self.tb_size + (1.0-0.005) * self.before_sched_avg_thp

        self.harqs[harq_id].tx_time.append(self.tti)
        self.harqs[harq_id].rx_time.append(None)
        self.harqs[harq_id].ack.append(None)

    def update_cqi(self):
        if self.cqi_time != None:
            prev_cqi_time = self.cqi_time
        else:
            prev_cqi_time = self.arrival_tti

        if self.tti - prev_cqi_time >= CQI_REPORT_PERIOD:
            if np.all(np.array(self.av_ue_info['cqi']['rb_cqi'][self.av_ue_info_idx]) < 0) == True:
                # if all cqi == -1, do not update cqi
                return

            self.cqi_time = self.tti
            self.rb_cqi = self.av_ue_info['cqi']['rb_cqi'][self.av_ue_info_idx]
            self.wb_cqi = int(np.mean(self.rb_cqi))

            if self.cqi_reported_indicator == 0:
                self.cqi_reported_indicator == 1

    def recv(self):
        for harq_id, harq in enumerate(self.harqs):
            if harq.tb_size == 0 or self.tti - harq.tx_time[-1] < HARQ_FEEDBACK_PERIOD - 1:
                continue

            snr = self.av_ue_info['snr'][self.av_ue_info_idx][1]
            self.av_ue_info_idx = (self.av_ue_info_idx + 1) % len(self.av_ue_info['snr'])
            harq.rx_time[-1] = self.tti

            harq.ack[-1] = ue_recv.is_ack(harq.mcs, harq.tb_size, snr)
            self.recv_ack_tbs_size += harq.ack[-1] * harq.eff_tb_size
            self.avg_rate = self.recv_ack_tbs_size / max(1, (self.tti - self.arrival_tti))

    def update_harq(self):
        self.before_sched_idle_harq_num = 0
        self.idle_harq_id = None
        for harq_id, harq in enumerate(self.harqs):
            if harq.tb_size > 0:
                if harq.ack[-1] == True:
                    self.tbs.append({'harq_id': harq_id, \
                                     'alloc_rbg': harq.alloc_rbg, \
                                     'mcs': harq.mcs, \
                                     'tb_size': harq.tb_size, \
                                     'eff_tb_size': harq.eff_tb_size, \
                                     'tx_time': harq.tx_time, \
                                     'rx_time': harq.rx_time, \
                                     'ack': harq.ack})
                    harq.clear()
                    self.olla_offset += self.olla_step
                elif harq.ack[-1] == False:
                    if len(harq.ack) == MAX_TX_TIME:
                        self.tbs.append({'harq_id': harq_id, \
                                         'alloc_rbg': harq.alloc_rbg, \
                                         'mcs': harq.mcs, \
                                         'tb_size': harq.tb_size, \
                                         'eff_tb_size': harq.eff_tb_size, \
                                         'tx_time': harq.tx_time, \
                                         'rx_time': harq.rx_time, \
                                         'ack': harq.ack})
                        harq.clear()
                    self.olla_offset -= self.olla_step * 9
            if harq.tb_size == 0:
                self.before_sched_idle_harq_num += 1
                self.idle_harq_id = harq_id if self.idle_harq_id == None else self.idle_harq_id

class BS():
    def __init__(self):
        self.is_rl_sched = True
        self.reset()

    def reset(self):
        self.ues = list()
        self.rbg_ue = [None for _ in range(RBG_NUM)]
        self.rbg_ue_iidx = [None for _ in range(RBG_NUM)]
        self.newtx_rbg_ue = [None for _ in range(RBG_NUM)]
        self.retx_rbg_ue = [None for _ in range(RBG_NUM)]
        self.to_newtx_ues_idx = list()
        self.to_retx_ues_idx = list()
        self.pf_reward = 0

    @property
    def state(self):   # need to delete some feature
        assert(len(self.ues) > 0)
        s = np.zeros((len(self.ues), 21))
        for ue_idx, ue in enumerate(self.ues):
            s[ue_idx,0] = ue.rsrp
            s[ue_idx,1] = ue.before_sched_buffer
            s[ue_idx,2] = ue.cqi_reported_indicator
            s[ue_idx,3] = ue.avg_rate
            s[ue_idx,4:21] = np.array(ue.rbg_rate) # mapping to ue.cqi
        return s

    def get_reward(self, ues_eff_tb_size=None):  # modify reward
        '''reward'''
        if ues_eff_tb_size == None:
            ues_eff_tb_size = [ue.eff_tb_size for ue in self.ues]
        ues_avg_rate = [ue.avg_rate for ue in self.ues]
        eff_tb_size_sum = sum(ues_eff_tb_size)
        worst_ue_idx = ues_avg_rate.index(min(ues_avg_rate))
        worst_ue_rbg_num = len(self.ues[worst_ue_idx].alloc_rbg)
#         worst_ue_tb_size = ues_eff_tb_size[worst_ue_idx]
        # define worst_ue_reward in another way
        if ues_eff_tb_size[worst_ue_idx] > 0 :
            reward = eff_tb_size_sum + BONUS * worst_ue_rbg_num
        else:
            reward = eff_tb_size_sum
        
        return reward

    def schedule_retx(self):
        for ue_idx in self.to_retx_ues_idx:
            ue = self.ues[ue_idx]
            for harq_id, harq in enumerate(ue.harqs):
                '''1.1. 找到需要重传的harq'''
                if harq.tb_size == 0 or harq.rx_time[-1] == None:
                    continue

                '''1.2. 检查harq需要的所有rbg是否空闲'''
                rbg_idle_flag = [self.rbg_ue[rbg] == None for rbg in harq.alloc_rbg]
                assert(False not in rbg_idle_flag)
                if False in rbg_idle_flag:
                    continue

                '''1.3. 更新调度结果'''
                for rbg in harq.alloc_rbg:
                    self.rbg_ue[rbg] = ue_idx
                    self.retx_rbg_ue[rbg] = ue_idx
                ue.update_sched_info(retx_flag=True, harq_id=harq_id)

    def schedule_newtx_part1(self):  # PF schedule
        ue_rbg_prior = np.full((len(self.to_newtx_ues_idx), RBG_NUM), np.nan)
        ue_buffer = np.full((len(self.to_newtx_ues_idx),), np.nan)
        '''1. 计算每个ue的每个RBG的rbg_rate'''
        for ue_iidx, ue_idx in enumerate(self.to_newtx_ues_idx):
            ue = self.ues[ue_idx]
            assert(ue.before_sched_buffer > 0) # make sure only schedule ues with buffer
            assert(ue.idle_harq_id is not None)
            if ue.retxing_flag == True:
                ue_rbg_prior[ue_iidx,:] = -1
                ue_buffer[ue_iidx] = -1
            else:
                ue_rbg_prior[ue_iidx,:] = np.array(ue.rbg_prior)
                ue_buffer[ue_iidx] = ue.before_sched_buffer
        assert(np.any(np.isnan(ue_rbg_prior)) == False)
        assert(np.any(np.isnan(ue_buffer)) == False)

        '''2. RBG优先级排序'''
        sort_rbg_idx = np.argsort(np.max(ue_rbg_prior, axis=0))[::-1]

        '''3. 遍历rbg，选出优先级最高的ue'''
        for rbg in sort_rbg_idx:
            '''3.1. 跳过已分配的rbg和没有优先级的rbg'''
            if self.retx_rbg_ue[rbg] != None or \
               np.all(ue_rbg_prior[:,rbg] < 0) == True:
                continue

            assert(self.rbg_ue[rbg] == None)

            '''3.2. 选出prior最大的ue'''
            ue_iidx = np.argmax(ue_rbg_prior[:,rbg])
            assert(ue_rbg_prior[ue_iidx,rbg] > 0)
            ue_idx = self.to_newtx_ues_idx[ue_iidx]
            self.rbg_ue[rbg] = ue_idx
            self.newtx_rbg_ue[rbg] = ue_idx
            self.rbg_ue_iidx[rbg] = ue_iidx
            self.ues[ue_idx].alloc_rbg.append(rbg)

            ''' 3.3. 分配后，如果ue的buffer已空，就把ue_prior设置为-1'''
            ue_buffer[ue_iidx] -= int(self.ues[ue_idx].rbg_rate[rbg])
            if ue_buffer[ue_iidx] <= 0:
                ue_rbg_prior[ue_iidx,:] = -1

        '''4. 计算PF的reward'''
        ues_eff_tb_size = list()
        for ue_idx, ue in enumerate(self.ues):
            if len(ue.alloc_rbg) > 0:
                alloc_rb = rbgs_2_rbs(ue.alloc_rbg)
                alloc_rb_mcs = [ue.rb_cqi[rb] for rb in alloc_rb]
                mcs = int(np.mean(alloc_rb_mcs))
                mcs = int(mcs + ue.olla_offset * ue.olla_enable)
                mcs = max(MIN_MCS, min(MAX_MCS, mcs))
                tb_size = int(180 * ue_recv.get_se(mcs) * len(alloc_rb))
                eff_tb_size = min(tb_size, ue.before_sched_buffer)
            else:
                eff_tb_size = 0
            ues_eff_tb_size.append(eff_tb_size)
        self.pf_reward = self.get_reward(ues_eff_tb_size)

    def schedule_newtx_part2(self, rl_rbg_ue):
        rl_punish = 0
        if rl_rbg_ue is None:
            return rl_punish

        self.rbg_ue_iidx = [None for i in range(RBG_NUM)]
        for rbg_idx, (pf_ue_idx, rl_ue_idx) in enumerate(zip(self.newtx_rbg_ue, rl_rbg_ue)):
            if self.retx_rbg_ue[rbg_idx] is not None:
                continue
            assert(rl_ue_idx is not None)
            if (rl_ue_idx not in self.to_newtx_ues_idx) or (rl_ue_idx in self.retx_rbg_ue):
                rl_punish += 500
                continue
            self.rbg_ue[rbg_idx] = rl_ue_idx
            self.newtx_rbg_ue[rbg_idx] = rl_ue_idx
            if pf_ue_idx is not None:
                self.ues[pf_ue_idx].alloc_rbg.remove(rbg_idx)
            self.ues[rl_ue_idx].alloc_rbg.append(rbg_idx)
        return rl_punish

    def schedule_newtx_part3(self):
        '''check newtx and retx'''
        for ue_idx in set(self.newtx_rbg_ue):
            if ue_idx is None:
                continue
            assert(ue_idx not in self.retx_rbg_ue)
        for ue_idx in set(self.retx_rbg_ue):
            if ue_idx is None:
                continue
            assert(ue_idx not in self.newtx_rbg_ue)

        '''4. 更新ue的调度结果'''
        for ue_idx in set(self.newtx_rbg_ue):
            if ue_idx is None:
                continue
            ue = self.ues[ue_idx]
            ue.update_sched_info(retx_flag=False)

    def schedule_init(self):
        self.rbg_ue = [None for _ in range(RBG_NUM)]
        self.rbg_ue_iidx = [None for _ in range(RBG_NUM)]
        self.newtx_rbg_ue = [None for _ in range(RBG_NUM)]
        self.retx_rbg_ue = [None for _ in range(RBG_NUM)]
        self.to_newtx_ues_idx = list()
        self.to_retx_ues_idx = list()
        self.pf_reward = 0

        for ue_idx, ue in enumerate(self.ues):
            ue.clear_sched_info()
            ue.before_sched_buffer = ue.after_sched_buffer
            ue.before_sched_avg_thp = ue.after_sched_avg_thp

            ue.update_harq()
            ue.calc_rbg_prior()

            for harq in ue.harqs:
                if harq.tb_size > 0 and harq.ack[-1] != None:
                    assert(harq.ack[-1] == False)
                    self.to_retx_ues_idx.append(ue_idx)
                    break
            if ue.before_sched_buffer > 0 and ue.idle_harq_id is not None:
                self.to_newtx_ues_idx.append(ue_idx)

    def update_cqi(self):
        for ue in self.ues:
            ue.update_cqi()

class AIRVIEW:
    def __init__(self):
        self.tti = 0
        self.av_ues_info = list()
        self.av_ues_idx = list()
        self.ue_num = 0
        self.bs = BS()
        self.history = dict()
        self.result = None

    def reset(self, av_ues_info, av_ues_idx):
        self.tti = 0
        self.av_ues_info = av_ues_info
        self.av_ues_idx = av_ues_idx
        self.ue_num = len(av_ues_idx)
        self.bs.reset()
        self.history = dict()
        self.result = None

        self.new_ue_arrive()

        self.bs.update_cqi()
        self.bs.schedule_init()
        if len(self.bs.to_retx_ues_idx) > 0:
            self.bs.schedule_retx()

        if len(self.bs.to_newtx_ues_idx) > 0:
            self.bs.schedule_newtx_part1()

        return self.bs.state

    def new_ue_arrive(self):
        '''new ue arrive'''
        for av_ue_idx in self.av_ues_idx:
            ue_info = self.av_ues_info[av_ue_idx]
            new_ue = UE(len(self.bs.ues), self.tti, ue_info, OLLA_ENABLE)
            self.bs.ues.append(new_ue)

    def get_result(self):
        ues_rate = list()
        for ue in self.bs.ues:
            last_rx_time = 0
            recv_ack_tbs_size = 0
            for tb in ue.tbs:
                recv_ack_tbs_size += tb['eff_tb_size'] * tb['ack'][-1]
                last_rx_time = max(last_rx_time, tb['rx_time'][-1])
            ues_rate.append(recv_ack_tbs_size / (last_rx_time - ue.arrival_tti) if last_rx_time != 0 else 0)
        avg_ue_rate = np.mean(ues_rate)
        worst_ue_rate = min(ues_rate)
        return {'avg_ue_rate': avg_ue_rate, 'worst_ue_rate': worst_ue_rate}

    def step(self, action):
        done_flag = False
        while True:
            '''bs sched'''
            if done_flag == False:
                rl_punish = self.bs.schedule_newtx_part2(action) # vs. self.get_reward() ???

            self.bs.schedule_newtx_part3()

            if done_flag == False:
         #      reward = self.bs.get_reward() - rl_punish - self.bs.pf_reward
                if rl_punish > 0:
                    reward = -1
                else: 
                    reward = self.bs.get_reward()

            '''ue recv'''
            for ue in self.bs.ues:
                ue.recv()
    
            with open('simulator.txt', 'a') as pf:
                pf.write('tti:%d\r\n' %(self.tti))
                for ue in self.bs.ues:
                    pf.write('ue:%d, rsrp:%.0f, buffer:[%d,%d], rate:%.0e, after_sched_avg_thp:%.1f, idle_harq_num:%d' \
                             %(ue.id, ue.rsrp, ue.before_sched_buffer, ue.after_sched_buffer, \
                               ue.avg_rate, ue.after_sched_avg_thp, ue.before_sched_idle_harq_num))
                    pf.write('\r\n      cqi:')
                    for cqi in ue.rb_cqi:
                        pf.write('%.d, ' %(cqi))
                    pf.write('\r\n      prior:')
                    for prior in ue.rbg_prior:
                        pf.write('%.1f, ' %(prior))
                    pf.write('\r\n      alloc_rbg:')
                    for rbg in ue.alloc_rbg:
                        pf.write('%d, ' %(rbg))
                    pf.write('\r\n      olla=%.2f, mcs=%s, tb_size=%d, eff_tb_size=%d, retx=%s\r\n' \
                             %(ue.olla_offset, ue.mcs, ue.tb_size, ue.eff_tb_size, ue.retxing_flag))

            self.tti += 1
            self.bs.tti = self.tti
            for ue in self.bs.ues:
                ue.tti = self.tti

            self.bs.update_cqi()

            self.bs.schedule_init()

            if len(self.bs.to_retx_ues_idx) > 0:
                self.bs.schedule_retx()

            if len(self.bs.to_newtx_ues_idx) > 0:
                self.bs.schedule_newtx_part1()

            if done_flag == False:
                state = self.bs.state

            done_flag |= self.is_done()
#            if done_flag == True:
#                self.bs.to_newtx_ues_idx = list()
#                for ue in self.bs.ues:
#                    ue.before_sched_buffer = 0


            exit_flag = self.is_exit()

            if done_flag == False or exit_flag == True:
                break

        return state, reward, done_flag, {}

    def is_exit(self):
        # for tti in range(MAX_TTI):
        #     self.step(tti)

        ues_buffers = list()
        for ue_idx, ue in enumerate(self.bs.ues):
            harq_buffer = sum([harq.tb_size for harq in ue.harqs])
            ues_buffers.append(harq_buffer + ue.before_sched_buffer)
        if sorted(ues_buffers)[-2] == 0: # only one user in the system
            return True
            # break
        return False

    def is_done(self):
        # for tti in range(MAX_TTI):
        #     self.step(tti)

        ues_buffers = list()
        for ue_idx, ue in enumerate(self.bs.ues):
            harq_buffer = sum([harq.tb_size for harq in ue.harqs])
            ues_buffers.append(harq_buffer + ue.before_sched_buffer)
        if sorted(ues_buffers)[-2] == 0: # only one user in the system
            return True
#        for ue in self.bs.ues:
#            if ue.idle_harq_id is None:
#                return True
#        if len(self.bs.to_retx_ues_idx) > 0:
#            return True

        return False


                