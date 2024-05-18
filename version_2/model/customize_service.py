# !/usr/bin/python
# -*- coding: UTF-8 -*-


import json
from model_service.pytorch_model_service import PTServingBaseService
from collections import defaultdict
import numpy as np
from ortools.sat.python import cp_model
import itertools


class OperationAssignment(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        self.input_key_1 = 'input_json'
        self.output_key_1 = 'output_score'

    def _preprocess(self, data):
        """
        you can add your special preprocess method here
        """

        return data

    def _postprocess(self, data):
        """
        you can add your special postprocess method here
        """
        return data

    def algorithm(self, data):
        # 工序标准时间和工序ID
        ######################################################################################
        oper_time_dict = dict()
        oper_id_dict = dict()
        oper_category_dict = dict()
        for idx, d in enumerate(data["process_list"]):
            oper_id_dict[idx] = d["operation"]
            oper_time_dict[idx] = d["standard_oper_time"]
            oper_category_dict[d["operation"]] = d["operation_category"]
        oper_name_to_id_dict = {i:j for j, i in oper_id_dict.items()}
        ######################################################################################

        # 工人效率值、工人ID和工序类别到工人的映射
        ######################################################################################
        worker_efficiency_dict = dict()
        worker_id_dict = dict()
        category_to_worker_dict = defaultdict(dict)
        for idx, d in enumerate(data["worker_list"]):
            worker_id_dict[idx] = d["worker_code"]
            worker_efficiency_dict[idx] = dict()
            for op in d["operation_skill_list"]:
                key = op["operation_code"]
                efficiency = op["efficiency"]
                worker_efficiency_dict[idx][key] = efficiency
            # 建立同类工序技能表
            for op in d["operation_category_skill_list"]:
                category = op["operation_category"]
                efficiency = op["efficiency"]
                category_to_worker_dict[category][idx] = efficiency
        category_to_worker_dict = dict(category_to_worker_dict)
        ######################################################################################

        # 找到没人会的工序，建立worker完整工序技能表
        ######################################################################################
        worker_oper_set = set()
        for d in data["worker_list"]:
            worker_oper_set |= set([op["operation_code"] for op in d["operation_skill_list"]])
        hard_oper_set = set([d["operation"] for d in data["process_list"]])
        hard_oper_set -= worker_oper_set

        for operation_code in hard_oper_set:
            category = oper_category_dict[operation_code]
            for worker_id, efficiency in category_to_worker_dict[category].items():
                worker_efficiency_dict[worker_id][operation_code] = efficiency
        ######################################################################################

        # 建立机器和工序的对应关系
        ######################################################################################
        oper_to_machine = dict()
        machine_id_dict = dict()
        for op in data["process_list"]:
            oper_num = op["operation"]
            machine = op["machine_type"]
            oper_to_machine[oper_num] = machine

        for idx, machine in enumerate(data["machine_list"]):
            machine_id_dict[idx] = machine["machine_type"]

        machine_type_to_id = {i:j for j, i in machine_id_dict.items()}
        ######################################################################################

        # 建立站台字典
        ######################################################################################
        station_id_dict = dict()
        for station in data["station_list"]:
            line_number = station["line_number"]
            station_code = station["station_code"]
            station_id_dict[line_number] = station_code
        ######################################################################################

        # 加载curr_machine_list
        ######################################################################################
        station_cml = dict()
        for station in data["station_list"]:
            station_code = station["station_code"]
            curr_machine_list = station["curr_machine_list"]
            station_cml[station_code] = [machine["machine_type"] for machine in curr_machine_list]
        ######################################################################################

        # 加载机器工位相关约束
        ######################################################################################
        machine_config = dict()
        for machine in data["machine_list"]:
            machine_type = machine["machine_type"]
            machine_config[machine_type] = dict()
            machine_config[machine_type] = {"is_mono": machine["is_mono"],
                                            "is_movable": machine["is_movable"],
                                            "is_machine_needed": machine["is_machine_needed"]}
        ######################################################################################

        # 建立machine_id 到 oper_id的映射关系
        ######################################################################################
        oper_id_to_machine_type_2 = dict()
        for op in data["process_list"]:
            oper_name = op["operation"]
            oper_id = oper_name_to_id_dict[oper_name]
            oper_id_to_machine_type_2[oper_id] = op["machine_type_2"]

        machine_to_oper = defaultdict(list)
        for oper_name, machine_type in oper_to_machine.items():
            oper_id = oper_name_to_id_dict[oper_name]
            machine_id = machine_type_to_id[machine_type]
            machine_to_oper[machine_id].append(oper_id)

        temp = {}
        for machine_id, oper_ids in machine_to_oper.items():
            temp[machine_id] = defaultdict(list)
            for oper_id in oper_ids:
                machine_type2 = oper_id_to_machine_type_2[oper_id]
                temp[machine_id][machine_type2].append(oper_id)
        
        machine_to_oper = dict()
        for machine_id, groups in temp.items():
            machine_to_oper[machine_id] = list(temp[machine_id].values())
        del temp
        
        oper_id_to_machine_id = dict()
        for operation, machine in oper_to_machine.items():
            oper_id = oper_name_to_id_dict[operation]
            machine_id = machine_type_to_id[machine]
            oper_id_to_machine_id[oper_id] = machine_id
        ######################################################################################

        # 提取先后关系
        ######################################################################################
        part_dict = defaultdict(lambda: list())
        for op in data["process_list"]:
            part_dict[op["part_code"]].append((op["operation"], op["operation_number"]))
        for ops in part_dict.values():
            ops.sort(key=lambda x: x[1])

        class LinkedNode:
            def __init__(self, part_code, oper_id):
                self.category = part_code
                self.oper_id = oper_id
                self.child = None

        part_last_oper = dict()
        table = dict()
        for part_code, ops in part_dict.items():
            # 当前工件的首个工序
            oper_name = ops[0][0]
            oper_id = oper_name_to_id_dict[oper_name]
            prev = LinkedNode(part_code, oper_id)
            table[oper_id] = prev
            # 制作单链表
            for op in ops[1:]:
                oper_id = oper_name_to_id_dict[op[0]]
                node = LinkedNode(part_code, oper_id)
                table[oper_id] = node
                prev.child = node
                prev = node
            part_last_oper[part_code] = prev

        joint_oper_dict = {jo["part_code"]: jo["joint_operation"] for jo in data["joint_operation_list"]}

        # 拼接工序的强先后关系
        for part_code in joint_oper_dict:
            oper_name = joint_oper_dict[part_code]
            oper_id = oper_name_to_id_dict[oper_name]
            next_node = table[oper_id]
            last_node = part_last_oper[part_code]
            last_node.child = next_node

        task_queue = dict()
        for part_code, ops in part_dict.items():
            task_queue[part_code] = []
            for op in ops:
                oper_name = op[0]
                oper_id = oper_name_to_id_dict[oper_name]
                task_queue[part_code].append(oper_id)
            if part_code in joint_oper_dict:
                oper_name = joint_oper_dict[part_code]
                oper_id = oper_name_to_id_dict[oper_name]
                task_queue[part_code].append(oper_id)
        ######################################################################################

        # 工序->员工对应表
        ######################################################################################
        num_workers = len(worker_efficiency_dict)
        worker_skills = dict()
        for i in range(num_workers):
            available_tasks = worker_efficiency_dict[i].keys()
            temp = list()
            for oper_code in available_tasks:
                oper_id = oper_name_to_id_dict[oper_code]
                temp.append(oper_id)
            worker_skills[i] = set(temp)
        ######################################################################################

        # 加载其他约束
        ######################################################################################
        w1 = data["config_param"]["upph_weight"]
        w2 = data["config_param"]["volatility_weight"]
        mmps = data["config_param"]["max_machine_per_station"]
        mspw = data["config_param"]["max_station_per_worker"]
        mcc = data["config_param"]["max_cycle_count"]
        mspo = data["config_param"]["max_station_per_oper"]
        ######################################################################################

        # config
        scale = 2

        # np.set_printoptions(threshold=np.inf)
        # print(costs)

        # Model
        model = cp_model.CpModel()

        # Data
        num_workers = len(worker_efficiency_dict)
        num_tasks = len(oper_time_dict)
        num_machines = len(machine_id_dict)
        num_stations = max(station_id_dict)
        available_stations = set(i-1 for i in station_id_dict.keys())

        costs = np.zeros((num_workers, num_tasks))
        for i in range(num_workers):
            available_tasks = worker_skills[i]
            for j in range(num_tasks):
                if j not in available_tasks:
                    continue
                costs[i,j] = oper_time_dict[j]
        costs = costs.astype(int)

        # Worker-Task Variables
        ######################################################################################
        # 建立工作-任务表征变量
        wt = np.empty((num_workers, num_tasks), dtype=object)
        for i in range(num_workers):
            for j in range(num_tasks):
                wt[i,j] = model.NewBoolVar(f'wt[{i},{j}]')

        # 不会的工序别做
        for i in range(num_workers):
            for j in range(num_tasks):
                if costs[i][j] != 0:
                    continue
                model.AddAbsEquality(0, wt[i,j])

        # Constraints
        # Each worker is assigned to at least one task.
        for i in range(num_workers):
            model.AddAtLeastOne(wt[i,j] for j in range(num_tasks))

        # Each task is assigned to at least one worker.
        for j in range(num_tasks):
            model.AddAtLeastOne(wt[i,j] for i in range(num_workers))

        # Objective
        ## Workers num per operation
        wnpo = np.empty(num_tasks, dtype=object)
        for j in range(num_tasks):
            workers_num = model.NewIntVar(1, num_workers, f"wnpo[{j}]")
            model.Add(workers_num == wt[:,j].sum())
            wnpo[j] = workers_num

        mean_unit_time = np.empty(num_tasks, dtype=object)
        for j in range(num_tasks):
            temp = model.NewIntVar(1, int(1e6), f"mean_unit_time[{j}]")
            # NP-hard problem
            # model.AddDivisionEquality(temp, int(oper_time_dict[j]), wnpo[j])
            model.AddDivisionEquality(temp, int(oper_time_dict[j]), 1)
            mean_unit_time[j] = temp

        ## worker time per task
        wtpt = np.empty((num_workers, num_tasks), dtype=object)
        for i in range(num_workers):
            available_tasks = worker_skills[i]
            for j in range(num_tasks):
                if j not in available_tasks:
                    wtpt[i,j] = 0
                    continue
                wtpt[i,j] = model.NewIntVar(0, int(1e6), f"wtpt[{i},{j}]")

        for i in range(num_workers):
            available_tasks = worker_skills[i]
            for j in range(num_tasks):
                if j not in available_tasks:
                    continue
                operation = oper_id_dict[j]
                efficiency = worker_efficiency_dict[i][operation]
                temp = model.NewIntVar(1, int(1e6), f"temp_wtpt[{i},{j}]")
                model.AddDivisionEquality(temp, 1000*mean_unit_time[j], int(1000*efficiency))
                model.AddMultiplicationEquality(wtpt[i,j], [temp, wt[i,j]])

        workers_time = np.empty(num_workers, dtype=object)
        for i in range(num_workers):
            workers_time[i] = model.NewIntVar(1, int(1e6), f"wt[{i}]_time")
            model.Add(workers_time[i] == wtpt[i].sum())
        ######################################################################################

        # auxiliary variable cwts
        ######################################################################################
        cwts = np.empty((mcc, num_workers, num_tasks, num_stations), dtype=object)
        for z in range(mcc):
            for i in range(num_workers):
                for j in range(num_tasks):
                    for k in range(num_stations):
                        if k not in available_stations:
                            cwts[z,i,j,k] = 0
                            continue
                        cwts[z,i,j,k] = model.NewBoolVar(f'cwts[{z},{i},{j},{k}]')

        oper_to_worker_dict = defaultdict(list)
        for j in range(num_tasks):
            for i in range(num_workers):
                if costs[i,j] != 0:
                    oper_to_worker_dict[j].append(i)

        ## 建立工位和任务的先后顺序
        for task_seq in task_queue.values():
            for j in range(len(task_seq)-1):
                prev_op = task_seq[j]
                next_op = task_seq[j+1]
                prev_op_workers = oper_to_worker_dict[prev_op]
                next_op_workers = oper_to_worker_dict[next_op]
                for x, y in itertools.product(prev_op_workers, next_op_workers):
                    temp = 0
                    for z in range(mcc):
                        for k in available_stations:
                            temp += (num_stations*(mcc-z)-k) * (cwts[z,x,prev_op,k] - cwts[z,y,next_op,k])
                    if x != y:
                        done_wt = model.NewBoolVar(f"done_wt[{x},{prev_op}]")
                        model.Add(cwts[:,x,prev_op].sum() >= 1).OnlyEnforceIf(done_wt)
                        model.Add(cwts[:,x,prev_op].sum() < 1).OnlyEnforceIf(done_wt.Not())
                        model.Add(temp >= 0).OnlyEnforceIf(done_wt)
                    else:
                        model.Add(temp >= 0)

        ## station-worker-task
        swt = np.empty((num_stations, num_workers, num_tasks), dtype=object)
        for i in range(num_stations):
            if i not in available_stations:
                swt[i,...] = 0
                continue
            for j in range(num_workers):
                for k in range(num_tasks):
                    swt[i,j,k] = model.NewBoolVar(f'swt[{i},{j},{k}]')

        ## 建立辅助变量swt和wt的联系
        ###############################################################
        ### swt <==> wt
        for i in range(num_workers):
            for j in range(num_tasks):
                model.Add(wt[i,j] == swt[:,i,j].sum())

        ## 建立辅助变量swt和cwts的联系
        ###############################################################
        for i in range(num_workers):
            for j in range(num_tasks):
                for k in available_stations:
                    model.Add(swt[k,i,j] == cwts[:,i,j,k].sum())

        ## max station per operation
        ###############################################################
        for j in range(num_tasks):
            model.Add(swt[...,j].sum() <= mspo)

        ### swt to sw
        ### 只要工人在工位上做过任务，sw[i][j] = 1
        sw = np.empty((num_stations, num_workers), dtype=object)
        for i in range(num_stations):
            if i not in available_stations:
                sw[i,...] = 0
                continue
            for j in range(num_workers):
                temp_sw = model.NewBoolVar(f'temp_sw[{i},{j}]')
                model.Add(swt[i,j].sum() >= 1).OnlyEnforceIf(temp_sw)
                model.Add(swt[i,j].sum() < 1).OnlyEnforceIf(temp_sw.Not())
                curr_sw = model.NewBoolVar(f'sw[{i},{j}]')
                model.Add(curr_sw == 1).OnlyEnforceIf(temp_sw)
                model.Add(curr_sw == 0).OnlyEnforceIf(temp_sw.Not())
                sw[i,j] = curr_sw

        # Each station is assigned to at most one worker.
        for i in available_stations:
            model.AddAtMostOne(sw[i][j] for j in range(num_workers))

        # Each worker uses at most mspw stations.
        for j in range(num_workers):
            model.Add(sw[:,j].sum() <= mspw)
        ######################################################################################

        # Station-Machine Variables
        ######################################################################################
        sm = np.empty((num_stations, num_machines), dtype=object)
        ## 创建工位-机器变量
        for i in range(num_stations):
            if i not in available_stations:
                sm[i,:] = 0
                continue
            for j in range(num_machines):
                sm[i][j] = model.NewIntVar(0, mmps, f'sm[{i},{j}]')

        ## 当前工位设备初始化;建立独立工位、可移动、辅助的约束
        for i in available_stations:
            line_number = i+1
            station_code = station_id_dict[line_number]
            curr_machine_list = station_cml[station_code]
            curr_machine_nums = defaultdict(lambda: 0)
            for machine_type in curr_machine_list:
                machine_id = machine_type_to_id[machine_type]
                curr_machine_nums[machine_id] += 1
            for j in range(num_machines):
                machine_type = machine_id_dict[j]
                config = machine_config[machine_type]
                if j in curr_machine_nums:
                    model.AddAbsEquality(curr_machine_nums[j], sm[i,j])
                    continue
                if not config["is_movable"]:
                    model.AddAbsEquality(0, sm[i,j])
                    continue
                if config["is_mono"]:
                    temp_var = model.NewBoolVar(f"(mono_sm[{i},{j}])")
                    model.Add(sm[i,j] >= 1).OnlyEnforceIf(temp_var)
                    model.Add(sm[i,j] < 1).OnlyEnforceIf(temp_var.Not())
                    model.Add(sm[i].sum() == 1).OnlyEnforceIf(temp_var)

        ## 建立有关工位设备数量的约束
        ### Each station has mmps machines at most.
        for i in available_stations:
            model.Add(sm[i].sum() <= mmps)
        ######################################################################################

        # auxiliary variable mst
        ######################################################################################
        mst = np.empty((num_machines, num_stations, num_tasks), dtype=object)
        ## 创建机器-工位-任务辅助变量
        for i in range(num_machines):
            for j in range(num_stations):
                if j not in available_stations:
                    mst[i,j,:] = 0
                    continue
                for k in range(num_tasks):
                    mst[i,j,k] = model.NewBoolVar(f'mst[{i},{j},{k}]')

        ms_budget = np.empty((num_machines, num_stations), dtype=object)
        for i in range(num_machines):
            for j in available_stations:
                # 特殊情况
                ###############################################################
                # 如果机器根本就用不着或者可以用手工代替，直接置零
                machine_type = machine_id_dict[i]
                config = machine_config[machine_type]
                if i not in machine_to_oper or not config["is_machine_needed"]:
                    ms_budget[i,j] = 0
                    continue
                ###############################################################
                temp = list()
                for g_id, oper_group in enumerate(machine_to_oper[i]):
                    sub_ms_budget = [mst[i,j][oper_id] for oper_id in oper_group]
                    temp_ms_budget = model.NewBoolVar(f"(temp_ms_budget[{i},{j}])")
                    model.Add(sum(sub_ms_budget) >= 1).OnlyEnforceIf(temp_ms_budget)
                    model.Add(sum(sub_ms_budget) < 1).OnlyEnforceIf(temp_ms_budget.Not())
                    curr_ms_budget = model.NewBoolVar(f"(ms_budget_g{g_id}[{i},{j}])")
                    model.Add(curr_ms_budget == 1).OnlyEnforceIf(temp_ms_budget)
                    model.Add(curr_ms_budget == 0).OnlyEnforceIf(temp_ms_budget.Not())
                    temp.append(curr_ms_budget)
                ms_budget_in_curr_station = model.NewIntVar(0, mmps, f'ms_budget_ics[{i},{j}]')
                model.Add(ms_budget_in_curr_station == sum(temp))
                ms_budget[i,j]= ms_budget_in_curr_station

        ## 建立ms_budget_ms和sm的联系
        ## sm <==> mst
        for i in range(num_machines):
            if i not in machine_to_oper:
                continue
            for j in available_stations:
                model.Add(sm[j,i] >= ms_budget[i,j])

        ## 建立任务和机器的联系
        ### stw => mst
        st = np.empty((num_stations, num_tasks), dtype=object)
        for i in available_stations:
            for j in range(num_tasks):
                temp_st = model.NewBoolVar(f"(temp_st[{i},{j}])")
                model.Add(swt[i,:,j].sum() >= 1).OnlyEnforceIf(temp_st)
                model.Add(swt[i,:,j].sum() < 1).OnlyEnforceIf(temp_st.Not())
                curr_st = model.NewBoolVar(f"(st[{i},{j}])")
                model.Add(curr_st == 1).OnlyEnforceIf(temp_st)
                model.Add(curr_st == 0).OnlyEnforceIf(temp_st.Not())
                st[i][j] = curr_st

        ## 若某工位上完成了某任务，则将该信号传回给mst，方便后续计算工位所需的机器数
        for i in available_stations:
            for j in range(num_tasks):
                machine_id = oper_id_to_machine_id[j]
                if machine_id not in machine_to_oper:
                    continue
                model.Add(mst[machine_id,i,j] == st[i,j])
        ######################################################################################

        # 用于确定顺序
        ######################################################################################
        tord = np.empty(num_tasks, dtype=object)
        for i in range(num_tasks):
            tord[i] = model.NewIntVar(1, num_tasks, f'tord[{i}]')
        
        for task_seq in task_queue.values():
            for j in range(len(task_seq)-1):
                prev_op = task_seq[j]
                next_op = task_seq[j+1]
                model.Add(tord[prev_op] < tord[next_op])
        model.AddAllDifferent(tord)
        ######################################################################################

        # Answer
        ######################################################################################
        sum_time = model.NewIntVar(1, 1000000, "sum_time")
        model.AddAbsEquality(workers_time.sum(), sum_time)
        mean_time = model.NewIntVar(1, 1000000, "mean_time")
        model.AddDivisionEquality(mean_time, sum_time, num_workers)

        # calculate the score1
        ###############################################################
        ## Pseudo maximum time
        max_time = model.NewIntVar(1, 1000000, 'max_time')
        model.AddMaxEquality(max_time, workers_time)
        score1 = model.NewIntVar(1, 1000000, "score1")
        ## "mean*scale_factor"是为了扩大可行域
        ## score1
        model.AddDivisionEquality(score1, scale*mean_time, max_time)

        # calculate the score2
        ###############################################################
        workers_time_power = np.empty(num_workers, dtype=object)
        for i in range(num_workers):
            workers_time_power[i] = model.NewIntVar(1, int(1e12), f"wt[{i}]_time_power")
            model.AddMultiplicationEquality(workers_time_power[i], [workers_time[i], workers_time[i]])

        sum_of_time_power = model.NewIntVar(1, int(1e12), "sotp")
        model.AddAbsEquality(workers_time_power.sum(), sum_of_time_power)
        mean_sotp = model.NewIntVar(1, int(1e12), "mean_sotp")
        model.AddDivisionEquality(mean_sotp, sum_of_time_power, num_workers)
        power_of_mean_time = model.NewIntVar(1, int(1e12), "pomt")
        model.AddMultiplicationEquality(power_of_mean_time, [mean_time, mean_time])
        var_time = model.NewIntVar(1, int(1e12), "var_time")
        model.AddAbsEquality(mean_sotp - power_of_mean_time, var_time)
        std_time = model.NewIntVar(1, int(1e12), "std_time")
        model.AddMultiplicationEquality(var_time, [std_time, std_time])
        score2 = model.NewIntVar(1, int(1e12), "score2")
        ## score2
        model.AddDivisionEquality(score2, scale*std_time, mean_time)

        # volatility_rate
        # for i in range(num_workers):
        #     norm_vol = model.NewIntVar(1, int(1e12), f"norm_vol[{i}]")
        #     temp = model.NewIntVar(1, int(1e12), f"time_dev[{i}]")
        #     model.AddAbsEquality(temp, scale*(workers_time[i]-mean_time))
        #     model.AddDivisionEquality(norm_vol, temp, mean_time)
        #     model.AddLinearConstraint(norm_vol, -scale*int(vr), scale*int(vr))

        # final target
        ###############################################################
        model.Minimize(int(100*w1)*(scale-score1)+int(100*w2)*score2)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Output the answer
        ###############################################################
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            answer = {"dispatch_results": []}
            results = dict()

            station_worker_map = dict()
            for i in available_stations:
                if solver.Value(sw[i].sum()) > 0:
                    one_station_result = dict()
                    line_number = i+1
                    station_code = station_id_dict[line_number]
                    one_station_result["station_code"] = station_code
                    for j in range(num_workers):
                        if solver.BooleanValue(sw[i,j]):
                            worker_code = worker_id_dict[j]
                            one_station_result["worker_code"] = worker_code
                            station_worker_map[i] = j
                            break
                    one_station_result["operation_list"] = list()
                    results[i] = one_station_result

            task_id_dict = dict()
            offset = 1
            for z in range(mcc):
                for i in results:
                    one_station_result = results[i]
                    operation_list = one_station_result['operation_list']
                    worker_id = station_worker_map[i]
                    pending_oper_ids = list()
                    for k in range(num_tasks):
                        if solver.BooleanValue(cwts[z,worker_id,k,i]):
                            if k in task_id_dict:
                                detail = {}
                                operation = oper_id_dict[k]
                                detail["operation"] = operation
                                detail["operation_number"] = task_id_dict[k]
                                operation_list.append(detail)
                            else:
                                pending_oper_ids.append(k)
                    # 站内排序
                    if pending_oper_ids:
                        temp = list()
                        for oper_id in pending_oper_ids:
                            dummy_task_id = solver.Value(tord[oper_id])
                            temp.append((oper_id, dummy_task_id))
                        temp.sort(key=lambda x: x[1])
                        for rel_id, (oper_id, _) in enumerate(temp):
                            task_id_dict[oper_id] = rel_id + offset
                            detail = {}
                            operation = oper_id_dict[oper_id]
                            detail["operation"] = operation
                            detail["operation_number"] = task_id_dict[oper_id]
                            operation_list.append(detail)
                        offset += len(pending_oper_ids)
            for one_station_result in results.values():
                answer["dispatch_results"].append(one_station_result)
        else:
            answer = None
        return answer

    def _inference(self, data):
        """
        model inference function. Here are a inference example,
        We store the results in advance, so we return the output directly based on the file name.
        if you use another model, please modify this function
        """
        for k, v in data.items():
            for input_file_name, file_content in v.items():
                train_data = json.load(file_content)
                output_json = self.algorithm(train_data)

        if output_json is not None:
            result = {'result': output_json}
        else:
            result = {'result': 'calculate result is None'}

        return result
