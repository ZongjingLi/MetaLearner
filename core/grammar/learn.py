import torch
import torch.nn as nn
import itertools
from typing import Dict, List, Tuple, Set, Optional, Any
try:
    from .lexicon import SemProgram, CCGSyntacticType, LexiconEntry
except:
    from lexicon import SemProgram, CCGSyntacticType, LexiconEntry
from helchriss.dsl.dsl_types import TypeBase, INT, BOOL

from itertools import permutations, product

def enumerate_types(arg_types : List[TypeBase], ret_type : TypeBase) -> List[CCGSyntacticType]:
    results = []
    if not arg_types: return [CCGSyntacticType(str(ret_type))]

    for arg_perm in permutations(arg_types):
        for directions in product(['/', '\\'], repeat=len(arg_perm)):
            current_type = CCGSyntacticType(ret_type)
            for i in reversed(range(len(arg_perm))):
                param_type = arg_perm[i]
                direction = directions[i]
                
                param_syn_type = CCGSyntacticType(param_type)

                current_type = CCGSyntacticType(
                    f"{current_type}{direction}{param_syn_type}",
                    param_syn_type,
                    current_type,
                    direction
                )
            results.append(current_type)
    return results

def enumerate_search(related_types: List[TypeBase], related_funcs: Dict, max_depth: int = 1) -> List:
    results: Set[Tuple[CCGSyntacticType, SemProgram, int]] = set()

    def dfs(current_nodes: List[Tuple[CCGSyntacticType, SemProgram]], depth: int):
        # 终止条件：超过最大深度则停止
        if depth > max_depth:
            return
        
        # 1. 将当前节点加入结果集（仅保留有效深度的节点）
        for syn_type, program in current_nodes:
            results.add((syn_type, program, depth))
        
        # 2. 生成新节点：函数应用与组合
        new_nodes = []
        
        # 2.1 引入基础函数（深度0时初始化）
        if depth == 0:
            for fn, func_info in related_funcs.items():
                params = func_info["parameters"]
                ret_type = func_info["type"]
                # 生成该函数的所有可能CCG类型
                syn_types = enumerate_types(params, ret_type)
                # 初始化语义程序：带lambda变量（数量=参数数）
                lambda_vars = [f"x{i}" for i in range(len(params))]
                sem_program = SemProgram(func_name=fn, args=[], lambda_vars=lambda_vars)
                # 加入初始节点
                for syn_type in syn_types:
                    new_nodes.append((syn_type, sem_program))
        
        # 2.2 函数应用：将现有节点组合（处理CCG类型的斜线消除）
        for (func_syn, func_prog) in current_nodes:
            # 仅处理复合类型（可应用的函数）
            if func_syn.is_primitive:
                continue  # 基本类型无法应用其他参数
            
            # 复合类型：X/Y 或 X\Y，需要找到匹配的参数类型
            target_arg_type = func_syn.arg_type
            direction = func_syn.direction
            
            # 遍历所有可能作为参数的节点
            for (arg_syn, arg_prog) in current_nodes:
                if arg_syn == target_arg_type:
                    # 类型匹配，执行应用：消除对应的斜线
                    new_syn_type = func_syn.result_type  # 应用后剩余的类型
                    
                    # 处理语义程序：绑定一个lambda变量，清理已绑定的变量
                    # 函数原有lambda变量：[x0, x1, ..., xn-1]，应用1个参数后，剩余[x1, ..., xn-1]
                    if len(func_prog.lambda_vars) == 0:
                        continue  # 无未绑定变量，无法再应用参数
                    
                    # 移除第一个lambda变量（按顺序绑定），添加参数到args
                    new_lambda_vars = func_prog.lambda_vars[1:]  # 清理已绑定的变量
                    new_args = func_prog.args + [arg_prog]       # 新增绑定的参数
                    
                    new_sem_prog = SemProgram(
                        func_name=func_prog.func_name,
                        args=new_args,
                        lambda_vars=new_lambda_vars
                    )
                    
                    new_nodes.append((new_syn_type, new_sem_prog))
        
        # 3. 递归搜索下一层（深度+1）
        if new_nodes:
            dfs(new_nodes, depth + 1)

    # 初始调用：从深度0开始，初始节点为空（由深度0时引入基础函数）
    dfs([], 0)
    
    # 过滤掉仍有未绑定lambda变量的中间结果（可选，根据需求）
    # 最终结果只保留完全绑定（无lambda变量）或达到最大深度的节点
    filtered_results = [
        (syn, prog, d) for (syn, prog, d) in results 
        if  d <= max_depth

    ]
    #        if len(prog.lambda_vars) == 0 or d == max_depth
    
    return filtered_results

def _enumerate_search(related_types: List[TypeBase], related_funcs: Dict, max_depth: int) -> List[List]:
    results: Set[Tuple[CCGSyntacticType, SemProgram]] = set()
    
    current_depth_results: List[Tuple[CCGSyntacticType, SemProgram]] = []

    for depth in range(1, max_depth + 1):
        next_depth_results: List[Tuple[CCGSyntacticType, SemProgram]] = []

        for func_key, func_info in related_funcs.items():
            func_params = func_info["parameters"]  # List[TypeBase]
            func_ret_type = func_info["type"]      # TypeBase
            param_count = len(func_params)
            
            # add nullary functions to the results
            if param_count == 0:
                ccg_type = CCGSyntacticType(name=func_ret_type, arg_type = None, result_type = func_ret_type)
                sem_program = SemProgram(
                    func_name=func_key,
                    args = [], lambda_vars=[]
                )
                next_depth_results.append((ccg_type, sem_program))
                results.add((ccg_type, sem_program))

            # 情况2：有参数函数（处理单参数和多参数，支持`/`和`\`方向）
            else:
                # 单参数函数：同时生成`/`（向前）和`\`（向后）方向的类型
                if param_count == 1:
                    param_type = func_params[0]  # 函数要求的参数类型
                    ret_type = func_ret_type      # 函数返回类型
                    
                    for (prev_ccg, prev_sem) in current_depth_results:
                        # 检查当前类型是否匹配函数参数类型（严格匹配TypeBase）
                        if prev_type != param_type: continue  # 类型不匹配，跳过
                        
                        # --------------------------
                        # 生成`/`方向的CCG类型（函数在前，参数在后：ret_type / param_type）
                        # --------------------------
                        ccg_forward = CCGSyntacticType(
                            name=f"{ret_type}/{param_type}",
                            arg_type=prev_ccg,  # 参数的CCG类型
                            result_type=CCGSyntacticType(name=ret_type),
                            direction="/"
                        )
                        # 语义程序：函数应用于参数，参数类型已匹配，返回类型为ret_type
                        sem_forward = SemProgram(
                            func_name=func_key,
                            args=[prev_sem],  # 传入参数的语义程序（类型已验证）
                            lambda_vars=[]
                        )
                        next_depth_results.append((ccg_forward, sem_forward, ret_type))
                        results.add((ccg_forward, sem_forward))
                        
                        # --------------------------
                        # 生成`\`方向的CCG类型（参数在前，函数在后：param_type \ ret_type）
                        # --------------------------
                        ccg_backward = CCGSyntacticType(
                            name=f"{param_type}\\{ret_type}",
                            arg_type=prev_ccg,  # 参数的CCG类型（此时参数在前）
                            result_type=CCGSyntacticType(name=ret_type),
                            direction="\\"
                        )
                        # 语义程序：参数应用于函数（CCG向后应用，逻辑上等价但顺序不同）
                        sem_backward = SemProgram(
                            func_name=func_key,
                            args=[prev_sem],  # 参数在前，函数在后的应用
                            lambda_vars=[]
                        )
                        next_depth_results.append((ccg_backward, sem_backward, ret_type))
                        results.add((ccg_backward, sem_backward))
                
                # 多参数函数：通过嵌套单参数函数实现，同时支持`/`和`\`方向
                elif param_count > 1:
                    first_param = func_params[0]
                    remaining_params = func_params[1:]
                    ret_type = func_ret_type
                    
                    for (prev_ccg, prev_sem, prev_type) in current_depth_results:
                        if prev_type != first_param:
                            continue  # 类型不匹配，跳过
                        
                        # 生成中间函数的返回类型字符串（剩余参数的函数类型）
                        remaining_str = "".join([f"/{p.typename}" for p in remaining_params])
                        middle_ret_type_str = f"({ret_type.typename}{remaining_str})"
                        
                        # --------------------------
                        # 生成`/`方向的嵌套类型
                        # --------------------------
                        ccg_forward = CCGSyntacticType(
                            name=f"{middle_ret_type_str}/{first_param.typename}",
                            arg_type=prev_ccg,
                            result_type=CCGSyntacticType(name=middle_ret_type_str),
                            direction="/"
                        )
                        sem_forward = SemProgram(
                            func_name=func_key,
                            args=[prev_sem],  # 绑定第一个参数
                            lambda_vars=[f"x{i}" for i in range(len(remaining_params))]  # 剩余参数变量
                        )
                        next_depth_results.append((ccg_forward, sem_forward, ret_type))
                        results.add((ccg_forward, sem_forward))

                        # --------------------------
                        # 生成`\`方向的嵌套类型
                        # --------------------------
                        ccg_backward = CCGSyntacticType(
                            name=f"{first_param}\\{middle_ret_type_str}",
                            arg_type=prev_ccg,
                            result_type=CCGSyntacticType(name=middle_ret_type_str),
                            direction="\\"
                        )
                        sem_backward = SemProgram(
                            func_name=func_key,
                            args=[prev_sem],  # 第一个参数在前，函数在后
                            lambda_vars=[f"x{i}" for i in range(len(remaining_params))]
                        )
                        next_depth_results.append((ccg_backward, sem_backward, ret_type))
                        results.add((ccg_backward, sem_backward))
        
        # update the current depth results of the CCG Program tuples
        current_depth_results = list({(ccg, sem) for ccg, sem in next_depth_results})

    return [[ccg, sem] for ccg, sem in results]