# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:25:05
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-20 08:59:38
import torch
import torch.nn as nn

class PrimitiveType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class ComplexType:
    def __init__(self, left, right, direction):
        self.left = left
        self.right = right
        self.direction = direction

    def __repr__(self):
        return f"{self.left}{self.direction}{self.right}"

class DSLFunction:
    def __init__(self, name, input_types, output_type):
        self.name = name
        self.input_types = input_types
        self.output_type = output_type

    def __repr__(self):
        return self.name


# 定义词汇表条目，使用可训练的权重
class LexicalEntry(nn.Module):
    def __init__(self, word, program, syntactic_type, initial_weight=1.0):
        super(LexicalEntry, self).__init__()
        self.word = word
        self.program = program
        self.syntactic_type = syntactic_type
        self.weight = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float32))

    def forward(self):
        return self.weight

    def __repr__(self):
        return f"{self.word}: {self.program}, {self.syntactic_type}, weight={self.weight.item()}"


# 枚举候选词汇表条目
def enumerate_lexicon_entries(words, dsl_functions):
    lexicon_entries = []
    for word in words:
        for func in dsl_functions:
            # 简单假设这里不考虑参数顺序，实际中需要枚举
            syntactic_type = ComplexType(func.output_type, func.input_types[0], '/') if func.input_types else func.output_type
            entry = LexicalEntry(word, func, syntactic_type)
            lexicon_entries.append(entry)
    return lexicon_entries


# 改进的CKY算法，计算解析概率
def cky_parse(sentence, lexicon_entries):
    n = len(sentence)
    chart = [[[] for _ in range(n)] for _ in range(n)]

    # 初始化对角线元素
    for i in range(n):
        word = sentence[i]
        for entry in lexicon_entries:
            if entry.word == word:
                # 初始概率等于词汇表条目的权重
                weight = entry()
                chart[i][i].append((entry, weight))

    # 填充图表
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for left, left_prob in chart[i][k]:
                    for right, right_prob in chart[k + 1][j]:
                        if isinstance(left.syntactic_type, ComplexType) and left.syntactic_type.right == right.syntactic_type:
                            new_type = left.syntactic_type.left
                            new_program = (left.program, right.program)  # 简单组合程序
                            new_entry = LexicalEntry(None, new_program, new_type)
                            # 计算新的解析概率，假设概率相乘
                            new_prob = left_prob * right_prob
                            chart[i][j].append((new_entry, new_prob))

    return chart


# 获取最终解析结果和概率
def get_final_parses(chart):
    n = len(chart)
    final_parses = []
    for entry, prob in chart[0][n - 1]:
        final_parses.append((entry, prob))
    return final_parses


# 主程序示例
if __name__ == "__main__":
    # 示例原始类型
    object_set_type = PrimitiveType('ObjectSet')
    single_object_type = PrimitiveType('SingleObject')

    # 示例DSL函数
    func1 = DSLFunction('filter_red', [object_set_type], object_set_type)
    func2 = DSLFunction('select', [object_set_type], single_object_type)
    dsl_functions = [func1, func2]

    # 示例句子
    sentence = ['red', 'select']

    # 枚举词汇表条目
    lexicon_entries = enumerate_lexicon_entries(sentence, dsl_functions)
    print("Lexicon Entries:")
    for entry in lexicon_entries:
        print(entry)

    # 执行CKY解析
    chart = cky_parse(sentence, lexicon_entries)

    # 获取最终解析结果和概率
    final_parses = get_final_parses(chart)
    print("\nFinal Parses and Probabilities:")
    for entry, prob in final_parses:
        print(f"Parse: {entry.program}, Probability: {prob.item()}")

    # 示例优化过程
    optimizer = torch.optim.SGD([entry.weight for entry in lexicon_entries], lr=0.01)
    # 假设一个简单的损失，这里用概率的负值作为损失
    loss = -sum([prob for _, prob in final_parses])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("\nUpdated Lexicon Entries:")
    for entry in lexicon_entries:
        print(entry)