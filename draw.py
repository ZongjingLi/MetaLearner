import networkx as nx
import matplotlib.pyplot as plt
import time

class SearchNode:
    def __init__(self, fn, next_nodes=None):
        self.fn = fn  # 节点标识（用于匹配query）
        self.next = next_nodes if next_nodes is not None else []  # 子节点列表
        self.distr = None  # 节点分布值（由effective函数计算）

def effective(query, node: SearchNode, is_in_query_branch: bool):
    # 计算子节点需要继承的状态：当前已在查询分支，或当前节点就是查询节点 → 子节点也在查询分支
    child_in_query_branch = is_in_query_branch or (node.fn == query)
    
    # 先递归遍历所有子节点（深度优先，后序遍历），传递更新后的分支状态
    for son in node.next:
        effective(query, son, child_in_query_branch)
    
    # 后序处理：统一赋值（查询节点/其后代 → 1，其他 → 0）
    if child_in_query_branch or (node.fn == query):
        node.distr = 1
    else:
        node.distr = 0

# 步骤3：定义可视化函数（已修复节点位置错误）
def visualize_search_node_tree(root: SearchNode, query: str):
    # 1. 初始化有向图（树形结构用有向图DiGraph更清晰）
    G = nx.DiGraph()
    
    # 2. 递归遍历所有节点，收集图数据（节点属性、边列表）
    node_attributes = {}  # 存储节点的fn和distr属性
    edge_list = []  # 存储节点间的父子边
    
    def traverse_and_build(node: SearchNode):
        # 记录当前节点属性（用fn作为节点唯一标识）
        node_id = node.fn
        node_attributes[node_id] = {
            "distr": node.distr,
            "is_query_match": node.fn == query
        }
        # 遍历子节点，添加边并递归
        for son in node.next:
            edge_list.append((node_id, son.fn))
            traverse_and_build(son)
    
    # 从根节点开始构建图数据
    traverse_and_build(root)
    
    # 3. 先添加所有节点和边到NetworkX图中（关键：先建图，再算布局）
    G.add_nodes_from(node_attributes.keys())
    G.add_edges_from(edge_list)
    
    # 4. 再计算节点布局（此时图已完整，所有节点都能被分配位置）
    try:
        # 优先使用dot布局（更规整的树形结构，需安装：pip install pygraphviz）
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
    except ImportError:
        # 备选：spring布局（无需额外依赖，结构稍松散）
        print("提示：未安装pygraphviz，使用spring布局替代")
        #pos = nx.spring_layout(G, seed=42, k=2)
        pos = nx.shell_layout(G)
    
    # 5. 自定义节点样式（根据distr值区分颜色）
    node_colors = []
    node_labels = {}  # 节点显示标签：fn + distr值
    for node_id, attrs in node_attributes.items():
        distr = attrs["distr"]
        # 颜色区分：distr=1（查询节点/后代，红/浅红）> distr=0（其他，浅灰）
        if attrs["is_query_match"]:
            node_colors.append("#ff4444")  # 深红红：查询节点本身
        elif distr == 1:
            node_colors.append("#ff9999")  # 浅红色：查询节点的后代
        else:
            node_colors.append("#e0e0e0")  # 浅灰色：其他节点
        # 格式化节点标签
        node_labels[node_id] = f"{node_id}\ndistr={distr}"
    
    # 6. 绘制图形
    plt.figure(figsize=(12, 8))
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8)
    # 绘制边
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, arrowstyle="->", arrowsize=20, width=2)
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")
    # 绘制图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff4444", label=f"Query Node (fn={query}, distr=1)"),
        Patch(facecolor="#ff9999", label="Query Node's Child (distr=1)"),
        Patch(facecolor="#e0e0e0", label="Other Nodes (distr=0)")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(f"SearchNode Tree Visualization (Query: {query})", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# 步骤4：带性能统计的辅助函数（可选）
def effective_with_stats(query, node: SearchNode, is_in_query_branch: bool, stats: dict):
    stats["node_count"] += 1  # 统计遍历节点数量
    start_time = time.perf_counter()
    child_in_query_branch = is_in_query_branch or (node.fn == query)
    for son in node.next:
        effective_with_stats(query, son, child_in_query_branch, stats)
    if child_in_query_branch or (node.fn == query):
        node.distr = 1
    else:
        node.distr = 0
    stats["total_time"] += time.perf_counter() - start_time

# 步骤5：构建测试树形结构 + 执行核心逻辑 + 可视化 + 性能统计
if __name__ == "__main__":
    # 1. 构建一棵测试树（多层结构，包含多个匹配query的节点）
    # 叶子节点
    node_g = SearchNode("G")
    node_h = SearchNode("H")
    node_i = SearchNode("I")
    node_j = SearchNode("J")
    # 中间节点
    node_d = SearchNode("D", [node_g, node_h])
    node_e = SearchNode("E", [node_i])
    node_f = SearchNode("F", [node_j])
    node_b = SearchNode("B", [node_d, node_e])
    node_c = SearchNode("C", [node_f])
    # 根节点
    root = SearchNode("A", [node_b, node_c])

    # 2. 定义查询目标（匹配节点B）
    query_target = "B"

    # 3. 执行修复后的effective函数（根节点无父节点，初始不在查询分支，传False）
    effective(query_target, root, is_in_query_branch=False)

    # 4. （可选）执行带性能统计的版本，输出量化结果
    stats = {"node_count": 0, "total_time": 0.0}
    effective_with_stats(query_target, root, False, stats)
    print("=" * 50)
    print("性能统计结果")
    print("=" * 50)
    print(f"遍历节点总数：{stats['node_count']}")
    print(f"总执行时间：{stats['total_time']:.6f} 秒")
    print("=" * 50)

    # 5. 调用可视化函数，展示结果
    visualize_search_node_tree(root, query_target)