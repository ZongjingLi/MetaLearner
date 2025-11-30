# @Author: Yiqi Sun
# @Create Time: 2025-11-27 07:55:05
# @Modified by: Yiqi Sun
# @Modified time: 2025-11-28 13:51:20
import regex as re
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Callable, Union, Optional, Dict, Any
from core.model import MetaLearner
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import json

@dataclass
class LearningTask:
    """Dataclass to store learning task metadata (vocab, domains, dataset loading code)."""
    name: str
    vocab: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    train: str = None
    test : str = None
    val  : str = None
    code : str = None

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        return (
            f"LearningTask(name='{self.name}', vocab={self.vocab[:5]}...,"
            f" domains={self.domains}, train='{self.train[:30]}...')"
        )


class GroundingTask(LearningTask):
    """ grounding a vocabulary of concepts"""
    pass

class PlanningTask(LearningTask):
    """grounding a planenr and inherent model and learn actions"""
    pass

class DependencyAutomata:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for FSM
        self.node_counter = 0  # Unique ID generator for nodes
        self._operand_instances: Dict[str, int] = {}  # Tracks operand usage (e.g., "A": 2 → next is A2)
        self.s: str = ""  # Stores raw task string for summary

    def _get_unique_node(self, label: str, original_operand: str) -> str:
        """
        Generate unique node ID with subscripted label and original operand tracking.
        Args:
            label: Subscripted label (e.g., "A0", "B1")
            original_operand: Original operand name (e.g., "A", "B") for reference
        Returns:
            Unique node ID
        """
        node_id = f"v{self.node_counter}"
        self.node_counter += 1
        self.graph.add_node(
            node_id,
            label=label,  # Subscripted label for display
            original_operand=original_operand  # Map back to original operand
        )
        return node_id

    def _parse_atomic(self, operand: str) -> Tuple[Set[str], Set[str]]:
        """
        FSM for atomic subgoal with unique subscripted labels.
        Example: First "A" → "A0", second "A" → "A1", etc.
        Returns: (start_nodes, terminal_nodes) of the atomic FSM
        """
        self._operand_instances[operand] = self._operand_instances.get(operand, 0)
        subscript = self._operand_instances[operand]

        subscripted_label = f"{operand}_{subscript}"
        atomic_node = self._get_unique_node(subscripted_label, original_operand=operand)
        self._operand_instances[operand] += 1
        
        return ({atomic_node}, {atomic_node})

    def _parse_then(self, left_starts: Set[str], left_terminals: Set[str],
                   right_starts: Set[str], right_terminals: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        FSM for "t1 then t2" (paper §Rational Subgoal Learning and Planning):
        Merge FSMs of t1 and t2, add edges from t1's terminals to t2's starts.
        """
        for left_term in left_terminals:
            for right_start in right_starts:
                self.graph.add_edge(left_term, right_start, label="then")
        return (left_starts, right_terminals)

    def _parse_or(self, operands: List[Tuple[Set[str], Set[str]]]) -> Tuple[Set[str], Set[str]]:
        """
        FSM for "t1 or t2 or ..." (paper §Rational Subgoal Learning and Planning):
        Merge FSMs (no new edges), union starts/terminals.
        """
        all_starts = set()
        all_terminals = set()
        for starts, terminals in operands:
            all_starts.update(starts)
            all_terminals.update(terminals)
        return (all_starts, all_terminals)

    def _parse_and(self, operands: List[Tuple[Set[str], Set[str]]]) -> Tuple[Set[str], Set[str]]:
        """
        FSM for "t1 and t2 and ..." 
        All permutations of "then" chains (any order of subgoals) with unique subscripted nodes.
        """
        from itertools import permutations

        all_starts = set()
        all_terminals = set()
        for perm in permutations(operands):
            perm_original_instances = self._operand_instances.copy()
            try:
                current_starts, current_terminals = perm[0]
                for next_op in perm[1:]:
                    current_starts, current_terminals = self._parse_then(
                        current_starts, current_terminals, next_op[0], next_op[1]
                    )
                all_starts.update(current_starts)
                all_terminals.update(current_terminals)
            finally: self._operand_instances = perm_original_instances
        return (all_starts, all_terminals)

    def _tokenize(self, s: str) -> List[str]:
        """
        Tokenize input into operands (A-Za-z_), operators (then/or/and), parentheses.
        Ignores whitespace; raises error for invalid characters.
        """
        token_pattern = r"then|or|and|\(|\)|[A-Za-z_][A-Za-z0-9_]*"
        invalid_pattern = r"[^A-Za-z0-9_\s()thenorand]"
        
        invalid_chars = re.findall(invalid_pattern, s)
        if invalid_chars: raise ValueError(f"Invalid characters in input: {set(invalid_chars)}")
        
        tokens = re.findall(token_pattern, s)
        return [t.strip() for t in tokens if t.strip()]

    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[Tuple[Set[str], Set[str]], int]:
        def parse_primary(pos: int) -> Tuple[Tuple[Set[str], Set[str]], int]:
            if pos >= len(tokens): raise SyntaxError("Unexpected end of input (expected operand or '(')")
            token = tokens[pos]
            if token == "(":
                expr, pos = self._parse_expression(tokens, pos + 1)
                if pos >= len(tokens) or tokens[pos] != ")": raise SyntaxError("Unclosed parentheses (missing ')')")
                return expr, pos + 1  # Skip closing ')'
            elif token in ["then", "or", "and", ")"]: raise SyntaxError(f"Unexpected token '{token}' ,expected operand or '(' ")
            else: return self._parse_atomic(token), pos + 1

        left_expr, pos = parse_primary(pos)
        while pos < len(tokens) and tokens[pos] == "then":
            op = tokens[pos]
            pos += 1
            right_expr, pos = parse_primary(pos)
            left_expr = self._parse_then(
                left_expr[0], left_expr[1], right_expr[0], right_expr[1])

        while pos < len(tokens) and tokens[pos] in ["or", "and"]:
            op = tokens[pos]
            pos += 1
            right_expr, pos = self._parse_expression(tokens, pos)
            if op == "or":left_expr = self._parse_or([left_expr, right_expr])
            elif op == "and":left_expr = self._parse_and([left_expr, right_expr])

        return left_expr, pos

    def parse_string(self, s: str) -> None:
        """
        Parse TL task description (e.g., "A and B") into FSM with unique subscripted nodes:
        - Reset state (graph, node counter, operand instances) for fresh parsing.
        - Add super-start (v0) and super-terminal (vT) nodes (paper §Figure 2).
        """
        def replace_and(expr: str) -> str:
            and_pattern = r'((?:\((?:[^()]+|(?R))*\)|[A-Za-z_][A-Za-z0-9_]*))\s*and\s*((?:\((?:[^()]+|(?R))*\)|[A-Za-z_][A-Za-z0-9_]*))'
            while re.search(and_pattern, expr): expr = re.sub(and_pattern, r'(\1 then \2) or (\2 then \1)', expr, count=1)
            return expr
        s = replace_and(s)

        self.graph.clear()
        self.node_counter = 0
        self._operand_instances = {}  # Reset operand instance tracking
        self.s = s  # Store raw task string for summary

        tokens = self._tokenize(s)
        if not tokens:raise ValueError("Empty input string (expected TL task description)")

        (main_starts, main_terminals), pos = self._parse_expression(tokens, pos=0)
        if pos < len(tokens):raise SyntaxError(f"Unexpected tokens after main expression: {tokens[pos:]}")


        v0 = self._get_unique_node("S", original_operand="super")
        vT = self._get_unique_node("T", original_operand="super")


        for start_node in main_starts:self.graph.add_edge(v0, start_node, label="init")
        for terminal_node in main_terminals:self.graph.add_edge(terminal_node, vT, label="end")
    
    def get_sorted_tasks(self) -> List[str]:
        """
        Extract a sequential task order from the FSM (from super-start v0 to super-terminal vT).
        Returns:
            List[str]: Sorted sequence of task labels (e.g., ["v0", "A0", "B0", "vT"]).
                      Excludes duplicate nodes and respects `then` dependencies via topological sort.
        
        Raises:
            ValueError: If no FSM exists (call parse_string() first) or FSM has no valid path.
        """
        if not self.graph.nodes: raise ValueError("No FSM found. Call parse_string() first to build the FSM.")
        node_labels = nx.get_node_attributes(self.graph, "label")
        v0_id: Optional[str] = None
        vT_id: Optional[str] = None

        for node_id, label in node_labels.items():
            if label == "S": v0_id = node_id
            elif label == "T": vT_id = node_id
        
        if not v0_id or not vT_id: raise ValueError("FSM is invalid: Missing super-start (v0) or super-terminal (vT) node.")

        if not nx.has_path(self.graph, source=v0_id, target=vT_id): raise ValueError("No valid path from start (v0) to end (vT) in the FSM.")

        try: node_id_path = nx.shortest_path(self.graph, source=v0_id, target=vT_id)
        except nx.NetworkXNoPath: raise ValueError("No valid path from start node ('S') to end node ('T') in the FSM.")
        except nx.NodeNotFound: raise ValueError("Start ('S') or end ('T') node not found in the FSM (corrupted FSM).")

        sorted_labels = [node_labels[node_id] for node_id in node_id_path]
        if sorted_labels[0] != "S" or sorted_labels[-1] != "T": raise ValueError("Generated path is invalid (does not start with 'S' or end with 'T').")

        return sorted_labels


    def visualize(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        if not self.graph.nodes: raise ValueError("No FSM to visualize (call parse_string() first)")

        try: pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot", args="-Grankdir=LR")
        except ImportError: pos = nx.spring_layout(self.graph, seed=42)

        node_colors = []
        node_labels = nx.get_node_attributes(self.graph, "label")
        raw_labels = nx.get_node_attributes(self.graph, "label")

        for node_id in self.graph.nodes:
            label = raw_labels.get(node_id, "")
            node_colors.append("orange" if label in ["v0", "vT", "Start", "End", "S", "T"] else "lightblue")

        nx.draw_networkx_nodes(
            self.graph, pos, node_size=2500, node_color=node_colors,
            edgecolors="black", linewidths=2
        )
        nx.draw_networkx_labels(
            self.graph, pos, labels=node_labels, font_size=11, font_weight="bold"
        )

        edge_labels = nx.get_edge_attributes(self.graph, "label")
        nx.draw_networkx_edges(
            self.graph, pos, arrowstyle="->", arrowsize=25, edge_color="gray"
        )
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")  # Avoid label overlap
        )

        plt.axis("off")
        plt.title(f"FSM for Task: {self._get_task_summary()}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

    def _get_task_summary(self) -> str:
        """Return raw task string as summary (e.g., "A and B")"""
        return self.s


class AutoTaskSchedule:
    """Main scheduler for parsing task dependencies, loading tasks, and training a meta-learner."""
    def __init__(self, task: Union[DependencyAutomata, LearningTask, None] = None):
        """Initialize the scheduler with an optional task/automata instance."""
        self.automata: Optional[DependencyAutomata] = task if isinstance(task, DependencyAutomata) else None
        self.tasks: Dict[str, LearningTask] = {}  # Task name → LearningTask
        self._task_config_dir: Path = Path("./configs/curriculum")  # Default dir for task JSON configs
        self._task_config_dir.mkdir(exist_ok=True)  # Create dir if it doesn't exist
        self.task = None

    def _load_task_from_json(self, task_name: str) -> LearningTask:
        """Load a single LearningTask from a JSON config file (task_configs/<task_name>.json)."""
        config_path = self._task_config_dir / f"{task_name}.json"
        if not config_path.exists(): raise FileNotFoundError(f"Task config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f: config = json.load(f)
        if "name" not in config: raise ValueError(f"Task config {config_path} missing required 'name' field")
        
        return LearningTask(**config)

    def _parse_text_config(self, content: str) -> Tuple[str, List[LearningTask]]:
        dependency_expr = ""
        tasks = []
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
        
        current_task: Optional[Dict[str, Any]] = None
        for line in lines:
            if line.startswith("Dependency:"): dependency_expr = line.split(":", 1)[1].strip()
            elif line.startswith("Task:"):
                if current_task is not None:tasks.append(LearningTask(**current_task))
                task_name = line.split(":", 1)[1].strip()
                current_task = {"name": task_name}
            elif current_task is not None and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "vocab":
                    current_task["vocab"] = [v.strip() for v in value.split(",") if v.strip()]
                elif key == "domains":
                    current_task["domains"] = [d.strip() for d in value.split(",") if d.strip()]
                else: current_task[key] = value
        if current_task is not None:
            tasks.append(LearningTask(**current_task))
        
        if not dependency_expr:
            raise ValueError("No dependency expression found in task config (expected 'Dependency: ...')")
        return dependency_expr, tasks

    def _parse_json_config(self, content: str) -> Tuple[str, List[LearningTask]]:
        config = json.loads(content)
        
        if "dependency" not in config: raise ValueError("JSON config missing required 'dependency' field")
        if "tasks" not in config or not isinstance(config["tasks"], list): raise ValueError("JSON config missing required 'tasks' list")
        
        dependency_expr = config["dependency"]
        parsed_tasks = [LearningTask(**task) for task in config["tasks"]]
        
        return dependency_expr, parsed_tasks

    def load_tasks(self, task_str: str) -> DependencyAutomata:
        """
        Load tasks and parse dependencies from a file (".txt"/".json"/".config") or raw string.
        Args:
            task_str: Path to a config file OR raw config string OR dependency expression.
        Returns:
            DependencyAutomata: Parsed dependency graph with linked tasks.
        """
        dependency_expr = ""
        parsed_tasks = []
        
        if Path(task_str).exists():
            content = Path(task_str).read_text(encoding="utf-8")
            if task_str.endswith(".json"): dependency_expr, parsed_tasks = self._parse_json_config(content)
            elif task_str.endswith((".txt", ".config")): dependency_expr, parsed_tasks = self._parse_text_config(content)
            else: raise ValueError(f"Unsupported file type: {task_str} (only .json/.txt/.config allowed)")
        else:
            dependency_expr = task_str
            parsed_tasks = []

        self.automata = DependencyAutomata()
        self.automata.parse_string(dependency_expr)

        for task in parsed_tasks: self.tasks[task.name] = task
        self.task = self.automata
        return self.automata

    def train(self, model: MetaLearner) -> MetaLearner:
        """
        Train the meta-learner on tasks in topological order (respects dependencies).
        Args:
            model: MetaLearner instance to train.
        Returns:
            MetaLearner: Trained meta-learner.
        """
        if self.automata is None: raise RuntimeError("No tasks loaded: call load_tasks() first")
        sorted_tasks = self.automata.get_sorted_tasks()

        for task in sorted_tasks[1:-1]:
            pattern = r'(?:_\d+|\d+)$'
            task_name = re.sub(pattern, '', task)
            assert task_name in self.tasks, f"{task_name} not defined as learning task"

            task_node = self.tasks[task_name]
            if task_node.code : exec(task_node.code) # execute the import code

            assert task_node.train, f"task {task_node.name} does not have train set"
            train_loader = eval(task_node.train)
            test_loader =  eval(task_node.test) if task_node.test else None
            val_loader =  eval(task_node.val) if task_node.test else None

            #model.train_on_task(task)
        
        return model
    


if __name__ == "__main__":
    schedule = AutoTaskSchedule()

    schedule.load_tasks("configs/curriculum/expr0.json")
    #schedule.automata.visualize()
    schedule.train(None)
    #schedule.automata.visualize()