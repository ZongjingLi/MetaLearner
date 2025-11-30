import numpy as np
from typing import Dict, List, Tuple

class NLELanguage:
    def __init__(self):
        # Glyph-实体映射表（扩展版）
        self.glyph_map = {
            ord('#'): "sink",
            ord('g'): "goblin",
            ord('f'): "fountain",
            ord('w'): "wall",
            ord('d'): "door",
            ord('%'): "food",
            ord('@'): "player",
            ord('|'): "horizontal wall",
            ord('-'): "vertical wall",
            ord('+'): "closed door",
            ord('/'): "wand",
            ord('.'): "floor",
            ord('='): "altar",
            ord('%'): "fountain",
            ord('&'): "throne",
            ord('`'): "boulder",
            ord('<'): "staircase up",
            ord('>'): "staircase down",
            ord('['): "passage",
            ord(']'): "passage end",
            ord('o'): "orc",
            ord('t'): "troll",
            ord('z'): "zombie",
            ord('s'): "snake",
            ord('b'): "bat",
            ord('c'): "centaur",
            ord('!'): "potion",
            ord(')'): "weapon",
            ord('"'): "ring",
            ord('('): "shield",
            ord('$'): "gold coin",
            ord('*'): "gem",
            #ord(' '): "nothing"
        }
        # 复数规则映射
        self.plural_map = {
            "sink"  : "sinks",
            "weapon" : "weapons",
            "food" : "food",
            "goblin": "goblins",
            "fountain": "fountains",
            "wall": "walls",
            "door": "doors",
            "floor" : "floor",
            "orc": "orcs",
            "troll": "trolls",
            "zombie": "zombies",
            "snake": "snakes",
            "bat": "bats",
            "centaur": "centaurs",
            "potion": "potions",
            "mace": "maces",
            "ring": "rings",
            "shield": "shields",
            "food": "foods",
            "gold coin": "gold coins",
            "gem": "gems",
            "nothing" : "nothings",
            "boulder" : "boulders"
        }

    def _process_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """处理三元组：合并重复项并复数化"""
        entity_stats = {}

        entity_stats = {}
        for entity, distance, direction in triples:
            key = (entity, distance)
            if key not in entity_stats:
                entity_stats[key] = set()  # 用集合去重方向
            entity_stats[key].add(direction)
        
        processed = []
        for (entity, distance), directions in entity_stats.items():
            combined_dir = " and ".join(directions)
            try:
                plural_entity = self.plural_map[entity] if len(directions) > 1 else entity
            except:
                plural_entity = self.plural_map[entity.split(" ")[-1]] if len(directions) > 1 else entity
            processed.append((plural_entity, distance, combined_dir))
        return processed

    def text_glyphs(self, glyphs: np.ndarray, blstats: np.ndarray) -> bytes:
        """将Glyphs转为语言描述（含去重复数化）"""
        agent_x, agent_y = self._get_agent_position(blstats)
        #print(agent_x, agent_y)

        triples = self._generate_triples(glyphs, agent_x, agent_y)
        processed_triples = self._process_triples(triples)
        observation = "\n".join([f"{e} {d} {dir}" for e, d, dir in processed_triples])
        return observation.encode("latin-1")

    def _get_agent_position(self, blstats: np.ndarray) -> Tuple[int, int]:
        #print(blstats)
        return int(blstats[0]), -int(blstats[1])

    def _generate_triples(self, glyphs: np.ndarray, agent_x: int, agent_y: int) -> List[Tuple[str, str, str]]:
        """生成(实体, 距离, 方向)三元组"""
        triples = []
        height, width = glyphs.shape
        for y in range(height):
            for x in range(width):
        
                glyph = glyphs[y, x]
                if glyph not in self.glyph_map or self.glyph_map[glyph] == "player":
                    continue
                entity = self.glyph_map[glyph]
                distance = self._calculate_distance(agent_y, agent_x, -y, x)
                distance_label = self._get_distance_label(distance)

                #print(agent_x, agent_y, x, -y, entity)
                direction_label = self._get_direction_label(agent_y, agent_x, -y, x)
                triples.append((entity, distance_label, direction_label))
        return triples

    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        return np.sqrt((x2 - x1)** 2 + (y2 - y1) ** 2)

    def _get_distance_label(self, distance: float) -> str:
        ranges = [
            (0, 1.5, "adjacent"), (1.5, 4, "very near"),
            (4, 5, "near"), (5, 20, "far"),
            (20, float('inf'), "very far")
        ]
        for min_d, max_d, label in ranges:
            if min_d <= distance <= max_d:
                return label
        return "very far"

    def _get_direction_label(self, x1: int, y1: int, x2: int, y2: int) -> str:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return "on self"
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angle = (angle + 360) % 360
        sectors = [
            "north", "northnortheast", "northeast", "eastnortheast",
            "east", "eastsoutheast", "southeast", "southsoutheast",
            "south", "southsouthwest", "southwest", "westsouthwest",
            "west", "westnorthwest", "northwest", "northnorthwest"
        ]
        sector = int((angle + 11.25) // 22.5) % 16
        return sectors[sector]

    def text_blstats(self, blstats: np.ndarray) -> bytes:
        blstats = blstats.astype(int)
        stats = [
            f"HP: {blstats[3]}/{blstats[4]}",
            f"Strength: {blstats[5]}/{blstats[6]}",
            f"Hunger: {self._get_hunger_label(blstats[17])}",
            f"Gold: {blstats[18]}",
            f"Depth: {blstats[21]}"
        ]
        return ", ".join(stats).encode("latin-1")

    def _get_hunger_label(self, hunger: int) -> str:
        labels = {0: "Satiated", 1: "Not Hungry", 2: "Hungry", 3: "Fainting", 4: "Starving"}
        return labels.get(hunger, "Unknown")

    def text_message(self, tty_chars: np.ndarray) -> bytes:
        #print(tty_chars)
        tty_chars = tty_chars.flatten()
        message = "".join([chr(c) for c in tty_chars if c != 0]).strip()
        return message.encode("latin-1")

    def text_inventory(self, inv_strs: np.ndarray, inv_letters: np.ndarray) -> bytes:
        inventory = []
        for idx in range(len(inv_letters)):
            if inv_letters[idx] == 0:
                continue
            letter = chr(inv_letters[idx]).lower()
            item = "".join([chr(c) for c in inv_strs[idx] if c != 0]).strip()
            inventory.append(f"{letter}: {item}")
        return ", ".join(inventory).encode("latin-1")

    def text_cursor(self, glyphs: np.ndarray, blstats: np.ndarray, tty_cursor: np.ndarray) -> bytes:
        cursor_x, cursor_y = int(tty_cursor[0]), int(tty_cursor[1])
        agent_x, agent_y = self._get_agent_position(blstats)
        if 0 <= cursor_y < glyphs.shape[0] and 0 <= cursor_x < glyphs.shape[1]:
            glyph = glyphs[cursor_y, cursor_x]
            entity = self.glyph_map.get(glyph, "unknown")
            distance = self._calculate_distance(agent_x, agent_y, cursor_x, cursor_y)
            distance_label = self._get_distance_label(distance)
            direction_label = self._get_direction_label(agent_x, agent_y, cursor_x, cursor_y)
            return f"Cursor: {entity} {distance_label} {direction_label}".encode("latin-1")
        return "Cursor: invalid".encode("latin-1")


class NLEWrapper:
    def __init__(self):
        self.nle_language = NLELanguage()

    def process_observation(self, nle_obsv: Dict[str, np.ndarray]) -> Dict[str, str]:
        return {
            "text_glyphs": self.nle_language.text_glyphs(nle_obsv["chars"], nle_obsv["blstats"]).decode("latin-1"),
            "text_message": self.nle_language.text_message(nle_obsv["tty_chars"]).decode("latin-1"),
            "text_blstats": self.nle_language.text_blstats(nle_obsv["blstats"]).decode("latin-1"),
            "text_inventory": self.nle_language.text_inventory(nle_obsv["inv_strs"], nle_obsv["inv_letters"]).decode("latin-1"),
            "text_cursor": self.nle_language.text_cursor(nle_obsv["chars"], nle_obsv["blstats"], nle_obsv["tty_cursor"]).decode("latin-1"),
        }


# 测试示例（匹配流程图逻辑）
if __name__ == "__main__":
    wrapper = NLEWrapper()
    # 模拟观测数据：两个fountain，一个goblin
    nle_obsv = {
        "glyphs": np.array([
            [0, ord('f'), ord('f')],
            [ord('g'), ord('g'), 0],
            [0, 0, 0]
        ]),
        "blstats": np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        "tty_chars": np.array([ord(c) for c in "Test Message"] + [0]*100),
        "inv_strs": np.array([[ord(c) for c in "mace"] + [0]*28, [0]*32, [0]*32]),
        "inv_letters": np.array([ord('a'), 0, 0]),
        "tty_cursor": np.array([1, 0])
    }
    result = wrapper.process_observation(nle_obsv)
    print("text_glyphs:", result["text_glyphs"])
    # 输出示例：fountains adjacent east and northeast goblins adjacent west and north