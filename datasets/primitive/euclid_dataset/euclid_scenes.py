'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-12-21 20:56:45
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-12-21 20:57:02
'''
import torch
import numpy as np
from .euclid_engine import generate_constrained_scene
from helchriss.utils.data import FilterableDatasetView, FilterableDatasetUnwrapped
from helchriss.utils.collate import VarLengthCollateV2
from torch.utils.data import DataLoader
import os
import json
from shutil import rmtree
from PIL import Image

def rand_color():return np.random.choice(["red","green","blue"])

def rand_primitive():return np.random.choice(["line", "circle"])

def rand_shape():return np.random.choice(["triangle", "rectangle"])


def colored_square(colors=["red", "green", "blue", "yellow"]):
    program = f"""
l1:line(p1, p2)[color({colors[0]})];
l2:line(p2, p3)[color({colors[1]}), perpendicular(l2, l1)];
l3:line(p3, p4)[color({colors[2]}), perpendicular(l3, l2), parallel(l3, l1)];
l4:line(p4, p1)[color({colors[3]}), perpendicular(l4, l3), parallel(l4, l2)];
"""
    return program.strip()

def cross_line_angle(colors = ["red","green"]):
    program = f"""
l1:line(p1, p2)[color({colors[0]}),];
l2:line(p3, p4)[color({colors[1]}),intersect(l2, l1)];
"""
    return program.strip()

def two_circle_line_connect(colors):
    circle_color1,circle_color2,line_color = colors
    program = f"""
    c1:circle(p1,p2)[color({circle_color1}),!overlap(c2,c1)];
    c2:circle(p3,p4)[color({circle_color2}), !overlap(c1,c2)];
    l1:line(p1,p3)[color({line_color})];

    """
    return program

def double_parallel(line_colors):
    program = f"""
    l1:line(p1,p2)[color({line_colors[0]})];
    l2:line(p2,p3)[color({line_colors[1]}), perpendicular(l1,l2),perpendicular(l2,l1)];
    l3:line(p3,p4)[color({line_colors[2]}), perpendicular(l2,l3),perpendicular(l3,l2)];
    """
    return program

def colored_right_triangle(colors = ["red", "green", "blue"]):
    program = f"""
    l1:line(p1, p2)[color({colors[0]})];
    l2:line(p2, p3)[perpendicular(l1,l2),perpendicular(l2,l1),color({colors[1]})];
    l3:line(p3, p1)[color({colors[2]})];
    """
    return program

def tangent_line_and_circle(colors):
    line_color, circle_color=colors
    dsl_program = f"""
    c1:circle(p1, p2)[color({circle_color})];
    l1:line(p3, p4)[color({line_color}), tangent(l1, c1)];
    """
    return dsl_program.strip()

def gaget(colors):
    line1_color, line2_color, line3_color = colors
    program = f"""
l1:line(p1, p2)[color({line1_color}),];
l2:line(p3, p4)[color({line2_color}),intersect(l2, l1)];
l3:line(p4, p6)[color({line2_color}), perpendicular(l3,l2), perpendicular(l2,l3)];
l4:line(p1, p8)[color({line3_color}),parallel(l3,l4), parallel(l4,l3)];
"""
    return program

def circle(colors):
    return f"c1:circle(p1,p1)[color({colors[0]})]"

"""generating scenes"""

from tqdm import tqdm

def gen_euclid_scene(num_scenes = 3):
    dsl, imgs, scene_qs, scene_ps, answers, scene_segs, scene_metas = [], [], [], [], [], [], []
    for i in range(num_scenes):
        scene_type = np.random.choice([
            "square", "biline", "fn", "doub_para",
            "right_triangle", "tangent_lc", "gaget", "circle"])
        if scene_type == "square":
            colors = [rand_color() for _ in range(4)]
            gen_program = colored_square(colors)
        if scene_type == "biline":
            colors = [rand_color() for _ in range(2)]
            gen_program = cross_line_angle(colors)
        if scene_type == "fn":
            colors = [rand_color() for _ in range(3)]
            gen_program = two_circle_line_connect(colors)
        if scene_type == "doub_para":
            colors = [rand_color() for _ in range(3)]
            gen_program = double_parallel(colors)
        if scene_type == "right_triangle":
            colors = [rand_color() for _ in range(3)]
            gen_program = colored_right_triangle(colors)
        if scene_type == "tangent_lc":
            colors = [rand_color() for _ in range(2)]
            gen_program = tangent_line_and_circle(colors)
        if scene_type == "gaget":
            colors = [rand_color() for _ in range(3)]
            gen_program = gaget(colors)
        if scene_type == "circle":
            colors = [rand_color() for _ in range(1)]
            gen_program = circle(colors)
        
        scene_img, scene_seg, scene_meta =  generate_constrained_scene(gen_program)
        
        query, program, answer = euclid_object_grounding_questions(scene_meta)

        dsl.append(gen_program)
        imgs.append(scene_img)
        scene_segs.append(scene_seg)
        scene_qs.append(query)
        scene_ps.append(program)
        answers.append(answer)
        scene_metas.append(scene_meta)
    
    return dsl, imgs, scene_qs, scene_ps, answers, scene_segs, scene_metas

def gen_angle_scene(num_scenes = 3):
    """about angles and lengths mapping"""
    return {}

def gen_shapes_scene(num_scenes = 3):
    """exists forall questions on primitive"""
    return {}

def gen_spatial_scene(num_scenes = 3):
    return {}


"""generate questions for grounding purpose"""

def euclid_object_grounding_questions(scene_metas):
    """exist forall questions on primitive property and relations"""
    templates = ["exists_primitive", "exist_color", "double_filter"] # "exist_color"
    qtype = np.random.choice(templates)

    """color based property filter"""
    if qtype == "exist_color":
        primitive = rand_color()
        q_template = f"exists {primitive} object"
        p_template = f"exists(filter(objects(), 'lambda x => {primitive}(color(x))' ))"
        answer = len([obj for obj in scene_metas["objects"] if obj["color_name"] == primitive]) > 0

    if qtype == "exists_primitive":
        primitive = rand_primitive()
        q_template = f"exists {primitive} object"
        p_template = f"exists(filter(objects(), 'lambda x => {primitive}(x)' ))"
        answer = len([obj for obj in scene_metas["objects"] if obj["type"] == primitive]) > 0

    if qtype == "double_filter":
        primitive = rand_primitive()
        color     = rand_color()
        q_template = f"exists {primitive} {color} object"
        p_template = f"exists(filter(filter(objects(), 'lambda x => {primitive}(x)' ), 'lambda x => {color}(color(x))' ))"
        answer = len([obj for obj in scene_metas["objects"] if (obj["type"] == primitive) and obj["color_name"] == color ] ) > 0


    
    """relations questions about line line relations"""

    """relations questions about line circle relations"""

    """relations questions about circle circle relations"""
    return q_template, p_template, answer

def euclid_angle_grounding_questions():
    """about angles and lengths and size mapping"""
    return

def euclid_spatial_grounding_questions():
    """exist forall questions on primitive property and relations"""
    templates = ["relate_direction", "relate_distance"]
    qtype = np.random.choice(templates)
    if qtype == "relate_direction":
        tgt_obj, ref_obj = rand_primitive(), rand_primitive()
        relation = np.random.choice(["left", "right", "above", "below"])
        q_template = f"exists a {tgt_obj} at {relation} of {ref_obj}"
        return
    return 

def euclid_shape_grounding_questions():
    """exist forall questions on primitive property and relations"""
    templates = ["exist_triangle", "exists_rectangle", "exists_polygon"]
    qtype = np.random.choice(templates)
    
    return 

def mereology_property_grounding_questions():
    """a euclid shape is consists of parts made of lines and circles"""
    templates = ["form_triangle", "form_rectangle", "form_polygon", "shape_contains", "shape_filter", "part_filter"]
    qtype = np.random.choice(templates)

    if qtype == "form_triangle":
        q_template = "the red line and blue line forms a triangle"
        return
    
    if qtype == "shape_contains":
        shape = rand_shape()
        color = rand_color()
        part  = rand_primitive()
        q_template = f"the {shape} contains a {color} {part}"
    
    if qtype == "shape_filter":
        q_template = f"exist a {shape} "

    if qtype == "part_filter":
        ref_color = rand_color()
        tgt_color = rand_color()
        part = rand_primitive()
        q_template = f"the {shape} that contains a {ref_color} part also have a {tgt_color} {part} "
    
    return

"""
generate the euclid dataset using the weights of each split
"""

def gen_euclid_dataset(dataset_size = 100, options = {}):
    dsl_programs = []
    images   = []
    programs = []
    answers  = []
    queries  = []
    segments = []

    weights = {"euclid": 1., "angle": 0., "shape": 0., "spatial": 0.}
    if "weights" in options: weights = weights

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    generator = {
        "euclid": gen_euclid_scene,
        "angle": gen_angle_scene,
        "shape": gen_shapes_scene,
        "spatial": gen_spatial_scene}

    for key, p in weights.items():
        if p == 0: continue
        dsl, imgs, scene_qs, scene_ps,scene_ans,scene_segs, scene_metas = generator[key](int(dataset_size * p))
        dsl_programs.extend(dsl)
        images.extend(imgs)
        queries.extend(scene_qs)
        programs.extend(scene_ps)
        answers.extend(scene_ans)
        segments.extend(scene_segs)
    
    return {
        "images":images,
        "questions":queries,
        "programs":programs,
        "answers" : answers,
        "segments" : scene_segs
        }


class EuclidDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, dataset_size, options = {}):
        super().__init__()
        self.data = gen_euclid_dataset(dataset_size, options)

    def _get_metainfo(self, index):
        return {
            'question': self.data['questions'][index],
            'program': self.data['programs'][index],
            'answer': self.data['answers'][index],  # can be bool or numeric
            'program' : self.data['programs'][index],
            'question_length': len(self.data['questions'][index].split()),
            'question_type': 'boolean' if isinstance(self.data['answers'][index], bool) else 'arithmetic'
        }

    def __getitem__(self, index):

        return {
            'image': _to_image(self.data['images'][index]),
            'query': self.data['questions'][index],
            'program': self.data['programs'][index],
            'answer': self.data['answers'][index],  # Can be bool or numeric
            'grounding' : {
                "image" : _to_image(self.data['images'][index]),
                "segment" : self.data["segments"][index]
                }
        }

    def __len__(self):
        return len(self.data['images'])


def _to_image(image):
    """Convert image to PyTorch tensor format (with RGB ↔ BGR swap)"""
    # Swap R and B channels (channel dimension is index 2 for HWC format)

    image = image.permute(1, 2, 0).flip(dims = [2])  # HWC → CHW
    #image = image / 255.0
    #image = image.astype(np.float32)
    #image = (image - 0.5) * 2
    return torch.tensor(image)

class EuclidDatasetFilterableView(FilterableDatasetView):
    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> DataLoader:
        collate_guide = {
            'query': 'skip',
            'program': 'skip',
            'answer': 'skip',
            'sprites': 'skip'
        }
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )

    def filter_question_length(self, length: int) -> 'EuclidDatasetFilterableView':
        """Filter dataset based on question length"""
        def filt(meta):
            return meta['question_length'] <= length
        return self.filter(filt, f'filter-qlength[{length}]')
    
    def filter_by_answer(self, answer) -> 'EuclidDatasetFilterableView':
        """Filter dataset based on answer (can be boolean or numeric)"""
        def filt(meta):
            return meta['answer'] == answer
        return self.filter(filt, f'filter-answer[{answer}]')
    
    def filter_by_question_type(self, question_type: str) -> 'EuclidDatasetFilterableView':
        """Filter dataset based on question type (boolean or arithmetic)"""
        def filt(meta):
            return meta['question_type'] == question_type
        return self.filter(filt, f'filter-qtype[{question_type}]')


def EuclidDataset(dataset_size, options = {}) -> EuclidDatasetFilterableView:
    return EuclidDatasetFilterableView(EuclidDatasetUnwrapped(dataset_size,options))


""""save dataset"""

def save_euclid_dataset(dataset: EuclidDatasetUnwrapped, save_root: str, overwrite: bool = True):
    """
    将Euclid数据集保存为类似shape3d的目录结构
    
    Args:
        dataset: 已初始化的EuclidDatasetUnwrapped实例
        save_root: 保存根目录（最终会创建save_root/euclid_dataset目录）
        overwrite: 是否覆盖已存在的目录
    """
    # 定义数据集根目录
    dataset_root = os.path.join(save_root)
    
    # 处理目录存在情况
    if os.path.exists(dataset_root):
        if overwrite:
            rmtree(dataset_root)
        else:
            raise FileExistsError(f"数据集目录 {dataset_root} 已存在，设置overwrite=True可覆盖")
    
    # 创建目录结构（对应shape3d格式）
    os.makedirs(dataset_root, exist_ok=False)
    imgs_dir = os.path.join(dataset_root, "imgs")
    queries_dir = os.path.join(dataset_root, "queries")
    segments_dir = os.path.join(dataset_root, "segments")
    os.makedirs(imgs_dir, exist_ok=False)
    os.makedirs(queries_dir, exist_ok=False)
    os.makedirs(segments_dir, exist_ok=False)
    
    # 1. 保存answers.json
    answers_data = [
        {
            "index": i,
            "answer": bool(ans) if isinstance(ans, bool) else ans,
        }
        for i, ans in enumerate(dataset.data['answers'])
    ]
    with open(os.path.join(dataset_root, "answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)
    import datetime
    
    # 2. 保存dataset_info.txt
    dataset_info = f"""
Euclid Dataset Information
==========================
Dataset Size: {len(dataset)}
Creation Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Question Types: boolean, arithmetic
Scene Types: square, biline, fn, doub_para, right_triangle, tangent_lc, gaget
Weights: euclid=1.0, angle=0.0, shape=0.0, spatial=0.0
Image Shape: {dataset.data['images'][0].shape if len(dataset) > 0 else "Unknown"}
    """.strip()
    with open(os.path.join(dataset_root, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(dataset_info)
    
    # 3. 保存imgs目录（图像文件）
    for i, img_tensor in enumerate(tqdm(dataset.data['images'], desc="Saving images")):
        # 将张量转为PIL图像
        img = _tensor_to_pil(img_tensor)
        img.save(os.path.join(imgs_dir, f"scene_{i:06d}.png"), format="PNG")
    
    # 4. 保存programs.txt
    with open(os.path.join(dataset_root, "programs.txt"), "w", encoding="utf-8") as f:
        for i, (dsl_prog, prog) in enumerate(zip(dataset.data['programs'], dataset.data['programs'])):
            f.write(f"=== Scene {i:06d} ===\n")
            f.write(f"DSL Program:\n{dsl_prog}\n")
            f.write(f"Query Program:\n{prog}\n\n")
    
    # 5. 保存queries目录（每个查询一个txt文件）
    for i, q in enumerate(tqdm(dataset.data['questions'], desc="Saving queries")):
        query_path = os.path.join(queries_dir, f"query_{i:06d}.txt")
        with open(query_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {q}\n")

    """
    # 6. 保存sprites_info.json（对应场景元信息）
    sprites_info = [
        {
            "index": i,
            "dsl_program": dataset.data['dsl_programs'][i],
            "objects_count": len(meta.get("objects", [])),  # 若有对象信息
            "scene_type": _infer_scene_type(dataset.data['dsl_programs'][i])
        }
        for i, meta in enumerate(dataset.metainfo_list)
    ]
    with open(os.path.join(dataset_root, "sprites_info.json"), "w", encoding="utf-8") as f:
        json.dump(sprites_info, f, indent=2, ensure_ascii=False)
    """
    
    # 7. 保存segments目录（分割图）
    for i, seg_tensor in enumerate(dataset.data['segments']):
        # 处理张量格式：转为NumPy数组，确保形状为(W,H,N)，保留原始维度信息
        if isinstance(seg_tensor, torch.Tensor):
            # 若为CHW格式（C=N），转为HWC（即W,H,N，此处H=W，对应调整）
            if seg_tensor.dim() == 3 and seg_tensor.shape[0] == seg_tensor.shape[2]:
                seg_np = seg_tensor.permute(1, 2, 0).cpu().numpy()  # CHW → HWC (W,H,N)
            else:
                seg_np = seg_tensor.cpu().numpy()
        else:
            seg_np = np.array(seg_tensor)
        
        # 确保形状为(W,H,N)，若缺少N维度则补充（N=1）
        if len(seg_np.shape) == 2:
            seg_np = np.expand_dims(seg_np, axis=-1)  # (W,H) → (W,H,1)
        
        # 保存为.npy文件
        seg_save_path = os.path.join(segments_dir, f"segment_{i:06d}.npy")
        np.save(seg_save_path, seg_np)
    """
    print(f"数据集已成功保存至: {dataset_root}")
    print(f"目录结构：")
    print(f"├── euclid_dataset/")
    print(f"│   ├── answers.json")
    print(f"│   ├── dataset_info.txt")
    print(f"│   ├── imgs/ (场景图像)")
    print(f"│   ├── programs.txt")
    print(f"│   ├── queries/ (查询文件)")
    print(f"│   ├── sprites_info.json")
    print(f"│   └── segments/ (分割图像)")
    """


def _tensor_to_pil(tensor, is_segment: bool = False):
    """将张量转换为PIL图像"""
    # 归一化到0-255范围
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        tensor = (tensor * 255).byte()
    
    # 处理通道格式
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)  # CHW → HWC
    
    # 转为numpy数组
    img_np = tensor.cpu().numpy()
    
    # 处理单通道分割图
    if is_segment and img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
    
    # 转为PIL图像
    if len(img_np.shape) == 2:
        return Image.fromarray(img_np, mode="L")  # 灰度图
    else:
        return Image.fromarray(img_np, mode="RGB")


def load_euclid_dataset(load_root: str, length = 1000) -> EuclidDatasetUnwrapped:
    """
    从保存的目录加载Euclid数据集，返回EuclidDatasetUnwrapped实例
    
    Args:
        load_root: 数据集根目录（包含euclid_dataset文件夹）
    
    Returns:
        初始化后的EuclidDatasetUnwrapped实例
    """
    # 确定数据集目录
    dataset_root = load_root
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"未找到数据集目录: {dataset_root}")
    
    # 定义各文件路径
    answers_path = os.path.join(dataset_root, "answers.json")
    dataset_info_path = os.path.join(dataset_root, "dataset_info.txt")
    imgs_dir = os.path.join(dataset_root, "imgs")
    programs_path = os.path.join(dataset_root, "programs.txt")
    queries_dir = os.path.join(dataset_root, "queries")
    sprites_info_path = os.path.join(dataset_root, "sprites_info.json")
    segments_dir = os.path.join(dataset_root, "segments")
    
    # 1. 加载answers.json
    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)
    answers = [item["answer"] for item in answers_data]
    dataset_size = len(answers)
    
    # 2. 加载dsl_programs和programs（从programs.txt）
    dsl_programs = []
    programs = []
    with open(programs_path, "r", encoding="utf-8") as f:
        lines = f.read().split("=== Scene ")
    for scene_block in lines[1:]:  # 跳过第一个空字符串
        blocks = scene_block.split("DSL Program:\n")[1].split("Query Program:\n")
        dsl_prog = blocks[0].split("Query Program:\n")[0].strip()
        prog = blocks[1].split("\n\n")[0].strip()
        dsl_programs.append(dsl_prog)
        programs.append(prog)
    
    # 3. 加载questions（从queries目录）
    questions = []
    query_files = sorted([f for f in os.listdir(queries_dir) if f.startswith("query_")])
    for q_file in tqdm(query_files, desc="Loading queries"):
        q_path = os.path.join(queries_dir, q_file)
        with open(q_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 提取问题内容
        question_line = [line for line in content.split("\n") if line.startswith("Question: ")][0]
        question = question_line.split("Question: ")[1].strip()
        questions.append(question)
    
    # 4. 加载images（从imgs目录）
    images = []
    img_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(".png")])
    for img_file in tqdm(img_files, desc="Loading images"):
        img_path = os.path.join(imgs_dir, img_file)
        # 加载为PIL图像并转为张量
        img = Image.open(img_path).convert("RGB")
        img_tensor = _pil_to_tensor(img)
        images.append(img_tensor/255.)
    
    segments = []
    # 筛选.npy格式的分割文件，按文件名排序（保证索引对应）
    seg_files = sorted([f for f in os.listdir(segments_dir) if f.endswith(".npy")])
    for seg_file in seg_files:
        seg_path = os.path.join(segments_dir, seg_file)
        

        seg_np = np.load(seg_path)
        
        # 验证形状：确保是3维(W,H,N)，避免维度缺失
        if len(seg_np.shape) != 3:
            raise ValueError(f"分割数据 {seg_file} 形状异常，预期(W,H,N)，实际{seg_np.shape}")
        

        seg_tensor = torch.tensor(seg_np, dtype=torch.float32)
        
        # 可选：若后续需要CHW格式，可在此转换（注释掉则保留(W,H,N)）
        # seg_tensor = seg_tensor.permute(2, 0, 1)  # (W,H,N) → (N,W,H)
        
        segments.append(seg_tensor)

    dataset_dict = {
        "dsl_programs": dsl_programs[:length],
        "images": images[:length],
        "questions": questions[:length],
        "programs": programs[:length],
        "answers": answers[:length],
        "segments": segments[:length]
    }

    load_dataset = EuclidDatasetUnwrapped(0,{})
      
    load_dataset.data = dataset_dict

    return EuclidDatasetFilterableView(load_dataset)

def _pil_to_tensor(pil_img, is_segment: bool = False):
    img_np = np.array(pil_img)
    if is_segment and len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=-1)
    tensor = torch.tensor(img_np).permute(2, 0, 1)  # HWC → CHW
    if tensor.shape[0] == 3: tensor = tensor[[2, 1, 0], :, :]
    return tensor.float()

if __name__ == "__main__":
    train_euclid_dataset = EuclidDataset(6)
    save_euclid_dataset("/Users/sunyiqi/Documents/Datasets/euclid/train")