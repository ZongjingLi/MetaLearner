import open3d as o3d
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset, DataLoader

# ----- Directory Setup -----
BASE_DATA_DIR = "data"

# ----- Object Creation Functions -----
def create_random_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=random.uniform(0.2, 0.5))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(np.random.rand(3))
    return sphere

def create_random_cube():
    cube = o3d.geometry.TriangleMesh.create_box(width=random.uniform(0.3, 0.7),
                                                height=random.uniform(0.3, 0.7),
                                                depth=random.uniform(0.3, 0.7))
    cube.compute_vertex_normals()
    cube.paint_uniform_color(np.random.rand(3))
    return cube

# List of object creators
object_creators = [create_random_sphere, create_random_cube]

# ----- Scene Generation -----
def generate_random_scene(num_objects=4, points_per_object=1024, contact_prob=0.5):
    scene = []
    contact_matrix = np.zeros((num_objects, num_objects))

    for _ in range(num_objects):
        obj = random.choice(object_creators)()
        translation = np.random.uniform(-2, 2, size=2)
        obj.translate([translation[0], translation[1], 0])
        scene.append(obj)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if random.random() < contact_prob:
                contact_matrix[i, j] = 1
                contact_matrix[j, i] = 1

    pointclouds = [np.asarray(o.sample_points_poisson_disk(number_of_points=points_per_object).points) for o in scene]
    end_score = [random.uniform(0, 1) for _ in range(num_objects)]

    return { 
        "input": pointclouds,
        "contact": contact_matrix,
        "end": end_score
    }

# ----- Save Generated Data -----
def save_data(name, split, num_samples):
    from tqdm import tqdm
    save_dir = os.path.join(BASE_DATA_DIR, name, split)
    os.makedirs(save_dir, exist_ok=True)
    import random

    for i in tqdm(range(num_samples)):
        data = generate_random_scene(random.randint(3,5))

        # Ensure point clouds are saved as a list of float32 arrays
        pointclouds = [np.array(pc, dtype=np.float32) for pc in data["input"]]

        np.savez(os.path.join(save_dir, f"scene_{i}.npz"),
                 input=pointclouds,  # Now correctly formatted
                 contact=np.array(data["contact"], dtype=np.float32),
                 end=np.float32(data["end"]))  # Ensure scalar is float32
    
    print(f"Generated and saved {num_samples} samples in '{save_dir}'.")


# ----- Scene Dataset -----
class SceneDataset(Dataset):
    def __init__(self, name, split):
        self.data_dir = os.path.join(BASE_DATA_DIR, name, split)
        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"No data found in '{self.data_dir}'. Run save_data() first.")
        
        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        if not self.files:
            raise RuntimeError(f"No scene files found in '{self.data_dir}'. Run save_data() first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        pointclouds = list(data["input"])
        contact_matrix = torch.tensor(data["contact"], dtype=torch.float32)
        end_score = torch.tensor(data["end"], dtype=torch.float32)
        
        return {
            "input": pointclouds,
            "predicate": {
                "contact": contact_matrix,
                "end": end_score
            }
        }

# ----- Custom Collate Function -----
def custom_collate(batch):
    inputs = [item['input'] for item in batch]
    predicates = [item['predicate'] for item in batch]

    collated_predicates = {key: [pred[key] for pred in predicates] for key in predicates[0]}

    for key in collated_predicates:
        if isinstance(collated_predicates[key][0], torch.Tensor):
            collated_predicates[key] = [collated_predicates[key]]
        elif isinstance(collated_predicates[key][0], (int, float)):
            collated_predicates[key] = torch.tensor(collated_predicates[key], dtype=torch.float32)

    return {
        "input": inputs,
        "predicate": collated_predicates
    }

# ----- Main Execution -----
if __name__ == "__main__":
    experiment_name = "contact_experiment"
    num_train, num_val, num_test = 100, 20, 20  # Set dataset sizes

    # Generate and save data for different splits
    generate = 0
    if generate:
        save_data(experiment_name, "train", num_train)
        save_data(experiment_name, "val", num_val)
        save_data(experiment_name, "test", num_test)

    # Load dataset and create DataLoader
    train_dataset = SceneDataset(experiment_name, "train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    for sample in train_loader:
        pointclouds, contact_matrix, end_score = sample["input"], sample["predicate"]["contact"], sample["predicate"]["end"]
        print(f'Contact Matrix:\n{contact_matrix}')
        print(f'End Score: {end_score}')
