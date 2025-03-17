import open3d as o3d
from env.blockworld.blockworld_env import Blockworld, generate_block_config
from rinarak.envs.recorder import SceneRecorder

if __name__ == "__main__":
    recorder = SceneRecorder(
        num_views = 8,
        camera_distance = 1.3,
        camera_height = 1.5,
        target_point = [0.0, 0.0, 0.5]
    )

    num_objs = 8
    env = Blockworld(gui = True)
    for scene_id in range(1):
        block_config = generate_block_config(num_objs)
        env.generate_blocks(block_config)

    recorder.record_scene_with_segmentation("outputs/blockworld", 1, True)

    print("saved")

    scene = 1
    clouds = [o3d.io.read_point_cloud(f"/Users/sunyiqi/Documents/GitHub/Aluneth/outputs/blockworld/scene_frame_{scene}/point_clouds/segmented/merged_object_{i}.ply") for i in range(1,3 + num_objs)]
    o3d.visualization.draw_geometries(clouds)    # Visualize point cloud    