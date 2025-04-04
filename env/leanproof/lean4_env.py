from lean_dojo import LeanGitRepo, Dojo, ProofFinished, trace


class SoulforgeLeanEnv:
    """this is the default repo to start with """
    def __init__(self, repo = None):
        if repo is None: repo = LeanGitRepo("https://github.com/ZongjingLi/Soulforge","b8ca887adaa135c625ad9630b295286559100a4e",)
        self.repo = repo
        self.traced_repo = trace(repo)

        entry = (repo, "Main.lean", 1)  # (repo, file_path, line_nb)
        dojo, state_0 = Dojo(entry).__enter__()
        self.dojo = dojo
        self.init_state = state_0
        print(dojo.run_cmd(state_0, "#eval 5"))
        print(dojo.run_cmd(state_0, "#eval 6"))



if __name__ == "__main__":
    soulforge = SoulforgeLeanEnv()
    #print(len(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'cube_pos', 'cube_quat', 'gripper_to_cube_pos', 'robot0_proprio-state', 'object-state']))