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
        self.curr_state = None
        self.steps = 0
        self.truncate_steps = 1000
    
    def step(self, action : str):
        self.curr_state = self.dojo.run_cmd(self.curr_state, action)
        reward = 0.0
        self.steps += 1
        terminated = False
        truncated = True if self.steps > self.truncate_steps else False
        info = {}
        return self.curr_state, reward, terminated, truncated, info
    
    def reset(self):
        self.curr_state = self.init_state
        self.steps = 0
        return self.curr_state
    


if __name__ == "__main__":
    soulforge = SoulforgeLeanEnv()
    done = False
    soulforge.reset()
    while not done:
        command = input()
        obs, reward, done, truncated, info = soulforge.step(command)
        print(obs.message)
        #print(dojo.run_cmd(state_0, "#eval 5"))
        #print(dojo.run_cmd(state_0, "#eval 6"))