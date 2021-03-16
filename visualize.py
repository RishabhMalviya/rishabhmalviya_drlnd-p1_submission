def run_random_agent(env, brain_name):
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = env.brains[env.brain_names[0]].vector_action_space_size

    while True:
        random_action = np.random.randint(action_size)
        env_info = env.step(random_action)[brain_name]

        next_state, done = env_info.vector_observations[0], env_info.local_done[0]

        if done:
            break


def run_trained_agent(env, brain_name, trained_agent):
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = env.brains[env.brain_names[0]].vector_action_space_size

    while True:
        state = env_info.vector_observations[0]
        action = trained_agent.act(state)
        env_info = env.step(action.item())[brain_name]

        reward, done = env_info.rewards[0], env_info.local_done[0]

        if done:
            break