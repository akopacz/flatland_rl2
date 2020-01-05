import numpy as np
import pickle
import networkx as nx
import sys, argparse

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionProcessData # MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.agent_utils import RailAgentStatus

from greedy_agent import GreedyAgent
from build_graph import GraphBuilder
from my_astar import AStarAgent, Node


######### Set parameters for rail map generation #########
width = 16 * 4 # 16 * 7  # With of map
height = 9 * 4 # 9 * 7  # Height of map
nr_trains = 10 # nr_trains = 50  # Number of trains that have an assigned task in the env
cities_in_map = 6 # 20  # Number of cities where agents can start or end
seed = 18  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 6  # Max number of parallel tracks within a city, representing a realistic trainstation


######### Initialize railway #########
rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )
# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train
schedule_generator = sparse_schedule_generator(speed_ration_map)
stochastic_data = { "malfunction_rate":10000,  # Rate of malfunction occurence
                                        "min_duration":15,  # Minimal duration of malfunction
                                        "max_duration":50  # Max duration of malfunction
}
observation_builder = GlobalObsForRailEnv()

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              obs_builder_object=observation_builder,
              malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
              remove_agents_at_target=True)
env.reset()

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                          show_debug=False,
                          screen_height=1200,  # Adjust these parameters to fit your resolution
                          screen_width=1800)  # Adjust these parameters to fit your resolution


######### Get arguments of the script #########
parser=argparse.ArgumentParser()
parser.add_argument("-step", type=int,
                    help="steps")
args = parser.parse_args()


######### Custom controller setup #########
controller = GreedyAgent(218, env.action_space[0])
my_grid = [[Node((i, j), env.rail.grid[i, j]) for j in range(env.rail.width)] for i in range(env.rail.height)]
astar_planner = AStarAgent(my_grid, env.rail.width, env.rail.height)


######### Select agents #########
show_agents = [ a_id in range(env.number_of_agents)]

# stores if agent direction is 0, 1 (or 2, 3) 
# 1 means True
agent_directions = np.zeros(env.number_of_agents)

# Place agent on map
action_dict = dict()
for a in show_agents:
    action = controller.act(0)
    action_dict.update({a: action})
    agent_directions[a] = 1 if env.agents[a].direction < 2 else 0
# Do the environment step
observations, rewards, dones, information = env.step(action_dict)

for a in show_agents:
    agent = env.agents[a]
    if agent.position is not None:
        astar_planner.add_cell_to_avoid(agent.position)

astar_paths_readable = [None for _ in range(env.number_of_agents)]
# run A* for the selected agent
for a_id in show_agents:
    ag = env.agents[a_id]
    start = ag.initial_position
    if ag.position is not None:
        start = ag.position
    start = Node(start, env.rail.grid[start[0], start[1]], dir=ag.direction)
    end = Node(ag.target, env.rail.grid[ag.target[0], ag.target[1]])
    astar_paths_readable[a_id] = astar_planner.aStar(start, end)

######### Initialize step  #########
step = 200
if args.step:
    step = args.step
    

######### Run simulation  #########
score = 0
agent_current_node = np.zeros(len(env.agents), dtype=int)

# Run episode
frame_step = 0
for step in range(step):
    # Chose an action for each agent in the environment
    for a_id in show_agents:
        ag = env.agents[a_id]
        action = None
        if ag.status == RailAgentStatus.ACTIVE:
            next_cell = None
            if astar_paths_readable[a_id] is not None:
                next_cell = astar_paths_readable[a_id][agent_current_node[a_id]+1]
            if next_cell is None or ag.position == next_cell.point:
                # check if there is already someone on the route before the next intersection
                # if entered a new cell decide action
                if next_cell is None or next_cell.intersection:
                # if False:
                    # plan route
                    start = Node(ag.position, env.rail.grid[ag.position[0], ag.position[1]], dir=ag.direction)
                    end = Node(ag.target, env.rail.grid[ag.target[0], ag.target[1]])
                    astar_paths_readable[a_id] = astar_planner.aStar(start, end)
                    agent_current_node[a_id] = 0
                    if astar_paths_readable[a_id] is None:
                        # no route found 
                        # wait
                        action = 4
                    else:
                        # decide which way to go next
                        from_ = astar_paths_readable[a_id][0].point
                        to = astar_paths_readable[a_id][1].point
                        action = controller.simple_act(from_, ag.direction, to)
                else:
                    # follow path defined by a*
                    agent_current_node[a_id] += 1
                    # decide which way to go next
                    to = astar_paths_readable[a_id][agent_current_node[a_id] + 1].point
                    action = controller.simple_act(next_cell.point, ag.direction, to)
            else:
                # (continue previous movement)
                # go forward
                action = 2
            
        elif ag.status == RailAgentStatus.READY_TO_DEPART:
            # initializing with a going forward movement
            action = 0 
        if action is not None:
            action_dict.update({a_id: action})

    # update list of cells to avoid
    for a_id in show_agents:
        if env.agents[a_id] == RailAgentStatus.ACTIVE or env.agents[a_id] == RailAgentStatus.DONE:
            astar_planner.remove_cell_from_avoid(env.agents[a_id].position)

    next_obs, all_rewards, done, _ = env.step(action_dict)

    # update list of cells to avoid
    for a_id in show_agents:
        if env.agents[a_id] == RailAgentStatus.ACTIVE:
            astar_planner.add_cell_to_avoid(env.agents[a_id].position)

    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
    env_renderer.gl.save_image('../misc/Fames2/flatland_frame_{:04d}.png'.format(step))
    frame_step += 1
    # Update replay buffer and train agent
    for a in show_agents:
        controller.step((observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        score += all_rewards[a]

    observations = next_obs.copy()
    for a in show_agents:
        agent_directions[a] = 1 if env.agents[a].direction < 2 else 0
    if done['__all__']:
        break
    print('Episode: Steps {}\t Score = {}'.format(step, score))
