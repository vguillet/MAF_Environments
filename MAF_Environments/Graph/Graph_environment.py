
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import numpy as np
from networkx import *

from MAF_Environments.to_delete.CAF_environment import CAF_environment
from MAF_Environments.Graph.Map_generator import *


# Environment layout
RANDOM = 0
GRID = 1 
STAR = 2


class Graph_environment(CAF_environment):
    def __init__(self,
                 layout: int = GRID,
                 nodes_count: int = 20,
                 connectivity_percent: float = 0.8,  # Ignored for star
                 branch_count: int = 8,  # Only relevant for the star layout

                 # Sim config
                 # Meta
                 instance_name: str = None
                 ):
        
        # -> Initialise CAF environment
        super().__init__(
            environment_type="GRAPH",
            instance_name=instance_name
        )

        if layout == RANDOM:
            self.environment_graph, self.nodes_pos = generate_random_layout(
                num_nodes=nodes_count,
                connectivity_percent=connectivity_percent
            )

        elif layout == GRID:
            self.environment_graph, self.nodes_pos = generate_grid_layout(
                num_nodes=nodes_count,
                connectivity_percent=connectivity_percent
            )

        elif layout == STAR:
            self.environment_graph, self.nodes_pos = generate_star_layout(
                num_nodes=nodes_count,
                num_branches=branch_count
            )
        
        else:
            raise f"ERROR: Invalid layout setting: {layout}"

    def _get_instruction_sequence(self, start, end, weighted: bool = False):
        # -> Determine shortest path to goal from current loc
        if weighted:
            path = astar_path(G=self.environment_graph, source=start, target=end, weight='weight')
        else:
            path = shortest_path(G=self.environment_graph, source=start, target=end)

        return path


    # ===================================== Plots
    def plot_environment(self):
        # -> Set up the figure and axis
        fig, ax = plt.subplots()

        # -> Set min and max of axis
        min_x = 0
        max_x = 0

        min_y = 0
        max_y = 0

        for x, y in self.nodes_pos:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        buffer = 0.5

        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)

        ax.set_axis_off()

        # -> Lock the aspect ratio to 1:1
        ax.set_aspect('equal')

        # -> Get edge weights
        weights = nx.get_edge_attributes(self.environment_graph, 'weight')

        # -> Draw the graph as a background
        nx.draw_networkx(
            G=self.environment_graph,
            pos=self.nodes_pos,
            with_labels=False,
            ax=ax,
            # node_color="white",
            alpha=0.3)

        nx.draw_networkx_edge_labels(
            G=self.environment_graph,
            pos=self.nodes_pos,
            edge_labels=weights,
            ax=ax)

        plt.plot()

    def plot_state(self):
        pass

    def animate_sim_snapshots(self, frame_count: int, snapshots, task_schedule):
        # Animation config
        playback_rate = 0.1

        # -------------------------------------- Prepare data
        # -> Reformat snapshots to correct lists
        # TODO

        # -> Gather all agent-task pairings for each epoch
        task_consensus_timeline = []

        for fleet_snapshot in snapshots["fleet_state"]:
            epoch_consensus_state = []

            for agent in fleet_snapshot.agent_list:
                agent_beliefs = []

                for task_id in agent.local_tasks_dict.keys():
                    if agent.local_tasks_dict[task_id]["status"] == "done":
                        continue

                    winning_bid = agent._get_local_task_winning_bid(task_id=task_id)
                    local_winner_id = winning_bid["agent_id"]

                    task_loc = agent.local_tasks_dict[task_id]["instructions"][-1][-1]

                    # -> Store (winning agent, goto loc)
                    agent_beliefs.append({
                        "local_winner_id": local_winner_id,
                        "task_loc": task_loc})

                epoch_consensus_state.append(agent_beliefs)

            for _ in range(int(frame_count / len(snapshots["fleet_state"]))):
                task_consensus_timeline.append(epoch_consensus_state)

                # -> Create the interpolator
        interpolator = interp1d(np.arange(len(snapshots["agents_pos"])),
                                snapshots["agents_pos"], axis=0)

        # -> Generate new frames by interpolation
        new_frames = []
        for i in np.linspace(0, len(snapshots["agents_pos"]) - 1, frame_count):
            interpolated_frame = interpolator(i)
            new_frames.append(interpolated_frame, )

        print(len(new_frames), len(task_consensus_timeline))

        # print(new_frames)
        merged_frames = []

        for i in range(frame_count):
            merged_frames.append({
                "agents_pos": new_frames[i],
                "agents_beliefs": task_consensus_timeline[i]})

        new_frames = merged_frames

        # -------------------------------------- Prepare plot
        # -> Set up the figure and axis
        fig, ax = plt.subplots()

        # -> Set min and max of axis
        min_x = 0
        max_x = 0

        min_y = 0
        max_y = 0

        for x, y in self.nodes_pos:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        buffer = 0.5

        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)

        ax.set_axis_off()

        # -> Lock the aspect ratio to 1:1
        ax.set_aspect('equal')

        # -> Get edge weights
        weights = nx.get_edge_attributes(self.environment_graph, 'weight')

        # -> Draw the graph as a background
        nx.draw_networkx(
            G=self.environment_graph,
            pos=self.nodes_pos,
            with_labels=False,
            ax=ax,
            # node_color="white", 
            alpha=0.3)

        nx.draw_networkx_edge_labels(
            G=self.environment_graph,
            pos=self.nodes_pos,
            edge_labels=weights,
            ax=ax)

        # -> Generate a color per agent
        agents_colors = cm.rainbow(np.linspace(0, 1, len(self.agents)))

        # -> Create a scatter plot for each agent loc
        agent_positions = []

        # ... for every agent
        for agent in range(len(self.get_agent_group(group_name="base_agent"))):
            agent_positions.append(ax.scatter([], [], color=agents_colors[agent], marker="D"))

        # -> Create belief line plots
        beliefs_lines = []

        # ... for every agent and every tasks
        for _ in range(self.agent_count * len(task_schedule)):
            beliefs_lines.append(ax.plot([], []))

        # -> Create a scatter plot for each task loc
        tasks_positions = []

        # ... for every tasks
        for task in range(len(task_schedule)):
            tasks_positions.append(ax.scatter([], [], color="green"))

        # -> Define the animation function
        def update(frame):
            # ... for every agent
            for agent in range(len(frame["agents_pos"])):
                agent_positions[agent].set_offsets(frame["agents_pos"][agent][0:-1])  # Update agent position

            # -> Cleanup prev lines
            for i in range(len(beliefs_lines)):
                beliefs_lines[i][0].set_data([], [])

            # -> Cleanup prev tasks positions
            for task in range(len(tasks_positions)):
                tasks_positions[task].set_alpha(0)

            offset_increment = 0.02
            offset_tracker = {}

            # ... for every agent
            for agent in range(len(frame["agents_beliefs"])):
                # ... for every tasks
                for task in range(len(frame["agents_beliefs"][agent])):
                    # if frame["agents_beliefs"][agent][task]:

                    # -> Set task position
                    tasks_positions[task].set_offsets((frame["agents_beliefs"][agent][task]["task_loc"][0],
                                                       frame["agents_beliefs"][agent][task]["task_loc"][1]))
                    tasks_positions[task].set_alpha(1)

                    # -> Find current agent loc
                    for agent_i in range(len(frame["agents_pos"])):
                        if frame["agents_pos"][agent_i][-1] == frame["agents_beliefs"][agent][task]["local_winner_id"]:
                            line_x = [frame["agents_pos"][agent_i][0],
                                      frame["agents_beliefs"][agent][task]["task_loc"][0]]
                            line_y = [frame["agents_pos"][agent_i][1],
                                      frame["agents_beliefs"][agent][task]["task_loc"][1]]

                            # -> If overlap, update and apply offset
                    if str(frame["agents_beliefs"][agent][task]) in offset_tracker.keys():
                        offset_tracker[str(frame["agents_beliefs"][agent][task])] += offset_increment

                    else:
                        offset_tracker[str(frame["agents_beliefs"][agent][task])] = 0

                    # -> Calculate the angle of the line segment
                    x0, y0 = line_x[0], line_y[0]
                    x1, y1 = line_x[-1], line_y[-1]
                    angle = math.atan2(y1 - y0, x1 - x0)

                    # -> Ponderate offset
                    # angle_diff = abs(angle - math.pi/4)
                    # offset_scale = max(1 - angle_diff/(math.pi/4), 0)
                    # offset = offset_scale * offset_increment
                    # x_coordinates = [x + offset for x in frame["agents_beliefs"][agent][task][0]]
                    # y_coordinates = [y + offset for y in frame["agents_beliefs"][agent][task][1]]

                    # -> Apply the offset to the x and y coordinates based on the angle
                    offset = offset_tracker[str(frame["agents_beliefs"][agent][task])]
                    x_offset = math.sin(angle) * offset
                    y_offset = math.cos(angle) * offset
                    x_coordinates = [x + x_offset for x in line_x]
                    y_coordinates = [y + y_offset for y in line_y]

                    # Apply the offset to the x and y coordinates
                    # offset = offset_tracker[str(frame["agents_beliefs"][agent][task])]
                    # x_coordinates = [x + offset for x in frame["agents_beliefs"][agent][task][0]]
                    # y_coordinates = [y + offset for y in frame["agents_beliefs"][agent][task][1]]

                    # -------------------------------------------------
                    beliefs_lines[agent * len(task_schedule) + task][0].set_data(x_coordinates,
                                                                                      y_coordinates)  # update x and y data of the plot object
                    beliefs_lines[agent * len(task_schedule) + task][0].set_color(
                        agents_colors[agent])  # set color

            return [agent_positions[i] for i in range(len(agent_positions))] + [beliefs_lines[i][0] for i in
                                                                                range(len(beliefs_lines))] + [
                       tasks_positions[i] for i in range(len(tasks_positions))]

        # -> Create the animation
        anim = FuncAnimation(fig, update, frames=new_frames, interval=(1000 / frame_count) / playback_rate, blit=True)

        # -> Save
        anim.save(filename=f"Sim.gif", writer='pillow')

        # -> Show the animation
        plt.show()

        return anim
