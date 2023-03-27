import os
from datetime import datetime

sim_name = "Demo_sim"

# -> Create environment
root = str(os.getcwd())

from MAF_Environments.RS.RS_environment import Environment
map_size = "M"
environment = Environment(
    namespace=sim_name,
    map_reference=f"RS_{map_size}",
    scale_factor=1,
    cache_grids=False,
    cache_path="/home/vguillet/Documents/Repositories/CBAA_with_intercession/Environment/RS"
    )

# from MAF_Environments.Caylus.Caylus_environment import Environment
# environment = Environment(
#     namespace=sim_name,
#     map_reference=f"Caylus_map",
#     scale_factor=5,
#     cache_grids=True
#     )

# from MAF_Environments.Empty.Empty_environment import Environment
# environment = Environment(
#     namespace=sim_name,
#     map_reference=f"Empty_world",
#     scale_factor=10,
#     cache_grids=True
#     )

environment.render_terrain(
    # paths=[[(15, 15), (150, 10)]],
    # positions = [(15, 15), (20, 20), (25, 25), (30, 30), (40, 40)],
    # POIs = [(150, 150), (200, 200), (250, 250), (300, 300), (400, 400)],
    flat=True
)

# environment.render_comms(
#     paths=[[(15, 15), (150, 10)]],
#     positions = [(15, 15), (20, 20), (25, 25), (30, 30), (40, 40)],
#     POIs = [(150, 150), (200, 200), (250, 250), (300, 300), (400, 400)],
# )

# - Create fake path
# path = [(11, 11), (12, 12), (13, 13), (14, 14)]
# environment.render_path(path=path)

print("------------- Environment created")
