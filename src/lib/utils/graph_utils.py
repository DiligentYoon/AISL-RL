import torch
import numpy as np
import scipy.sparse.csgraph as csgraph

# =========== DOF Names Dictionary for Diffrent Node Building ===========

DOF_NAMES_ANT = ['front_left_leg', 'front_right_leg', 
                 'left_back_leg', 'right_back_leg', 
                 'front_left_foot', 'front_right_foot', 
                 'left_back_foot', 'right_back_foot']

DOF_NAMES_HUMANOID = [
    'lower_waist:0', 'lower_waist:1',                     # Waist (2)
    'right_upper_arm:0', 'right_upper_arm:2',             # Right Shoulder (2)
    'left_upper_arm:0', 'left_upper_arm:2',               # Left Shoulder (2)
    'pelvis',                                               # Pelvis (1)
    'right_lower_arm',                                      # Right Elbow (1)
    'left_lower_arm',                                       # Left Elbow (1)
    'right_thigh:0', 'right_thigh:1', 'right_thigh:2',   # Right Hip (3)
    'left_thigh:0', 'left_thigh:1', 'left_thigh:2',      # Left Hip (3)
    'right_shin',                                           # Right Knee (1)
    'left_shin',                                            # Left Knee (1)
    'right_foot:0', 'right_foot:1',                       # Right Ankle (2)
    'left_foot:0', 'left_foot:1'                          # Left Ankle (2)
]

# =======================================================================

def get_id(dof_names: list[str], name_pattern: str):
    """
    Helper function to find the node index from the dof_name string
    Since index 0 is the body, the joint index is (dof_index + 1).
    
    :param dof_names: Description
    :param name_pattern: Description
    """
    for i, dof_name in enumerate(dof_names):
        if name_pattern in dof_name:
            return i + 1 
    raise ValueError(f"Joint matching '{name_pattern}' not found in dof_names: {dof_names}")

def build_node_info(robot_name: str, device: torch.device):
    """
    Node Info Builder for GNN Policy Construction

    :param robot_name: Articulation Name
    :param device: torch.device (cpu or cuda)
    """
    robot_name_low = robot_name.lower()

    if robot_name_low == "ant":
        dof_names = DOF_NAMES_ANT
        node_info, num_nodes = build_ant_node_info(device)
        
    elif robot_name_low == "humanoid":
        dof_names = DOF_NAMES_HUMANOID
        node_info, num_nodes = build_humanoid_node_info(device)

    elif robot_name_low == "g1":
        # TODO: return build_g1_node_info(cfg, device, dof_names) # 구현 필요
        pass
        
    else:
        raise ValueError(f"Unknown robot configuration! Proposed Robot Name: {robot_name}")

    # ========== Shortest Path Matrix ==========
    # Parent -> Child Edges
    edges = node_info['edge_types']['downstream'].cpu().numpy()

    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[edges[0], edges[1]] = 1
    adj_matrix[edges[1], edges[0]] = 1

    # Shortest path graph for maksed message passing (Multi DoFs is considered as a node with d=1)
    sp_matrix = csgraph.shortest_path(adj_matrix, directed=False, unweighted=True)

    # ========== Mapiing Dictionary ===========
    map_dict = {}

    body_dim = node_info['node_types_dim']['body']
    joint_dim = node_info['node_types_dim']['joint']

    # Body (Root) Node
    map_dict['body'] = ([1, body_dim], [])

    # Joint Nodes
    current_obs_idx = 0
    current_act_idx = 0

    for i, dof_name in enumerate(dof_names):
        node_key = dof_name
        map_dict[node_key] = ([current_obs_idx, joint_dim], [current_act_idx])

        current_obs_idx += 1
        current_act_idx += 1
    
    mapping_config = {
        'map': map_dict,
        'sp_matrix': sp_matrix
    }

    return node_info, num_nodes, mapping_config


def build_ant_node_info(device: torch.device) -> tuple[dict, int]:
    """
    Automatically generates 'node_info' based on the DoF names of the Ant robot.
    
    Assumption: 
        - The robot has 8 joints (DoFs).
        - The naming convention follows standard Ant patterns (e.g., 'front_left_leg', 'front_left_foot').
        
    :param device: torch.device (cpu or cuda)
    :param dof_names: list of strings containing the names of the joints in order.
    :return: dict containing graph topology and dimension info.
    """
    
    # Index 0 is reserved for the 'body' (Torso).
    # Indices 1 to 8 are assigned to joints based on the order in 'dof_names'.
    dof_names = DOF_NAMES_ANT
    
    node_types_ids = {
        'body': [0],
        'joint': list(range(1, 9))  # [1, 2, ..., 8]
    }
    
    # Define input dimensions for each node type
    # Example: 
    #   - Body: 12 dim (pos, ori, lin_vel, ang_vel)
    #   - Joint: 4 dim (pos, vel, prev_torque, pos_b)
    node_types_dim = {
        'body': 12,  
        'joint': 6   
    }

    # The Ant has 4 symmetric legs.
    # Structure per leg: Body(0) -> Upper Leg (Hip) -> Lower Leg (Foot/Ankle)
    downstream_edges = []
    upstream_edges = []
    # Prefixes for the 4 legs (Standard Isaac Sim / Mujoco naming)
    legs = ['front_left', 'front_right', 'left_back', 'right_back']


    for leg_prefix in legs:
        # [Connection 1] Body (0) <-> Upper Leg (Hip)
        hip_idx = get_id(dof_names, f"{leg_prefix}_leg") 
        
        downstream_edges.append([0, hip_idx]) # Parent -> Child
        upstream_edges.append([hip_idx, 0])   # Child -> Parent
        
        # [Connection 2] Upper Leg (Hip) <-> Lower Leg (Foot/Ankle)
        foot_idx = get_id(dof_names, f"{leg_prefix}_foot") 
        
        downstream_edges.append([hip_idx, foot_idx]) # Parent -> Child
        upstream_edges.append([foot_idx, hip_idx])   # Child -> Parent

    # Convert lists to Tensor with shape [2, E]
    edge_types = {
        'downstream': torch.tensor(downstream_edges, dtype=torch.long, device=device).t(),
        'upstream': torch.tensor(upstream_edges, dtype=torch.long, device=device).t()
    }

    # Since all 4 legs are mechanically identical, they share the same policy weights (NerveNet Assumptions).
    # We group them into 'leg' (hip joints) and 'foot' (ankle joints).
    output_node_types = {
        'leg': [],  # Group for upper joints
        'foot': []  # Group for lower joints
    }
    
    for leg_prefix in legs:
        output_node_types['leg'].append(get_id(dof_names, f"{leg_prefix}_leg"))
        output_node_types['foot'].append(get_id(dof_names, f"{leg_prefix}_foot"))

    # Construct the final dictionary
    node_info = {
        'node_types_dim': node_types_dim,
        'node_types_ids': node_types_ids,
        'edge_types': edge_types,
        'output_node_types': output_node_types
    }
    
    return node_info, len(dof_names) + 1

def build_humanoid_node_info(device: torch.device) -> tuple[dict, int]:
    """
    Generates 'node_info' for the Humanoid (21 DoF) robot.
    
    Structure:
      - Node 0: Body (Torso/Root)
      - Node 1~21: Joints
      
    Topology:
      - Torso <-> Waist, Upper Arms
      - Waist <-> Pelvis
      - Pelvis <-> Thighs
      - Thighs <-> Shins <-> Feet
      - Upper Arms <-> Lower Arms
    """ 
    dof_names = DOF_NAMES_HUMANOID
    num_joints = len(dof_names)

    node_types_ids = {
        'body': [0],
        'joint': list(range(1, num_joints + 1))
    }

    # Humanoid usually requires more features for body (orientation, velocity)
    # and joints (position, velocity, rotation_matrix/sin_cos, relative_pos)
    node_types_dim = {
        'body': 12,  # pos_z(1), ori(3), lin_vel(3), ang_vel(3), attitude (2)
        'joint': 12   # joint_pos(1), joint_vel(1), prev_torque(1), link_rel_pos(3), link_lin_vel(3), link_ang_vel (3)
    }

    downstream_edges = []
    upstream_edges = []

    # ---------------------------------------------------------
    # Helper 1: Fully Connect (Sibling / Clique)
    # ---------------------------------------------------------
    def connect_all_to_all(node_list):
        if len(node_list) < 2: return
        for i in node_list:
            for j in node_list:
                if i != j:
                    downstream_edges.append([i, j])
                    upstream_edges.append([j, i])

    # ---------------------------------------------------------
    # Helper 2: Bipartite Connect (Parent-Child / Hierarchy)
    # ---------------------------------------------------------
    def connect_group_to_group(src_list, dst_list):
        if not src_list or not dst_list: return
        for s in src_list:
            for d in dst_list:
                downstream_edges.append([s, d])
                upstream_edges.append([d, s])


    # Helper to find ALL indices matching a pattern (since Thigh has 3 DoFs, etc.)
    def get_ids(pattern):
        ids = []
        for i, name in enumerate(dof_names):
            if pattern in name:
                ids.append(i + 1)
        return ids
    
    # --- Torso Connections ---
    # Torso <-> Waist
    # Connect all Dofs of Waist to each other (Self Connection)
    waist_ids = get_ids('waist')
    connect_all_to_all(waist_ids)
    # Connect all DoFs of Waist to Torso
    for w_id in waist_ids:
        downstream_edges.append([0, w_id])
        upstream_edges.append([w_id, 0])

    # Torso <-> Upper Arms
    # Connect all DoFs of Upper Arm to Torso
    upper_arms_ids = get_ids('upper_arm')
    for u_id in upper_arms_ids:
        downstream_edges.append([0, u_id])
        upstream_edges.append([u_id, 0])
    
    # --- Joint-to-Joint Chain ---
    # Waist <-> Pelvis
    # Connect all DoFs of Waist to all DoFs of Pelvis
    pelvis_ids = get_ids('pelvis')
    for w_id in waist_ids:
        for p_id in pelvis_ids:
            downstream_edges.append([w_id, p_id])
            upstream_edges.append([p_id, w_id])
    
    sides = ['left', 'right']
    for side in sides:
        # === Leg Chain (Thigh <-> Shin <-> Foot) ===
        thigh_ids = get_ids(f"{side}_thigh") # (3 DoFs)
        shin_ids  = get_ids(f"{side}_shin")   # (1 DoFs)
        foot_ids  = get_ids(f"{side}_foot")   # (2 DoFs)

        # Self Connection
        connect_all_to_all(thigh_ids)
        connect_all_to_all(foot_ids)

        # Group Connection
        # Pelvis <-> Thighs
        connect_group_to_group(pelvis_ids, thigh_ids)
        # Thighs <-> Shin
        connect_group_to_group(thigh_ids, shin_ids)
        # Shin <-> Foot
        connect_group_to_group(shin_ids, foot_ids)

        # === Arm Chain (Upper Arm <-> Lower Arm) ===
        u_arm_ids = get_ids(f"{side}_upper_arm")  # Upper Arm (Shoulder, 2DoFs)
        l_arm_ids = get_ids(f"{side}_lower_arm" ) # Lower Arm (Elbow,    1DoFs)

        # Self Connection
        connect_all_to_all(u_arm_ids)

        # Group Connection
        connect_group_to_group(u_arm_ids, l_arm_ids)

    # Convert to Tensor
    edge_types = {
        'downstream': torch.tensor(downstream_edges, dtype=torch.long, device=device).t(),
        'upstream': torch.tensor(upstream_edges, dtype=torch.long, device=device).t()
    }

    # --- Parameter Sharing (Symmetry Groups) ---
    # Group symmetric parts (Left & Right) to share weights
    output_node_types = {
        'waist': get_ids('waist') + get_ids('pelvis'), # Central body parts
        'upper_arm': get_ids('upper_arm'),             # Shoulders
        'lower_arm': get_ids('lower_arm'),             # Elbows
        'thigh': get_ids('thigh'),                     # Hips
        'shin': get_ids('shin') + get_ids('knee'),     # Knees (cover both naming conventions)
        'foot': get_ids('foot') + get_ids('ankle')     # Ankles
    }

    # Filter out empty lists just in case
    output_node_types = {k: v for k, v in output_node_types.items() if v}

    node_info = {
        'node_types_dim': node_types_dim,
        'node_types_ids': node_types_ids,
        'edge_types': edge_types,
        'output_node_types': output_node_types
    }
    
    return node_info, num_joints + 1



class Mapping:
    def __init__(self, mapping):
        self.map = mapping['map']
        self.shortest_path_matrix = mapping['sp_matrix']
    
    def get_adjacency_matrix(self):
        return self.shortest_path_matrix < 2
    
    def create_observation(self, observations:dict[str, torch.Tensor]):
        """
        Convert 2 dims tensor to dict of 1 dim tensors based on the mapping [B,N,D] -> [B,D]_1, [B,D]_2, ...
        
        :param observations: Dictionary of observations. Keys = {'body', 'joint'}

        :return: Dictionary of observations split by all nodes.
        :rtype: Dictionary. Keys = {DOF_names_1, DOF_names_2}
        """
        obs = observations
        new_obs = {}
        for k, v in self.map.items():
            # Body : Root Node
            if k == 'body':
                new_obs['body'] = obs['body'] # [Batch, Nboides, Dim_per_body]
                continue

            # Joint : DOF names as keys 
            input_dims, _ = v  # input indices are indexs of 2 dims tensor
            new_obs[k] = obs['joint'][:, input_dims[0]].unsqueeze(1) # [B, Njoints, Dim_per_joint]
        return new_obs

    def create_action(self, action: torch.Tensor):
        new_action = {}
        for k, v in self.map.items():
            new_action[k] = action[:,v[1]]
        return new_action