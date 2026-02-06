import torch

# =========== DOF Names Dictionary for Diffrent Node Building ===========

DOF_NAMES_ANT = ['front_left_leg', 'front_right_leg', 'left_back_leg', 'right_back_leg', 'front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

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
        node_info, num_nodes = build_ant_node_info(device)
        
    elif robot_name_low == "humanoid":
        # TODO: return build_humanoid_node_info(cfg, device, dof_names) # 구현 필요
        pass

    elif robot_name_low == "g1":
        # TODO: return build_g1_node_info(cfg, device, dof_names) # 구현 필요
        pass
        
    else:
        raise ValueError(f"Unknown robot configuration! Proposed Robot Name: {robot_name}")

    print(f"Node Info for {robot_name} built successfully.")
    return node_info, num_nodes


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

if __name__ == "__main__":
    device = torch.device("cpu")
    node_info = build_node_info(robot_name="Ant", device=device)
    print("Node Info:", node_info)