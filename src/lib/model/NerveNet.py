import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym

from typing import Union

from lib.model.model import Model
from lib.utils.Running_mean_std import RunningMeanStd
from lib.utils.wrapper_utils import unflatten_tensorized_space, flatten_tensorized_space

def count_module_params(module: Model):
    return sum(p.numel() for p in module.parameters())

class NerveNetPolicy(Model):
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 node_info: dict[str, Union[dict, str]],
                 device: torch.device,
                 num_nodes: int = None,
                 num_actuated_nodes: int = None,
                 num_prop_steps: int = 3, 
                 action_dim: int = 1,
                 min_log_std: float = -20,
                 max_log_std: float = -2,
                 hidden_dim: int = 128):
        """
        Stochastic Policy based on NerveNet Architecture
        References: https://github.com/WilsonWangTHU/NerveNet/tree/master
        
        :param node_info: a dictionary containing the information of each node constructing the graph
        :type node_info: dict[str, Union[dict, str]]
        :param num_prop_steps: the number of massage passing operations
        :type num_prop_steps: int
        :param action_dim: dimension of final action
        :type action_dim: int
        """
        if num_nodes is None or num_actuated_nodes is None:
            raise ValueError("Please provide 'num_nodes' and 'num_actuated_nodes' when initializing NerveNetPolicy.")
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.node_info = node_info
        self.node_types_with_dim = node_info['node_types_dim']
        self.node_types_with_ids = node_info['node_types_ids']
        self.edge_types = node_info['edge_types']
        self.output_node_types = node_info['output_node_types']

        self.num_nodes = num_nodes
        self.num_actuated_nodes = num_actuated_nodes
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_prop_steps = num_prop_steps
        self.device = device

        # Running mean, standard deviation standardizer
        self.body_standardizer = RunningMeanStd(shape=observation_space['body'].shape, device=device)
        self.joint_standardizer = RunningMeanStd(shape=observation_space['joint'].shape, device=device)
        
        # -----------------------------------------------------------
        # 1. Embedding Layers
        # 각 노드 타입 별로 입력 차원이 다르므로 별도 MLP 사용
        # Type of Node : {body, joint}
        # -----------------------------------------------------------
        self.encoders = nn.ModuleDict()
        for node_type, input_dim in self.node_types_with_dim.items():
            self.encoders[node_type] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )

        # -----------------------------------------------------------
        # 2. Propagation Layers
        # 엣지 타입 별로 별도의 Propagation Layer
        # Type of Edge : {downstream, upstream}
        # -----------------------------------------------------------
        self.prop_mlps = nn.ModuleDict()
        for edge_type in self.edge_types.keys():
            self.prop_mlps[edge_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

        # -----------------------------------------------------------
        # 3. Update Layers
        # 메시지를 받은 후 내 상태를 갱신하는 최종 Feature Extract Layer
        # Type of Node : {body, joint}
        # -----------------------------------------------------------
        self.gru_cells = nn.ModuleDict()
        for node_type in self.node_types_with_dim.keys():
            self.gru_cells[node_type] = nn.GRUCell(hidden_dim, hidden_dim)

        # -----------------------------------------------------------
        # 4. Readout/Action Layers
        # 최종 Hidden State에서 Action을 뽑아내는 Action Head
        # output_node_types: {'hip': [idx...], 'knee': [idx...], 'ankle': [idx...]}
        # -----------------------------------------------------------
        self.actor_heads = nn.ModuleDict()
        for out_type in self.output_node_types.keys(): 
            self.actor_heads[out_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
        
        # LogStd
        self.log_std = nn.Parameter(torch.zeros(self.num_actuated_nodes, device=device))


        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)

        # Model Info Logging
        print("\n[Model Info]")
        print(f"1. Encoders      : {count_module_params(self.encoders):,}")
        print(f"2. Prop MLPs     : {count_module_params(self.prop_mlps):,}")
        print(f"3. GRU Cells     : {count_module_params(self.gru_cells):,}")
        print(f"4. Actor Heads   : {count_module_params(self.actor_heads):,}")
        print(f"--> Total Params : {count_module_params(self):,}\n") 


    def forward(self, 
                observations: Union[dict[str, torch.Tensor], torch.Tensor],
                taken_actions: Union[torch.Tensor, None],
                deterministic: bool = False,
                update_rms: bool = False):
        """
        Feature Extraction by GNN Policy

        :param observations: {'body': [B, N_b, m], 'joint': [Batch, N_j, n]}
        :type observations: dict[str, torch.Tensor]
        :param taken_actions: actions from buffer for log_prob calculation
        """
        if isinstance(observations, torch.Tensor):
            observations = unflatten_tensorized_space(self.observation_space, observations)
        # -----------------------------------------------------------
        # Step 1: Embedding & Alignment
        # 각 노드 데이터들을 순차적으로 모두 인코딩하여 GNN 신경망 입력 텐서 세팅 
        # -----------------------------------------------------------
        observations['body'] = self.body_standardizer.standardize(observations['body'], update=update_rms)
        observations['joint'] = self.joint_standardizer.standardize(observations['joint'], update=update_rms)
        batch_size = observations['body'].shape[0]
        all_hidden = torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=observations['body'].device)
        
        for node_type, node_obs in observations.items():
            indices = self.node_types_with_ids[node_type]
            encoded = self.encoders[node_type](node_obs)
            
            all_hidden[:, indices, :] = encoded

        # -----------------------------------------------------------
        # Step 2: Message Passing Loop (GNN Core)
        # -----------------------------------------------------------
        for _ in range(self.num_prop_steps):
            aggregated_messages = torch.zeros_like(all_hidden) # [B,N,H]
            
            # 각 엣지 타입별로 메시지 전파
            for edge_type, edge_index in self.edge_types.items():
                source_idx = edge_index[0]   # 보내는 노드 인덱스들 (1행)
                target_idx = edge_index[1]   # 받는 노드 인덱스들 (2행)
                
                # Gather: 보내는 놈들의 Hidden State 가져오기
                source_hidden = all_hidden[:, source_idx, :]
                
                # Transform: MLP 통과
                message = self.prop_mlps[edge_type](source_hidden)
                
                # Aggregate: 받는 놈 인덱스 위치에 메시지 더하기
                aggregated_messages.index_add_(1, target_idx, message)
            
            # Update: GRU를 통해 상태 갱신
            next_hidden = torch.zeros_like(all_hidden) # [B, N, H]
            for node_type, indices in self.node_types_with_ids.items():
                # 해당 타입 노드들의 message와 hidden만 골라냄
                type_msg = aggregated_messages[:, indices]
                type_h = all_hidden[:, indices]

                # [B,N,H] -> [B*N,H] for GRU
                flat_type_msg = type_msg.view(-1, self.hidden_dim)
                flat_type_h = type_h.view(-1, self.hidden_dim)
                
                # 해당 타입 전용 GRU로 업데이트
                updated_h = self.gru_cells[node_type](flat_type_msg, flat_type_h)
                
                # 결과 저장 [B*N,H] -> [B,N,H]
                next_hidden[:, indices] = updated_h.view(type_msg.shape)
            
            all_hidden = next_hidden # [B,N,H]

        # -----------------------------------------------------------
        # Step 3: Readout (Action Generation) - Node Type별 분리 적용
        # -----------------------------------------------------------
        
        # [B, N, A]
        actions_mean = torch.zeros((batch_size, self.num_nodes, self.action_dim), device=all_hidden.device)
        actions_mask = torch.zeros_like(actions_mean)
        # Output Type-specific Head (ex: 'hip', 'knee' etc..)
        for out_type, head_layer in self.actor_heads.items():
            indices = self.output_node_types[out_type]
            
            # [B,N,H]
            partial_hidden = all_hidden[:, indices, :]
            
            # [B,N,A]
            partial_action = head_layer(partial_hidden)
            
            actions_mean[:, indices] = partial_action
            actions_mask[:, indices] = 1.0
        
        # [B,N,A] -> [B,N] (action_dim=1)
        actions_mean = actions_mean.squeeze(-1)
        actions_mask = actions_mask.squeeze(-1)

        active_ids = actions_mask[0, :].bool()
        actions_mean = actions_mean[:, active_ids]

        # -----------------------------------------------------------
        # Step 4: Stochastic Action Processing
        # -----------------------------------------------------------

        log_std = self.log_std
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        # Action distribution
        self.action_distribution = Normal(actions_mean, log_std.exp())

        if deterministic:
            actions = actions_mean
        else:
            actions = self.action_distribution.rsample()
        
        # Log of the probability density function
        if taken_actions is not None:
            log_prob = self.action_distribution.log_prob(taken_actions)
        else:
            log_prob = self.action_distribution.log_prob(actions)

        # Log prob with action masking
        log_prob = log_prob
        log_prob = log_prob.sum(dim=-1)

        # Entropy : mean of (Batch, Action) dimension with action masking
        entropy = self.action_distribution.entropy()
        entropy = entropy.sum() / (actions_mask.sum() + 1e-8)

        return actions, log_prob, entropy


if __name__ == "__main__":
    # Virutal Robot: Body(0) -> UpperLeg(1, 3) -> LowerLeg(2, 4)
    device = torch.device("cpu") 
    BATCH_SIZE = 4
    NUM_NODES = 5
    HIDDEN_DIM = 16
    
    # Node Info Virtual Structure
    node_info = {
        # Input dimensions per type
        'node_types_dim': {
            'body': 12,  # Body features : 12 dims
            'joint': 4   # Joint features : 4 dims
        },
        # Node Ids per type
        'node_types_ids': {
            'body': [0],
            'joint': [1, 2, 3, 4]
        },
        # Edge
        'edge_types': {
            'downstream': torch.tensor([
                [0, 1, 0, 3], # Source
                [1, 2, 3, 4]  # Target
            ], dtype=torch.long, device=device),
            
            'upstream': torch.tensor([
                [1, 2, 3, 4], # Source (reversed)
                [0, 1, 0, 3]  # Target
            ], dtype=torch.long, device=device)
        },
        # Output groping
        'output_node_types': {
            'upper_leg': [1, 3], 
            'lower_leg': [2, 4],
        }
    }

    print(f"Testing NerveNetPolicy on {device}...")

    # ---------------------------------------------------------
    # Model Instantiation
    # ---------------------------------------------------------
    policy = NerveNetPolicy(
        node_info=node_info,
        device=device,
        num_nodes=NUM_NODES,
        hidden_dim=HIDDEN_DIM,
        action_dim=1
    ).to(device)


    print("\n" + "="*40)
    print("Model initialized successfully.")
    print(f"Model Architecture: {policy.__class__.__name__}")

    total_params = 0
    trainable_params = 0
    
    # 각 파라미터 그룹별로 이름과 크기를 출력
    # (너무 길어질 수 있으니 모듈 단위로 요약해서 보여줍니다)
    for name, param in policy.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"Total Parameters     : {total_params:>10,}")
    print(f"Trainable Parameters : {trainable_params:>10,}")
    print(f"Non-Trainable params : {total_params - trainable_params:>10,}")
    print("=" * 40)

    # ---------------------------------------------------------
    # Random Input
    # ---------------------------------------------------------
    observations = {
        # Body: [Batch, 1, 12]
        'body': torch.randn(BATCH_SIZE, 1, 12, device=device),
        # Joint: [Batch, 4, 4]
        'joint': torch.randn(BATCH_SIZE, 4, 4, device=device)
    }

    # ---------------------------------------------------------
    # Forward propagation
    # ---------------------------------------------------------
    actions, log_prob, entropy = policy(observations, taken_actions=None)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("\n[Output Shapes]")
    print(f"Actions  : {actions.shape} \t(Expected: [{BATCH_SIZE}, {NUM_NODES}])")
    print(f"Log Prob : {log_prob.shape} \t(Expected: [{BATCH_SIZE}])")
    print(f"Entropy  : {entropy.shape} \t(Expected: []) - Scalar")

    print("\n[Logic Checks]")
    
    body_action_sum = actions[:, 0].abs().sum().item()
    if body_action_sum == 0.0:
        print("✅ Body Action Masking: PASSED (All zero)")
    else:
        print(f"❌ Body Action Masking: FAILED (Sum: {body_action_sum})")

    joint_action_sum = actions[:, 1:].abs().sum().item()
    if joint_action_sum > 0.0:
        print("✅ Joint Action Generation: PASSED (Non-zero values)")
    else:
        print("❌ Joint Action Generation: FAILED (All zero)")
        
    if not torch.isnan(log_prob).any() and not torch.isnan(entropy):
        print("✅ Numerical Stability: PASSED")
    else:
        print("❌ Numerical Stability: FAILED (NaN detected)")