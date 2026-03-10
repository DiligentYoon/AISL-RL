import torch
import gymnasium as gym

from typing import Union, Any

from lib.model.MLP import Actor, Critic, SharedActor, JointActor
from lib.model.MLP_test import SuperConnectedActor, CommunetActor
from lib.model.NerveNet import NerveNetPolicy
from lib.utils.graph_utils import Mapping
from lib.model.BodyTransformer.body_transformer import BodyLevelActor, BodyLevelCritic
from lib.model.BodyTransformer.linear_components import ObsTokenizer, ValueDetokenizer, ActionDetokenizer
from lib.model.BodyTransformer.transformer_components import BodyTransformer

class ModelFactory:
    def __init__(self,
                 cfg: dict[str, Union[dict, Any]],
                 device: Union[torch.device, str]):
        """
        ModelFactor class for generating nn Model of actor and critic agent.

        :param cfg: configuration dictionary of model and agent
        :param device: torch device
        """

        self.model_cfg = cfg
        
        # Load model cfg
        # self.is_shared = self.model_cfg.get("shared", False)
        # self.is_joint  = self.model_cfg.get("joint", False)
        self.is_shared = False                                          # NOTE: 이거 최적화해야할지도?
        self.is_squashed = self.model_cfg.get("squashed", False)
        self.is_multi_agent = self.model_cfg.get("multi_agent", False)
        self.model_type = self.model_cfg["policy"].get("type", None)
        
        if self.model_type is None:
            self.model_type_lower = "mlp"
        
        else:
            self.model_type_lower = self.model_type.lower()

        self.device = device


    def generate_mlp_models(self,
                            observation_size: Union[dict[str, int], int],
                            state_size: Union[dict[str, int], int],
                            action_size: Union[dict[str, int], int],
                            possible_agents: list[str] = None):
        
        if (possible_agents is None) and (self.is_multi_agent):
            raise RuntimeError("Please confirm the cfg file whether multi_agent is assigned true or false")

        if self.is_multi_agent:
            # Multi agent
            models = {}
            if self.model_type_lower == "joint":
                self.is_shared = True
                actor = JointActor(possible_agents=possible_agents,
                                   num_observations=observation_size,
                                   num_actions=action_size,
                                   encoder_hidden_dim=128,
                                   RMA_hidden_dim=0,
                                   min_log_std=self.model_cfg["policy"]["min_log_std"],
                                   max_log_std=self.model_cfg["policy"]["max_log_std"],
                                   squash=self.is_squashed,
                                   device=self.device)
                
            elif self.model_type_lower == "shared":
                self.is_shared = True
                actor = SharedActor(possible_agents=possible_agents,
                                    num_observations=observation_size,
                                    num_actions=action_size,
                                    encoder_hidden_dim=128,
                                    RMA_hidden_dim=0,
                                    min_log_std=self.model_cfg["policy"]["min_log_std"],
                                    max_log_std=self.model_cfg["policy"]["max_log_std"],
                                    squash=self.is_squashed,
                                    device=self.device)
            
            elif self.model_type_lower == "superconnected":
                self.is_shared = True
                actor = SuperConnectedActor(possible_agents=possible_agents,
                                            num_observations=observation_size,
                                            num_actions=action_size,
                                            min_log_std=self.model_cfg["policy"]["min_log_std"],
                                            max_log_std=self.model_cfg["policy"]["max_log_std"],
                                            squash=self.is_squashed,
                                            device=self.device)
            
            elif self.model_type_lower == "communet":
                self.is_shared = True
                actor = CommunetActor(possible_agents=possible_agents,
                                      num_observations=observation_size, 
                                      num_actions=action_size,
                                      hidden_dim=128,
                                      communet_depth=4,
                                      min_log_std=self.model_cfg["policy"]["min_log_std"], 
                                      max_log_std=self.model_cfg["policy"]["max_log_std"],
                                      squash=self.is_squashed, 
                                      device=self.device)
            
            # Critic initialization
            for uid in possible_agents:
                # Multi Agent : Per-agent network
                if self.model_type_lower == "mlp":
                    self.is_shared = False
                    actor = Actor(num_observations=observation_size[uid],
                                num_actions=action_size[uid],
                                min_log_std=self.model_cfg["policy"]["min_log_std"],
                                max_log_std=self.model_cfg["policy"]["max_log_std"],
                                squash=self.is_squashed,
                                device=self.device)
                    critic = Critic(num_states=state_size[uid],
                                    device=self.device)
                
                else:
                    critic = Critic(num_states=state_size[uid],
                                    device=self.device)
                
                models[uid] = {
                    'actor': actor,
                    'critic': critic}

        elif self.is_multi_agent is False:
            # Single Agent
            if self.model_type_lower == "mlp":
                self.is_shared = False
                actor = Actor(num_observations=observation_size,
                                num_actions=action_size,
                                min_log_std=self.model_cfg["policy"]["min_log_std"],
                                max_log_std=self.model_cfg["policy"]["max_log_std"],
                                squash=self.is_squashed,
                                device=self.device)
                critic = Critic(num_states=state_size,
                                device=self.device)
                
            models = {
                'actor': actor,
                'critic': critic}
            
        return models
            

    def generate_gnn_models(self,
                            observation_space: gym.Space,
                            state_space: Union[gym.Space, None],
                            action_space: gym.Space,
                            node_cfg: dict = None,
                            mapping_cfg: dict = None):
        
        if self.model_type_lower == "nervenet":
            actor = NerveNetPolicy(
                    observation_space=observation_space,
                    action_space=action_space,
                    node_info=node_cfg["node_info"],
                    device=self.device,
                    num_nodes=node_cfg["num_nodes"],
                    num_actuated_nodes=node_cfg["num_actuated_nodes"],
                    min_log_std=self.model_cfg['policy']['min_log_std'],
                    max_log_std=self.model_cfg['policy']['max_log_std'],
                )

            if state_size is None:
                raise RuntimeError("Nervenet should have state space for asyncronous actor critic")
            state_size = state_space.shape[-1]

            critic = Critic(num_states=state_size,
                            device=self.device)

        elif self.model_type_lower == "bodytransformer":
            mapping = Mapping(mapping_cfg)
            use_mlp = self.model_cfg.get("use_mlp", False)
            action_detokenizer = ActionDetokenizer(mapping=mapping,
                                                    action_dim=action_space.shape[0], 
                                                    device=self.device)
            value_detokenizer = ValueDetokenizer(mapping=mapping,
                                                use_mlp=use_mlp, 
                                                device=self.device)
            if self.is_shared:
                if state_space is not None:
                    raise RuntimeError("Shared structure can't use state space different from observation sapce.")
                
                tokenizer = ObsTokenizer(mapping=mapping,
                                        device=self.device)
                trunk = BodyTransformer(mapping=mapping,
                                        device=self.device)

                actor = BodyLevelActor(
                    observation_space=observation_space,
                    action_space=action_space,
                    mapping=mapping,
                    tokenizer=tokenizer,
                    trunk=trunk,
                    detokenizer=action_detokenizer,
                    device=self.device,
                    min_log_std=self.model_cfg['policy']['min_log_std'],
                    max_log_std=self.model_cfg['policy']['max_log_std'],)
                
                critic = BodyLevelCritic(
                    state_space=observation_space,
                    mapping=mapping,
                    tokenizer=tokenizer,
                    trunk=trunk,
                    detokenizer=value_detokenizer,
                    device=self.device)
                
            else:
                actor = BodyLevelActor(
                    observation_space=observation_space,
                    action_space=action_space,
                    mapping=mapping,
                    tokenizer=ObsTokenizer(mapping=mapping,
                                        device=self.device),
                    trunk=BodyTransformer(mapping=mapping,
                                        device=self.device),
                    detokenizer=action_detokenizer,
                    device=self.device,
                    min_log_std=self.model_cfg['policy']['min_log_std'],
                    max_log_std=self.model_cfg['policy']['max_log_std'],)
                
                critic = BodyLevelCritic(
                    state_space=observation_space if state_space is None else state_space,
                    mapping=mapping,
                    tokenizer=ObsTokenizer(mapping=mapping,
                                        device=self.device),
                    trunk=BodyTransformer(mapping=mapping,
                                        device=self.device),
                    detokenizer=value_detokenizer,
                    device=self.device)

        else:
            raise ValueError(f"Unknown model type specified in cfg: {self.model_type}")

        models = {
            'actor': actor, 
            'critic': critic}
        
        return models