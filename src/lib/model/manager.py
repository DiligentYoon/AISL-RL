import torch

from typing import Union, Any

from lib.model.MLP import Actor, Critic
from lib.model.NerveNet import NerveNetPolicy
from lib.utils.graph_utils import Mapping
from lib.model.BodyTransformer.body_transformer import BodyLevelActor, BodyLevelCritic
from lib.model.BodyTransformer.linear_components import ObsTokenizer, ValueDetokenizer, ActionDetokenizer
from lib.model.BodyTransformer.transformer_components import BodyTransformer

class ActorCriticManager:
    def __init__(self,
                 cfg: dict[str, Union[dict, Any]],
                 device: Union[torch.device, str]):
        
        self.models = {}

        self.model_cfg = cfg["models"]
        self.agent_cfg = cfg["agent"]
        self.is_shared = self.model_cfg.get("shared", False)
        self.is_squashed = self.model_cfg.get("squashed", False)
        self.is_cooperative = self.model_cfg.get("cooperative", False)
        self.is_multi_agent = self.model_cfg.get("multi_agent", False)
        
        self.model_type = self.model_cfg["policy"].get("type", None)

        self.device = device


    def generate_models(self,
                        observation_size: Union[dict[str, int], int],
                        state_size: Union[dict[str, int], int],
                        action_size: Union[dict[str, int], int],
                        possible_agents: list[str] = None):
        
        if (possible_agents is None) and (not self.is_multi_agent):
            raise RuntimeError("Please confirm the cfg file whether multi_agent is assigned true or false")

        if self.is_multi_agent:
            for uid in possible_agents:
                # Multi Agent : Per-agent network
                actor = Actor(num_observations=observation_size[uid],
                            num_actions=action_size[uid],
                            min_log_std=self.model_cfg["policy"]["min_log_std"],
                            max_log_std=self.model_cfg["policy"]["max_log_std"],
                            squash=self.is_squashed,
                            device=self.device)
                critic = Critic(num_states=state_size[uid],
                                device=self.device)
                
                self.models[uid] = {
                    'actor': actor,
                    'critic': critic}
            
            else:
                # Single Agent
                if self.model_type is None:
                    actor = Actor(num_observations=observation_size,
                                num_actions=action_size,
                                min_log_std=self.model_cfg["policy"]["min_log_std"],
                                max_log_std=self.model_cfg["policy"]["max_log_std"],
                                squash=self.is_squashed,
                                device=self.device)
                    critic = Critic(num_states=state_size,
                                    device=self.device)
                
                else:
                    model_type_lower = self.model_type.lower()
                    if model_type_lower == "nervenet":
                        actor = NerveNetPolicy(
                            observation_space=observation_space,
                            action_space=action_space,
                            node_info=env._unwrapped.cfg.node_info,
                            device=env.device,
                            num_nodes=env._unwrapped.cfg.num_nodes,
                            num_actuated_nodes=env._unwrapped.cfg.num_actuated_nodes,
                            min_log_std=cfg['models']['policy']['min_log_std'],
                            max_log_std=cfg['models']['policy']['max_log_std'],
                        )

                        critic = Critic(num_states=state_size,
                                        device=env.device)
                        
                    elif model_type_lower == "bodytransformer":
                        mapping = Mapping(env._unwrapped.cfg.map_info)
                        use_mlp = cfg["models"].get("use_mlp", False)
                        action_detokenizer = ActionDetokenizer(mapping=mapping,
                                                            action_dim=action_space.shape[0], 
                                                            device=env.device)
                        value_detokenizer = ValueDetokenizer(mapping=mapping,
                                                            use_mlp=use_mlp, 
                                                            device=env.device)
                        if is_shared:
                            if state_space is not None:
                                raise RuntimeError("Shared structure should not use state space different from observation sapce.")
                            
                            tokenizer = ObsTokenizer(mapping=mapping,
                                                    device=env.device)
                            trunk = BodyTransformer(mapping=mapping,
                                                    device=env.device)

                            actor = BodyLevelActor(
                                observation_space=observation_space,
                                action_space=action_space,
                                mapping=mapping,
                                tokenizer=tokenizer,
                                trunk=trunk,
                                detokenizer=action_detokenizer,
                                device=env.device,
                                min_log_std=cfg['models']['policy']['min_log_std'],
                                max_log_std=cfg['models']['policy']['max_log_std'],)
                            
                            critic = BodyLevelCritic(
                                state_space=observation_space,
                                mapping=mapping,
                                tokenizer=tokenizer,
                                trunk=trunk,
                                detokenizer=value_detokenizer,
                                device=env.device)
                            
                        else:
                            actor = BodyLevelActor(
                                observation_space=observation_space,
                                action_space=action_space,
                                mapping=mapping,
                                tokenizer=ObsTokenizer(mapping=mapping,
                                                    device=env.device),
                                trunk=BodyTransformer(mapping=mapping,
                                                    device=env.device),
                                detokenizer=action_detokenizer,
                                device=env.device,
                                min_log_std=cfg['models']['policy']['min_log_std'],
                                max_log_std=cfg['models']['policy']['max_log_std'],)
                            
                            critic = BodyLevelCritic(
                                state_space=observation_space if state_space is None else state_space,
                                mapping=mapping,
                                tokenizer=ObsTokenizer(mapping=mapping,
                                                    device=env.device),
                                trunk=BodyTransformer(mapping=mapping,
                                                    device=env.device),
                                detokenizer=value_detokenizer,
                                device=env.device)
            