import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones, _get_seq_len, _detect_is_causal_mask
import warnings
from typing import Optional
from torch import Tensor
from torch.nn import functional as F

from lib.utils.graph_utils import Mapping
from lib.model.model import Model


class MixedTransformerEncoder(Model):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True, bias_type='mixed'):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        self.bias_type = bias_type

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False


    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None,
            return_intermediate: bool = False) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # ========= Prepare for sparsity fast path (Not Necessary) =========
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                print("Inside here")
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        if return_intermediate:
            intermediate = []

        # ========== Encode (Necessary) ==========
        for i, mod in enumerate(self.layers):
            if self.bias_type == 'mixed':
                if i % 2 == 0:
                    current_mask = mask
                else:
                    current_mask = None
            elif self.bias_type == 'hard':
                current_mask = mask
            else:
                raise NotImplementedError(self.bias_type)

            output = mod(output, src_mask=current_mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
            if return_intermediate:
                intermediate.append(output)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)
        if return_intermediate:
            return torch.stack(intermediate, dim=1)

        return output


class BodyTransformer(Model):
    def __init__(self, 
                 mapping: Mapping, 
                 input_dim=64, dim_feedforward=256, 
                 nhead=4, num_layers=3, 
                 bias_type='mixed', 
                 device='cuda'):
        """
        BodyTransformer Network
        Reference: https://github.com/carlosferrazza/BodyTransformer
        
        :param mapping: 
        :param input_dim: Encoder input dim (equal to output dim of tokenizer)
        :param dim_feedforward: Hidden layer dim in transformer encoder
        :param nhead: Number of heads of multi-head attention
        :param num_layers: Number of layers of transformer encoder
        :param bias_type: Maksed attention ['mixed', 'hard']. 'hard': all layer is masked. 'mixed': masked-unmasked-maksed ...
        """
        super(BodyTransformer, self).__init__()
        shortest_path_matrix = mapping.shortest_path_matrix
        adjacency_matrix = torch.tensor(shortest_path_matrix < 2, device=device)

        self.nbodies = adjacency_matrix.shape[0]
        self.output_dim = input_dim
        self.device = device

        # We assume (B x nbodies x input_dim) batches
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)

        self.bias_type = bias_type
        self.embed_absolute_position = nn.Embedding(self.nbodies, embedding_dim=input_dim)
        self.encoder = MixedTransformerEncoder(encoder_layer, num_layers=num_layers, bias_type=bias_type)

        self.register_buffer('adjacency_matrix', adjacency_matrix)

    def forward(self, x):
        """
        :params x: [batch_size, nbodies, input_dim]
        """
        limb_indices = torch.arange(0, self.nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding
        x = self.encoder(x, mask=~self.adjacency_matrix, return_intermediate=False)

        return x