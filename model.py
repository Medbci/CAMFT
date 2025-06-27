import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, List

from timm.models.layers import DropPath


########################################################################
# 1. General Transformer Encoder (supports optional CLS Token)
########################################################################
class TransformerEncoderWithCLS(nn.Module):
	def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
	             dropout_rate: float, seq_len: int):
		super().__init__()
		
		self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
		
		encoder_layers = []
		for _ in range(num_layers):
			layer = nn.TransformerEncoderLayer(
				d_model=embed_dim,
				nhead=num_heads,
				dim_feedforward=embed_dim * 4,
				dropout=dropout_rate,
				activation='gelu',
				batch_first=True
			)
			encoder_layers.append(layer)
		self.encoder = nn.Sequential(*encoder_layers)
		self.norm = nn.LayerNorm(embed_dim)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, seq_len, _ = x.shape
		if self.use_cls:
			cls_tokens = self.cls_token.expand(b, -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)
			x = x + self.pos_embedding[:, :seq_len + 1, :]
		else:
			x = x + self.pos_embedding[:, :seq_len, :]
		
		x = self.encoder(x)
		x = self.norm(x)
		return x[:, 0] if self.use_cls else x.mean(dim=1)


########################################################################
# 2. EEG Dual-Modal Fusion Module (fusing modality A and B)
########################################################################
class CrossAttentionFFNBlock(nn.Module):
	"""
	Single-layer cross-attention + FFN module.
	Uses one modality as query to interact with another modality (key/value).
	"""
	
	def __init__(self, dim: int, num_heads: int, dropout_rate: float, drop_path_rate: float):
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
		                                  dropout=dropout_rate, batch_first=True)
		self.norm1 = nn.LayerNorm(dim)
		self.ffn = nn.Sequential(
			nn.Linear(dim, dim * 4),
			nn.GELU(),
			nn.Dropout(dropout_rate),
			nn.Linear(dim * 4, dim),
			nn.Dropout(dropout_rate)
		)
		self.norm2 = nn.LayerNorm(dim)
		self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
	
	def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
		attn_out, _ = self.attn(query=x, key=kv, value=kv)
		x = self.norm1(x + self.drop_path(attn_out))
		ffn_out = self.ffn(x)
		x = self.norm2(x + self.drop_path(ffn_out))
		return x


class EEGFusionBlock(nn.Module):
	"""
	Fuse two EEG modalities A and B:
	  1. Add positional embedding to A and B and project to a unified fusion dimension;
	  2. Use multi-layer bidirectional cross-attention (A→B and B→A) for information exchange;
	  3. Finally, apply normalization and Dropout.
	"""
	
	def __init__(self, input_dim: int, num_heads: int, num_layers: int,
	             fusion_dim: int, dropout_rate: float, drop_path_rate: float, seq_len: int):
		super().__init__()
		assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"
		self.pos_embed_a = nn.Parameter(torch.randn(1, seq_len, input_dim) * 0.02)
		self.pos_embed_b = nn.Parameter(torch.randn(1, seq_len, input_dim) * 0.02)
		self.proj_a = nn.Linear(input_dim, fusion_dim)
		self.proj_b = nn.Linear(input_dim, fusion_dim)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
		self.attn_blocks_a2b = nn.ModuleList([
			CrossAttentionFFNBlock(fusion_dim, num_heads, dropout_rate, dpr[i])
			for i in range(num_layers)
		])
		self.attn_blocks_b2a = nn.ModuleList([
			CrossAttentionFFNBlock(fusion_dim, num_heads, dropout_rate, dpr[i])
			for i in range(num_layers)
		])
		self.norm = nn.LayerNorm(fusion_dim)
		self.dropout = nn.Dropout(dropout_rate)
		
	def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		a = a + self.pos_embed_a
		b = b + self.pos_embed_b
		a = self.proj_a(a)
		b = self.proj_b(b)
		for block_a2b, block_b2a in zip(self.attn_blocks_a2b, self.attn_blocks_b2a):
			a = block_a2b(a, b)
			b = block_b2a(b, a)
		fused = self.norm(a)
		fused = self.dropout(fused)
		return fused


########################################################################
# 3. ECG-Guided Enhancement Module (multi-layer)
########################################################################
class ECGGuidedEnhancementBlock(nn.Module):
	"""
	Single-layer ECG-guided EEG enhancement module:
	  1. Project ECG features to EEG dimension;
	  2. Bidirectional cross-attention for information exchange;
	  3. Dynamic gating fusion and further feature enhancement.
	"""
	
	def __init__(self, eeg_dim: int, ecg_dim: int, expansion_ratio: float = 0.25, num_heads: int = 4):
		super().__init__()
		hidden_dim = int(eeg_dim * expansion_ratio)
		self.adapter = nn.Sequential(
			nn.Linear(ecg_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, eeg_dim)
		)
		self.eeg_attn = nn.MultiheadAttention(eeg_dim, num_heads=num_heads, batch_first=True)
		self.ecg_attn = nn.MultiheadAttention(eeg_dim, num_heads=num_heads, batch_first=True)
		self.gate = nn.Sequential(
			nn.Linear(eeg_dim * 2, eeg_dim),
			nn.Sigmoid()
		)
		self.enhance = nn.Sequential(
			nn.Linear(eeg_dim, eeg_dim * 2),
			nn.GLU(dim=-1),
			nn.LayerNorm(eeg_dim)
		)
	
	def forward(self, eeg: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
		ecg_proj = self.adapter(ecg).unsqueeze(1)  # [B, 1, eeg_dim]
		eeg_att, _ = self.eeg_attn(query=eeg, key=ecg_proj, value=ecg_proj)
		ecg_att, _ = self.ecg_attn(query=ecg_proj, key=eeg, value=eeg)
		combined = torch.cat([eeg_att, ecg_att], dim=-1)
		gate_val = self.gate(combined)
		enhanced = eeg + gate_val * ecg_att
		enhanced = self.enhance(enhanced)
		return enhanced


class ECGGuidedEnhancement(nn.Module):
	"""
	Multi-layer ECG-guided enhancement module, stacking basic units for multi-layer interaction.
	"""
	
	def __init__(self, eeg_dim: int, ecg_dim: int, n_layers: int = 1,
	             expansion_ratio: float = 0.25, num_heads: int = 4):
		super().__init__()
		layers = [ECGGuidedEnhancementBlock(eeg_dim, ecg_dim, expansion_ratio, num_heads)
		          for _ in range(n_layers)]
		self.layers = nn.Sequential(*layers)
	
	def forward(self, eeg: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
		if eeg.ndim == 2:
			eeg = eeg.unsqueeze(1)
		out = eeg
		for layer in self.layers:
			out = layer(out, ecg)
		return out.squeeze(1)

########################################################################
# 5. Main Model: CAMT
########################################################################
class CAMT(nn.Module):
	# Cross-Attention Multimodal Fusion Transformer
	def __init__(
			self,
			eeg_input_dim: int = 310,
			num_classes: int = 4,
			dropout_rate: float = 0.2,
			drop_path_rate: float = 0.5,
			eeg_seq_len: int = 4,
			ecg_seq_len: int = 1,
			fusion_dim_eeg: int = 256,
			fusion_heads: int = 16,
			fusion_layers: int = 2,
			encoder_heads: int = 4,
			encoder_layers: int = 2,
			ecg_embed_dim: int = 19,
			ecg_transformer_dim: int = 256,
			ecg_heads: int = 16,
			ecg_layers: int = 4,
			ecg_enhance_layers: int = 4,
			ecg_enhance_heads: int = 16,
			ecg_enhance_expansion: float = 4
	):
		super().__init__()
		# EEG fusion module: interactively fuse two EEG modalities A and B (shape [B, eeg_seq_len, eeg_input_dim])
		self.eeg_fusion = EEGFusionBlock(
			input_dim=eeg_input_dim,
			num_heads=fusion_heads,
			num_layers=fusion_layers,
			fusion_dim=fusion_dim_eeg,
			dropout_rate=dropout_rate,
			drop_path_rate=drop_path_rate,
			seq_len=eeg_seq_len
		)
		# EEG global feature extraction: Transformer Encoder with CLS token
		self.eeg_encoder = TransformerEncoderWithCLS(
			embed_dim=fusion_dim_eeg,
			num_heads=encoder_heads,
			num_layers=encoder_layers,
			dropout_rate=dropout_rate,
			seq_len=eeg_seq_len
		)
		# ECG module: project through a linear layer, then use Transformer Encoder to obtain ECG features
		self.ecg_trans = nn.Linear(ecg_embed_dim, ecg_transformer_dim)

		self.ecg_encoder = TransformerEncoderWithCLS(
			embed_dim=ecg_transformer_dim,
			num_heads=ecg_heads,
			num_layers=ecg_layers,
			dropout_rate=dropout_rate,
			seq_len=ecg_seq_len,
		)
		self.ecg_enhancement = ECGGuidedEnhancement(
			eeg_dim=fusion_dim_eeg,
			ecg_dim=ecg_transformer_dim,
			n_layers=ecg_enhance_layers,
			expansion_ratio=ecg_enhance_expansion,
			num_heads=ecg_enhance_heads
		)

		
		# Classifier
		self.classifier = nn.Sequential(
			nn.Linear(fusion_dim_eeg, fusion_dim_eeg // 2),
			nn.LayerNorm(fusion_dim_eeg // 2),
			nn.GELU(),
			nn.Linear(fusion_dim_eeg // 2, num_classes)
		)
		
	
	def forward_features(self, eeg_inputs: List[torch.Tensor],
	                     ecg_input: Optional[torch.Tensor] = None) -> torch.Tensor:
		fused_eeg_seq = self.eeg_fusion(eeg_inputs[0], eeg_inputs[1])
		eeg_features = self.eeg_encoder(fused_eeg_seq)
		if ecg_input is not None:
			if ecg_input.ndim == 2:
				ecg_input = ecg_input.unsqueeze(1)
			ecg_input = self.ecg_trans(ecg_input)
			ecg_features = self.ecg_encoder(ecg_input)
			enhanced_features = self.ecg_enhancement(eeg_features, ecg_features)
			return enhanced_features
		else:
			return eeg_features

	
	def forward(self, eeg: Optional[List[torch.Tensor]] = None,
	            ecg: Optional[torch.Tensor] = None):
		if eeg is None or len(eeg) != 2:
			raise ValueError("Please input a list of two EEG modality tensors.")
		features = self.forward_features(eeg, ecg)
		return self.classifier(features)
