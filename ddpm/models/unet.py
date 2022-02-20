import torch
import torch.nn as nn
from torch.nn import functional as F

def embed_timestep(timesteps, embedding_dim):
	"""
	input: list of timesteps, size of embedding (512 in transformer paper)
	use first half sine, second half cosine embedding
	return: embedded list of timesteps [-1, 1]
	"""

	half_dim = embedding_dim // 2
	w_k = 1/ 10000^(2 * torch.arange(half_dim)/embedding)

	timesteps = torch.cat([torch.sin(w_k * ])

	return timesteps

def swish(x):
	return x * torch.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
	
	def forward(x):
		x = F.interpolate(x, scale_factor=2)
		x = self.conv(x) # why conv?
		return x

class Downsample(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)

	def forward(x):
		x = self.conv(x)
		return x

class AttnBlock(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.normalize = Normalize(in_channels)
		# define Q, K, V vectors as trivial convolutions
		# kernel size, stride, padding
		self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
		self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
		self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
		self.proj = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

	def forward(x):
		N,C,H,W = x.shape
		h = self.normalize(x)
		q, k, v = self.q(h), self.k(h), self.v(h)

		q = q.reshape(N,C,H*W)
		q = q.permute(0, 2, 1) # N, HW, C
		k = k.reshape(N,C,H*W)
		sim = torch.bmm(q, k) # N, HW, HW
		sim /= sqrt(C)
		sim = F.softmax(sim, dim=2)

		v = v.reshape(N,C,H*W)
		v = v.permute(0, 2, 1)
		h = torch.bmm(v, sim) # N, HW, HW
		h = h.view(N,C,H,W).permute(0, 3, 1, 2)
		h = self.proj(h)

		return x + h

class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, dropout, tdim):
		super().__init__()

		self.norm1 = Normalize(in_channels)
		self.conv1 = nn.Conv2d(in_channels,out_channels,3,1,1)
		self.temb_proj = nn.Linear(tdim,out_channels)
		self.norm2 = Normalize(out_channels)
		self.dropout = nn.Dropout(dropout)
		self.conv2 = nn.Conv2d(out_channels,out_channels,3,1,1)
		self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)# what is this?
	
	def forward(x, temb):
		h = x
		h = self.norm1(h)
		h = swish(h)
		h = self.conv1(h)

		h = h + self.temb_proj(temb)

		h = self.norm2(h)
		h = swish(h)
		h = self.dropout(h)
		h = self.conv2(h)

		x = self.shortcut(x)

		return x + h

class UNet(nn.Module):
	def __init__(self, ch, out_ch, ch_mult, attn, num_res_blocks, dropout, device=None):
		"""
		attn: list of layers at which we apply attention
		ch_mult: tuple of channel sizes in UNET
		"""
		# default params
		self.embedding_dim = 512
		self.head = nn.Conv2d(3, ch, 3, 1, 1)

		self.ch = ch
		self.out_ch = out_ch
		self.temb_ch = self.ch * 4
		self.num_resolutions = len(ch_mult)
		self.num_res_blocks = num_res_blocks
	
		chs = [ch] # record of all the channel sizes
		curr_channel = ch

		# downsampling blocks
		self.downblocks = nn.ModuleList()
		for i, mult in enumerate(ch_mult):
			res_blocks = nn.ModuleList()
			attn_blocks = nn.ModuleList()

			out_ch = ch * mult
			for _ in range(num_res_blocks):
				res_blocks.append(ResBlock(curr_channel,out_ch,dropout,temb_ch))
				if i in attn:
					attn_blocks.append(AttnBlock(curr_channel))
			curr_channel = out_ch
			chs.append(curr_channel)

			down = nn.Module() # specific to this channel size
			down.res_block = res_blocks
			down.attn_blocks = attn_blocks
			if i != len(ch_mult) - 1:
				down.downsample = Downsample(curr_channel)
			self.downblocks.append(down)

		# middle blocks
		self.mid = nn.Module()
		self.mid.block1 = ResBlock(curr_channel, curr_channel, dropout,temb_ch)
		self.mid.attn1 = AttnBlock(curr_channel)
		self.mid.block2 = ResBlock(curr_channel, curr_channel, dropout,temb_ch)

		# upsampling blocks
		self.upblocks = nn.ModuleList()
		for i, mult in reversed(enumerate(ch_mult)):
			res_blocks = nn.ModuleList()
			attn_blocks = nn.ModuleList()

			out_ch = ch * mult
			for _ in range(num_res_blocks+1):
				res_blocks.append(ResBlock(curr_channel + chs.pop(), out_ch, dropout, temb_ch))
				curr_channel = out_ch
			
				if i in attn:
					attn_blocks.append(AttnBlock(curr_channel))
			
			curr_channel = out_ch

			up = nn.Module()
			up.res_block = res_blocks
			up.attn_blocks = attn_blocks
			if i != 0:
				up.upsample = Upsample(curr_channel)

			self.upblocks.insert(0, up) # prepend for same order

		# end
		self.norm_out = Normalize(curr_channel)
		self.conv_out = nn.Conv2d(curr_channel, out_ch, 3, 1, 1)


	def forward(self, x, t):
		
		temb = embed_timestep(t, self.embedding_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

		h = self.head(x)
		hs = [h]

		# downsample
		for i_level in range(len(self.downblocks)):
			layer = self.downblocks[i_level]
			for i_block in range(self.num_res_blocks):
				h = layer.res_blocks[i_block](hs[-1], temb)
				if len(layer.attn_blocks) > 0:
					h = layer.attn_blocks[i_block](h)
				hs.append(h)
			if i_level != len(self.downblocks) - 1:
				hs.append(layer.downsample(hs[-1]))

		# middle
		h = hs[-1]
		h = self.mid.block1(h, temb)
		h = self.mid.attn1(h)
		h = self.mind.block2(h, temb)

		for i_level in reversed(range(len(self.upblocks))):
			layer = self.upblocks[i_level]
			for i_block in range(self.num_res_blocks + 1):
				h = layer.res_block[i_block](torch.cat([h, hs.pop()]), temb)
				if len(layer.attn_blocks)>0:
					h = layer.attn_blocks[i_block](h)
			if i_level != 0:
				h = layer.upsample(h)
		
		h = self.norm_out(h)
		h = swish(h)
		h = self.conv_out(h)
		return h
