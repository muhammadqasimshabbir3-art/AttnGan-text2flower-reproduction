"""
Minimal Gradio app for inference with the reproduced AttnGAN text->flower models.

This app expects the following files to be present under
`training_results_epoches_79/` (relative to the repo root):
 - `netG_epoch_79.pth`   (generator weights)
 - `text_encoder79.pth`  (text encoder weights)
 - `captions_fixed.pickle` (vocab mappings)

The app loads the text encoder and generator, tokenizes user text, converts to indices
and runs the generator to produce a single image (highest-resolution branch).

Designed to run on CPU or GPU; on Spaces it will use the runtime device.
"""

import os
import pickle
import re
from pathlib import Path

import numpy as np
from nltk.tokenize import RegexpTokenizer
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import gradio as gr

# ---------------------- Minimal config (match training) ---------------------
class Cfg:
	class TEXT: pass
	class GAN: pass
	class TREE: pass
	class TRAIN: pass

cfg = Cfg()
cfg.TEXT.WORDS_NUM = 18
cfg.TEXT.EMBEDDING_DIM = 256
cfg.GAN.GF_DIM = 128
cfg.GAN.Z_DIM = 100
cfg.GAN.CONDITION_DIM = 100
cfg.TREE.BRANCH_NUM = 3  # we will pick the last (highest-res) generated image
cfg.TRAIN.FLAG = True

# ---------------------- Model helper layers (copied/adapted) ----------------
class GLU(nn.Module):
	def forward(self, x):
		nc = x.size(1)
		assert nc % 2 == 0
		nc = int(nc/2)
		return x[:, :nc] * torch.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def upBlock(in_planes, out_planes):
	block = nn.Sequential(
		nn.Upsample(scale_factor=2, mode='nearest'),
		conv3x3(in_planes, out_planes * 2),
		nn.BatchNorm2d(out_planes * 2),
		GLU())
	return block

def Block3x3_relu(in_planes, out_planes):
	block = nn.Sequential(
		conv3x3(in_planes, out_planes * 2),
		nn.BatchNorm2d(out_planes * 2),
		GLU())
	return block

class ResBlock(nn.Module):
	def __init__(self, channel_num):
		super(ResBlock, self).__init__()
		self.block = nn.Sequential(
			conv3x3(channel_num, channel_num * 2),
			nn.BatchNorm2d(channel_num * 2),
			GLU(),
			conv3x3(channel_num, channel_num),
			nn.BatchNorm2d(channel_num))

	def forward(self, x):
		return x + self.block(x)

# Conditioning augmentation net
class CA_NET(nn.Module):
	def __init__(self):
		super(CA_NET, self).__init__()
		self.t_dim = cfg.TEXT.EMBEDDING_DIM
		self.c_dim = cfg.GAN.CONDITION_DIM
		self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
		self.relu = GLU()

	def encode(self, text_embedding):
		x = self.relu(self.fc(text_embedding))
		mu = x[:, :self.c_dim]
		logvar = x[:, self.c_dim:]
		return mu, logvar

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def forward(self, text_embedding):
		mu, logvar = self.encode(text_embedding)
		c_code = self.reparametrize(mu, logvar)
		return c_code, mu, logvar

# Initial stage generator producing low-res feature map
class INIT_STAGE_G(nn.Module):
	def __init__(self, ngf, ncf):
		super(INIT_STAGE_G, self).__init__()
		self.gf_dim = ngf
		self.in_dim = cfg.GAN.Z_DIM + ncf
		self.fc = nn.Sequential(
			nn.Linear(self.in_dim, ngf * 4 * 4 * 2, bias=False),
			nn.BatchNorm1d(ngf * 4 * 4 * 2),
			GLU())
		self.upsample1 = upBlock(ngf, ngf // 2)
		self.upsample2 = upBlock(ngf // 2, ngf // 4)
		self.upsample3 = upBlock(ngf // 4, ngf // 8)
		self.upsample4 = upBlock(ngf // 8, ngf // 16)

	def forward(self, z_code, c_code):
		c_z_code = torch.cat((c_code, z_code), 1)
		out_code = self.fc(c_z_code)
		out_code = out_code.view(-1, self.gf_dim, 4, 4)
		out_code = self.upsample1(out_code)
		out_code = self.upsample2(out_code)
		out_code32 = self.upsample3(out_code)
		out_code64 = self.upsample4(out_code32)
		return out_code64

class ATT_NET(nn.Module):
	def __init__(self, idf, cdf):
		super(ATT_NET, self).__init__()
		self.conv_context = nn.Conv2d(cdf, idf, kernel_size=1, stride=1, padding=0, bias=False)
		self.sm = nn.Softmax(dim=1)
		self.mask = None

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, input, context):
		ih, iw = input.size(2), input.size(3)
		queryL = ih * iw
		batch_size, sourceL = context.size(0), context.size(2)
		target = input.view(batch_size, -1, queryL)
		targetT = torch.transpose(target, 1, 2).contiguous()
		sourceT = context.unsqueeze(3)
		sourceT = self.conv_context(sourceT).squeeze(3)
		attn = torch.bmm(targetT, sourceT)
		attn = attn.view(batch_size*queryL, sourceL)
		if self.mask is not None:
			mask = self.mask.repeat(queryL, 1)
			attn.data.masked_fill_(mask.data.bool(), -float('inf'))
		attn = self.sm(attn)
		attn = attn.view(batch_size, queryL, sourceL)
		attn = torch.transpose(attn, 1, 2).contiguous()
		weightedContext = torch.bmm(sourceT, attn)
		weightedContext = weightedContext.view(batch_size, -1, ih, iw)
		attn = attn.view(batch_size, -1, ih, iw)
		return weightedContext, attn

class NEXT_STAGE_G(nn.Module):
	def __init__(self, ngf, nef, ncf):
		super(NEXT_STAGE_G, self).__init__()
		self.gf_dim = ngf
		self.ef_dim = nef
		self.cf_dim = ncf
		self.num_residual = 2
		self.att = ATT_NET(ngf, self.ef_dim)
		self.residual = nn.Sequential(*[ResBlock(ngf * 2) for _ in range(2)])
		self.upsample = upBlock(ngf * 2, ngf)

	def forward(self, h_code, c_code, word_embs, mask):
		self.att.applyMask(mask)
		c_code_att, att = self.att(h_code, word_embs)
		h_c_code = torch.cat((h_code, c_code_att), 1)
		out_code = self.residual(h_c_code)
		out_code = self.upsample(out_code)
		return out_code, att

class GET_IMAGE_G(nn.Module):
	def __init__(self, ngf):
		super(GET_IMAGE_G, self).__init__()
		self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())
	def forward(self, h_code):
		return self.img(h_code)

class G_NET(nn.Module):
	def __init__(self):
		super(G_NET, self).__init__()
		ngf = cfg.GAN.GF_DIM
		nef = cfg.TEXT.EMBEDDING_DIM
		ncf = cfg.GAN.CONDITION_DIM
		self.ca_net = CA_NET()
		self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
		self.img_net1 = GET_IMAGE_G(ngf)
		self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
		self.img_net2 = GET_IMAGE_G(ngf)
		self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
		self.img_net3 = GET_IMAGE_G(ngf)

	def forward(self, z_code, sent_emb, word_embs, mask):
		fake_imgs = []
		att_maps = []
		c_code, mu, logvar = self.ca_net(sent_emb)
		h_code1 = self.h_net1(z_code, c_code)
		fake_img1 = self.img_net1(h_code1)
		fake_imgs.append(fake_img1)
		h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
		fake_img2 = self.img_net2(h_code2)
		fake_imgs.append(fake_img2)
		if att1 is not None:
			att_maps.append(att1)
		h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
		fake_img3 = self.img_net3(h_code3)
		fake_imgs.append(fake_img3)
		if att2 is not None:
			att_maps.append(att2)
		return fake_imgs, att_maps, mu, logvar

# ---------------------- Text encoder (RNN) ----------------------------------
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN_ENCODER(nn.Module):
	def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=256, nlayers=1, bidirectional=True):
		super(RNN_ENCODER, self).__init__()
		self.n_steps = cfg.TEXT.WORDS_NUM
		self.ntoken = ntoken
		self.ninput = ninput
		self.drop_prob = drop_prob
		self.nlayers = nlayers
		self.bidirectional = bidirectional
		self.rnn_type = 'LSTM'
		if bidirectional:
			self.num_directions = 2
		else:
			self.num_directions = 1
		self.nhidden = nhidden // self.num_directions
		self.encoder = nn.Embedding(self.ntoken, self.ninput)
		self.drop = nn.Dropout(self.drop_prob)
		self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers, batch_first=True,
						   dropout=0 if self.nlayers == 1 else self.drop_prob,
						   bidirectional=self.bidirectional)

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhidden),
					weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhidden))
		else:
			return weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhidden)

	def forward(self, captions, cap_lens, hidden, mask=None):
		emb = self.drop(self.encoder(captions))
		cap_lens = cap_lens.data.tolist() if isinstance(cap_lens, torch.Tensor) else [int(cap_lens)]
		emb_packed = pack_padded_sequence(emb, cap_lens, batch_first=True, enforce_sorted=False)
		output, hidden = self.rnn(emb_packed, hidden)
		output = pad_packed_sequence(output, batch_first=True)[0]
		words_emb = output.transpose(1, 2)
		if self.rnn_type == 'LSTM':
			sent_emb = hidden[0].transpose(0, 1).contiguous()
		else:
			sent_emb = hidden.transpose(0, 1).contiguous()
		sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
		return words_emb, sent_emb

# ---------------------- Utilities and model loading -------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'training_results_epoches_79'

def load_captions_pickle(pickle_path):
	with open(pickle_path, 'rb') as f:
		data = pickle.load(f)
	# Support different formats from notebook: list/tuple or dict
	if isinstance(data, (list, tuple)):
		# [train_captions, test_captions, ixtoword, wordtoix]
		if len(data) >= 4:
			ixtoword = data[2]
			wordtoix = data[3]
		else:
			raise RuntimeError('Unexpected captions_fixed.pickle format')
	elif isinstance(data, dict):
		ixtoword = data.get('ixtoword') or data.get('ix_to_word')
		wordtoix = data.get('wordtoix') or data.get('word_to_ix')
		if ixtoword is None or wordtoix is None:
			# try common keys
			keys = list(data.keys())
			# fallback: if dict maps indices->words
			if all(isinstance(k, int) for k in keys):
				ixtoword = {k: data[k] for k in keys}
				wordtoix = {v: k for k, v in ixtoword.items()}
			else:
				raise RuntimeError('captions_fixed.pickle missing ixtoword/wordtoix')
	else:
		raise RuntimeError('Unsupported captions_fixed.pickle format')
	# Ensure ixtoword is a dict mapping ints->str
	if isinstance(ixtoword, list):
		ixtoword = {i: w for i, w in enumerate(ixtoword)}
	return ixtoword, wordtoix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocabulary and models at import time for faster Gradio responses
_models_ready = False
_vocab = None
_wordtoix = None
_text_encoder = None
_netG = None

def prepare_models():
	global _models_ready, _vocab, _wordtoix, _text_encoder, _netG
	if _models_ready:
		return
	# load captions mapping
	captions_path = MODEL_DIR / 'captions_fixed.pickle'
	if not captions_path.exists():
		# try in nested dir
		captions_path = MODEL_DIR / 'captions_fixed.pickle'
	if not captions_path.exists():
		raise FileNotFoundError(f'captions_fixed.pickle not found at {captions_path}')
	ixtoword, wordtoix = load_captions_pickle(captions_path)
	_vocab = ixtoword
	_wordtoix = wordtoix

	# instantiate text encoder
	n_words = len(_vocab)
	_text_encoder = RNN_ENCODER(n_words, ninput=cfg.TEXT.EMBEDDING_DIM, nhidden=cfg.TEXT.EMBEDDING_DIM)
	text_encoder_path = MODEL_DIR / 'Flowers_2026_05_17_18_29_29' / 'Model' / 'text_encoder79.pth'
	if not text_encoder_path.exists():
		# fallback top-level model
		text_encoder_path = MODEL_DIR / 'text_encoder79.pth'
	if not text_encoder_path.exists():
		raise FileNotFoundError(f'text_encoder79.pth not found at {text_encoder_path}')
	state = torch.load(text_encoder_path, map_location='cpu')
	if 'state_dict' in state and isinstance(state['state_dict'], dict):
		state = state['state_dict']
	_text_encoder.load_state_dict(state)
	_text_encoder.to(device)
	_text_encoder.eval()

	# instantiate generator
	_netG = G_NET()
	netG_path = MODEL_DIR / 'netG_epoch_79.pth'
	if not netG_path.exists():
		netG_path = MODEL_DIR / 'netG_epoch_79.pth'
	if not netG_path.exists():
		raise FileNotFoundError(f'netG_epoch_79.pth not found at {netG_path}')
	state = torch.load(netG_path, map_location='cpu')
	if 'state_dict' in state and isinstance(state['state_dict'], dict):
		state = state['state_dict']
	_netG.load_state_dict(state)
	_netG.to(device)
	_netG.eval()

	_models_ready = True

# ---------------------- Text processing & inference ------------------------
tokenizer = RegexpTokenizer(r"\w+")

def text_to_indices(text, wordtoix, max_len=cfg.TEXT.WORDS_NUM):
	text = text.lower()
	tokens = tokenizer.tokenize(text)
	tokens = [t.encode('ascii', 'ignore').decode('ascii') for t in tokens]
	inds = []
	for t in tokens[:max_len]:
		inds.append(wordtoix.get(t, 0))
	if len(inds) < max_len:
		inds = inds + [0] * (max_len - len(inds))
	return np.array(inds, dtype='int64'), min(len(tokens), max_len)

def tensor_to_pil(img_tensor):
	img = img_tensor.detach().cpu().numpy()
	img = (img + 1.0) / 2.0 * 255.0
	img = img.clip(0, 255).astype('uint8')
	img = img[0].transpose(1, 2, 0)
	return Image.fromarray(img)

def generate_image_from_text(text, seed=None):
	prepare_models()
	global _vocab, _wordtoix, _text_encoder, _netG
	if seed is not None and seed != 0:
		torch.manual_seed(int(seed))
	inds, cap_len = text_to_indices(text, _wordtoix)
	captions = torch.LongTensor(inds).unsqueeze(0).to(device)
	cap_lens = torch.LongTensor([cap_len]).to(device)
	hidden = _text_encoder.init_hidden(1)
	if isinstance(hidden, tuple):
		hidden = (hidden[0].to(device), hidden[1].to(device))
	else:
		hidden = hidden.to(device)
	with torch.no_grad():
		words_embs, sent_emb = _text_encoder(captions, cap_lens, hidden)
		words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
		mask = (captions == 0)
		if mask.size(1) > words_embs.size(2):
			mask = mask[:, :words_embs.size(2)]
		z = torch.randn(1, cfg.GAN.Z_DIM).to(device)
		fake_imgs, att_maps, mu, logvar = _netG(z, sent_emb, words_embs, mask)
		img_tensor = fake_imgs[-1]
		pil = tensor_to_pil(img_tensor)
		return pil

# ---------------------- Gradio interface -----------------------------------
def launch_gradio():
	demo = gr.Interface(
		fn=generate_image_from_text,
		inputs=[gr.Textbox(lines=2, placeholder='Enter a descriptive sentence about a flower...', label='Text prompt'),
				gr.Number(label='seed (optional)', value=None, precision=0)],
		outputs=gr.Image(type='pil'),
		title='AttnGAN: text -> flower (inference)',
		description='Enter a description and the model will generate a flower image.'
	)
	demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))

if __name__ == '__main__':
	print('Starting Gradio app...')
	launch_gradio()


