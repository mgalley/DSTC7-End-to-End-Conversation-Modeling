import os, random, sys
import numpy as np
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from create_input_files import SOS_token, EOS_token

"""
a simple seq2seq model prepared as a baseline model for DSTC7
https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling

following Keras tutorial: 
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

NOTE:
*	word-level, GRU-based
* 	no attention mechanism
* 	no beam search

CONTACT: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

def set_random_seed(seed=912):
	random.seed(seed)
	np.random.seed(seed)


def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)

class Dataset:

	"""
	assumptions of the data files
	* SOS and EOS are top 2 tokens
	* dictionary ordered by frequency
	"""

	def __init__(self, 
		path_source, path_target, path_vocab, 
		max_num_sample=None,
		max_num_token=None,
		max_seq_len=32,
		test_split=0.2,		# how many hold out as test data
		):


		# load token dictionary

		self.index2token = {0: ''}
		self.token2index = {'': 0}
		self.max_seq_len = max_seq_len

		with open(path_vocab, encoding="utf-8") as f:
			lines = f.readlines()
		if max_num_token is not None:
			lines = lines[:max_num_token]
		for i, line in enumerate(lines):
			token = line.strip('\n').strip()
			assert(len(token)>0)
			self.index2token[i + 1] = token
			self.token2index[token] = i + 1

		assert(SOS_token in self.token2index)
		assert(EOS_token in self.token2index)

		self.SOS = self.token2index[SOS_token]
		self.EOS = self.token2index[EOS_token]
		self.num_tokens = len(self.token2index) - 1	# not including 0-th


		# load source-target pairs, tokenized

		seqs = dict()
		for k, path in [('source', path_source), ('target', path_target)]:
			seqs[k] = []
			with open(path, encoding="utf-8") as f:
				lines = f.readlines()
			if max_num_sample is not None:
				lines = lines[:min(max_num_sample, len(lines))]
			for line in lines:
				seq = []
				for c in line.strip('\n').strip().split(' '):
					i = int(c)
					if i <= self.num_tokens:	# delete the "unkown" words
						seq.append(i)
				seqs[k].append(seq[:min(self.max_seq_len - 2, len(seq))])
				max_seq_len = max(max_seq_len, len(seq))
		self.pairs = list(zip(seqs['source'], seqs['target']))

		# train-test split

		np.random.shuffle(self.pairs)
		self.n_train = int(len(self.pairs) * (1. - test_split))

		self.i_sample_range = {
			'train': (0, self.n_train),
			'test': (self.n_train, len(self.pairs)),
			}
		self.i_sample = dict()
		self.reset()


	def reset(self):
		for task in self.i_sample_range:
			self.i_sample[task] = self.i_sample_range[task][0]

	def all_loaded(self, task):
		return self.i_sample[task] == self.i_sample_range[task][1]

	def load_data(self, task, max_num_sample_loaded=None):

		i_sample = self.i_sample[task]
		if max_num_sample_loaded is None:
			max_num_sample_loaded = self.i_sample_range[task][1] - i_sample
		i_sample_next = min(i_sample + max_num_sample_loaded, self.i_sample_range[task][1])
		num_samples = i_sample_next - i_sample
		self.i_sample[task] = i_sample_next

		print('building %s data from %i to %i'%(task, i_sample, i_sample_next))
		
		encoder_input_data = np.zeros((num_samples, self.max_seq_len))
		decoder_input_data = np.zeros((num_samples, self.max_seq_len))
		decoder_target_data = np.zeros((num_samples, self.max_seq_len, self.num_tokens + 1))		# +1 as mask_zero

		source_texts = []
		target_texts = []

		for i in range(num_samples):

			seq_source, seq_target = self.pairs[i_sample + i]
			if not bool(seq_target) or not bool(seq_source):
				continue

			if seq_target[-1] != self.EOS:
				seq_target.append(self.EOS)

			source_texts.append(' '.join([self.index2token[j] for j in seq_source]))
			target_texts.append(' '.join([self.index2token[j] for j in seq_target]))

			for t, token_index in enumerate(seq_source):
				encoder_input_data[i, t] = token_index

			decoder_input_data[i, 0] = self.SOS
			for t, token_index in enumerate(seq_target):
				decoder_input_data[i, t + 1] = token_index
				decoder_target_data[i, t, token_index] = 1.

		return encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts




class Seq2Seq:

	def __init__(self, 
		dataset, model_dir, 
		token_embed_dim, rnn_units, encoder_depth, decoder_depth, dropout_rate=0.2):

		self.token_embed_dim = token_embed_dim
		self.rnn_units = rnn_units
		self.encoder_depth = encoder_depth
		self.decoder_depth = decoder_depth
		self.dropout_rate = dropout_rate
		self.dataset = dataset

		makedirs(model_dir)
		self.model_dir = model_dir


	def load_models(self):
		self.model_train = load_model(os.path.join(self.model_dir, 'model.h5'))
		self.build_model_test()


	def _stacked_rnn(self, rnns, inputs, initial_states=None):
		if initial_states is None:
			initial_states = [None] * len(rnns)
		outputs, state = rnns[0](inputs, initial_state=initial_states[0])
		states = [state]
		for i in range(1, len(rnns)):
			outputs, state = rnns[i](outputs, initial_state=initial_states[i])
			states.append(state)
		return outputs, states


	def build_model_train(self):

		# build layers
		embeding = Embedding(
				self.dataset.num_tokens + 1,		# +1 as mask_zero 
				self.token_embed_dim, mask_zero=True, 
				name='embeding')

		encoder_inputs = Input(shape=(None,), name='encoder_inputs')
		encoder_rnns = []
		for i in range(self.encoder_depth):
			encoder_rnns.append(GRU(
				self.rnn_units, 
				return_state=True,
				return_sequences=True, 
				name='encoder_rnn_%i'%i,
				))

		decoder_inputs = Input(shape=(None,), name='decoder_inputs')
		decoder_rnns = []
		for i in range(self.decoder_depth):
			decoder_rnns.append(GRU(
				self.rnn_units, 
				return_state=True,
				return_sequences=True, 
				name='decoder_rnn_%i'%i,
				))

		decoder_dense = Dense(
			self.dataset.num_tokens + 1, 		# +1 as mask_zero
			activation='softmax', name='decoder_dense')

		# set connections: teacher forcing

		encoder_outputs, encoder_states = self._stacked_rnn(
				encoder_rnns, embeding(encoder_inputs))

		decoder_outputs, decoder_states = self._stacked_rnn(
				decoder_rnns, embeding(decoder_inputs), [encoder_states[-1]] * self.decoder_depth)

		decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
		decoder_outputs = decoder_dense(decoder_outputs)
		self.model_train = Model(
				[encoder_inputs, decoder_inputs], 	# [input sentences, ground-truth target sentences],
				decoder_outputs)					# shifted ground-truth sentences


	def build_model_test(self):

		# load/build layers

		names = ['embeding', 'decoder_dense']
		for i in range(self.encoder_depth):
			names.append('encoder_rnn_%i'%i)
		for i in range(self.decoder_depth):
			names.append('decoder_rnn_%i'%i)

		reused = dict()
		for name in names:
			reused[name] = self.model_train.get_layer(name)
		
		encoder_inputs = Input(shape=(None,), name='encoder_inputs')
		decoder_inputs = Input(shape=(None,), name='decoder_inputs')
		decoder_inital_states = []
		for i in range(self.decoder_depth):
			decoder_inital_states.append(Input(shape=(self.rnn_units,), name="decoder_inital_state_%i"%i))

		# set connections: autoregressive

		encoder_outputs, encoder_states = self._stacked_rnn(
				[reused['encoder_rnn_%i'%i] for i in range(self.encoder_depth)], 
				reused['embeding'](encoder_inputs))
		self.model_infer_encoder = Model(encoder_inputs, encoder_states[-1])

		decoder_outputs, decoder_states = self._stacked_rnn(
				[reused['decoder_rnn_%i'%i] for i in range(self.decoder_depth)], 
				reused['embeding'](decoder_inputs),
				decoder_inital_states)

		decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
		decoder_outputs = reused['decoder_dense'](decoder_outputs)
		self.model_infer_decoder = Model(
				[decoder_inputs] + decoder_inital_states,
				[decoder_outputs] + decoder_states)


	def save_model(self, name):
		path = os.path.join(self.model_dir, name)
		self.model_train.save(path)
		print('saved to: '+path)


	def train(self, 
		batch_size, epochs, 
		batch_per_load=10,
		lr=0.001):


		self.model_train.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

		for epoch in range(epochs):
			load = 0
			self.dataset.reset()
			while not self.dataset.all_loaded('train'):
				load += 1
				print('\n***** Epoch %i/%i - load %i/%i *****'%(epoch + 1, epochs, load, np.ceil(self.dataset.n_train/batch_size/batch_per_load)))
				encoder_input_data, decoder_input_data, decoder_target_data, _, _ = self.dataset.load_data('train', batch_size * batch_per_load)

				self.model_train.fit(
					[encoder_input_data, decoder_input_data], 
					decoder_target_data,
					batch_size=batch_size,)

				self.save_model('model_epoch%i.h5'%(epoch + 1))
		self.save_model('model.h5')


	def test(self, samples_per_load=1000, print_step=100):

		self.dataset.reset()
		avg_NLL = 0.
		avg_perplexity = 0.
		path_log = os.path.join(self.model_dir, 'test_results.txt')
		open(path_log, 'w', encoding="utf-8")
		n = 0
		sum_loss = 0.

		if samples_per_load is None:
			samples_per_load = len(self.dataset.pairs) - self.dataset.n_train

		while not self.dataset.all_loaded('test'):

			encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts = self.dataset.load_data('test', samples_per_load)
			for i in range(len(source_texts)):

				source_seq_mat = encoder_input_data[i: i + 1]
				decoded_sentence = self._infer(source_seq_mat)
				target_text = target_texts[i]

				loss = self.model_train.evaluate(
					x=[encoder_input_data[i:i+1,:], decoder_input_data[i:i+1,:]], 
					y=decoder_target_data[i:i+1,:,:],
					verbose=0)

				if not np.isnan(loss):
					sum_loss += loss
					n += 1

				if n%print_step==0:

					s = '\n'.join([
						'%i'%n+'-'*20,
						'source: \t' + source_texts[i],
						'target: \t' + target_text,
						'pred:   \t' + decoded_sentence,
						'loss:   \t%.2f'%loss,
						'avg loss\t%.2f'%(sum_loss/n),
						'',
						])
					print(s)
					with open(path_log, 'a', encoding="utf-8") as f:
						f.write(s)



	def _infer(self, source_seq_int):

		state = self.model_infer_encoder.predict(source_seq_int)
		prev_word = np.atleast_2d([self.dataset.SOS])
		states = [state] * self.decoder_depth
		decoded_sentence = ''
		t = 0
		while True:

			out = self.model_infer_decoder.predict([prev_word] + states)
			tokens_proba = out[0]
			states = out[1:]
			sampled_token_index = np.argmax(tokens_proba[0, -1, :])
			sampled_token = self.dataset.index2token[sampled_token_index]
			decoded_sentence += sampled_token+' '

			t += 1
			if sampled_token_index == self.dataset.EOS or t > self.dataset.max_seq_len:
				break

			prev_word = np.atleast_2d([sampled_token_index])

		return decoded_sentence



	def dialog(self, input_text):

		source_seq_int = []
		for token in input_text.strip().strip('\n').split(' '):
			if token in self.dataset.token2index:
				source_seq_int.append(self.dataset.token2index[token])
		return self._infer(np.atleast_2d(source_seq_int))

	def interact(self):
		while True:
			print('----- please input -----')
			input_text = input()
			if not bool(input_text):
				break
			print(self.dialog(input_text))



def main(mode):


	token_embed_dim = 100
	rnn_units = 512
	encoder_depth = 2
	decoder_depth = 2
	dropout_rate = 0.5
	learning_rate = 1e-3
	max_num_token = 20000
	max_seq_len = 32

	batch_size = 100
	epochs = 10

	path_source = os.path.join('trial','source_num.txt')
	path_target = os.path.join('trial','target_num.txt')
	path_vocab = os.path.join('trial','dict.txt')

	dataset = Dataset(path_source, path_target, path_vocab, 
		max_num_token=max_num_token, max_seq_len=max_seq_len)
	model_dir = 'model'
	
	s2s = Seq2Seq(dataset, model_dir, 
		token_embed_dim, rnn_units, encoder_depth, decoder_depth, dropout_rate)

	if mode == 'train':
		s2s.build_model_train()
		s2s.train(batch_size, epochs, lr=learning_rate)
	else:
		s2s.load_models()
		if mode == 'test':
			s2s.build_model_test()
			s2s.test()
		elif mode == 'interact':
			s2s.interact()
	

if __name__ == '__main__':
	set_random_seed()
	main(sys.argv[1])
