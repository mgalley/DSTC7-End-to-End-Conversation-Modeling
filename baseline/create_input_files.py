import os, queue
from baseline import SOS_token, EOS_token, UNK_token



def main(path_txt, fld_out,
	max_vocab_size=2e4,
	):


	with open(path_txt, encoding="utf-8") as f:
		lines = f.readlines()

	if not os.path.exists(fld_out):
		os.makedirs(fld_out)

	path = dict()
	for end in ['source', 'target']:
		path[end] = os.path.join(fld_out, '%s_num.txt'%end)
	path['dict'] = os.path.join(fld_out, 'dict.txt')

	for k in path:
		open(path[k], 'w')

	sources = []
	targets = []
	n = 0
	for line in lines:
		n += 1
		if n%1e5 == 0:
			print('checked %.2fM/%.2fM lines'%(n/1e6, len(lines)/1e6))
		sub =  line.split('\t')
		source = sub[-2]
		target = sub[-1]
		if source == 'START':	# skip if source has nothing
			continue
		sources.append(source.strip().split())
		targets.append(target.strip().split())

	vocab = dict()
	for tokens in sources + targets:
		for token in tokens:
			if token not in vocab:
				vocab[token] = 0
			vocab[token] += 1

	pq = queue.PriorityQueue()
	for token in vocab:
		pq.put((-vocab[token], token))

	ordered_tokens = [SOS_token, EOS_token, UNK_token]
	while not pq.empty():
		freq, token = pq.get()
		ordered_tokens.append(token)
		if len(ordered_tokens) == max_vocab_size:
			break

	print('vocab size = %i'%len(ordered_tokens))

	token2index = dict()
	for i, token in enumerate(ordered_tokens):
		token2index[token] = str(i + 1)			# +1 as 0 is for padding
	with open(path['dict'], 'a', encoding="utf-8") as f:
		f.write('\n'.join(ordered_tokens))

	nums = []
	for tokens in sources:
		nums.append(' '.join([token2index.get(token, token2index[UNK_token]) for token in tokens]))
	with open(path['source'], 'a', encoding="utf-8") as f:
		f.write('\n'.join(nums))

	nums = []
	for tokens in targets:
		nums.append(' '.join([token2index.get(token, token2index[UNK_token]) for token in tokens]))
	with open(path['target'], 'a', encoding="utf-8") as f:
		f.write('\n'.join(nums))



if __name__ == '__main__':
	fld_out = 'official'
	path_txt = 'F:/DSTC/data-official/train.convos.txt'
	main(path_txt, fld_out)
	print('done!')






