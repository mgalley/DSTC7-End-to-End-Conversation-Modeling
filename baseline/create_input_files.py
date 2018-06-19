import os, queue

SOS_token = '_SOS_'
EOS_token = '_EOS_'

def main(path_txt, fld_out,
	delimiter='\t'
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
	for line in lines:
		sub =  line.split(delimiter)
		source = sub[-2]
		target = sub[-1]
		sources.append(source.strip().split(' '))
		targets.append(target.strip().split(' '))

	vocab = dict()
	for tokens in sources + targets:
		for token in tokens:
			if token not in vocab:
				vocab[token] = 0
			vocab[token] += 1

	pq = queue.PriorityQueue()
	for token in vocab:
		pq.put((-vocab[token], token))

	ordered_tokens = [SOS_token, EOS_token]
	while not pq.empty():
		freq, token = pq.get()
		ordered_tokens.append(token)

	token2index = dict()
	for i, token in enumerate(ordered_tokens):
		token2index[token] = str(i + 1)
	with open(path['dict'], 'a', encoding="utf-8") as f:
		f.write('\n'.join(ordered_tokens))

	nums = []
	for tokens in sources:
		nums.append(' '.join([token2index[token] for token in tokens]))
	with open(path['source'], 'a', encoding="utf-8") as f:
		f.write('\n'.join(nums))

	nums = []
	for tokens in targets:
		nums.append(' '.join([token2index[token] for token in tokens]))
	with open(path['target'], 'a', encoding="utf-8") as f:
		f.write('\n'.join(nums))




if __name__ == '__main__':
	fld_out = 'trial'
	path_txt = os.path.join('trial','trial.convos.txt')
	main(path_txt, fld_out)
	print('done!')






