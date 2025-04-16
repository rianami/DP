import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import datetime
import random


class TransHModel(nn.Module):
	def __init__(self, config):
		super(TransHModel, self).__init__()
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.batch_size = config.batch_size

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		norm_weight = floatTensor(self.relation_total, self.embedding_size)
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
		nn.init.xavier_uniform(norm_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.norm_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)
		self.norm_embeddings.weight = nn.Parameter(norm_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb
		self.norm_embeddings.weight.data = normalize_norm_emb

	def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)
		pos_norm = self.norm_embeddings(pos_r)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)
		neg_norm = self.norm_embeddings(neg_r)

		pos_h_e = projection_transH_pytorch(pos_h_e, pos_norm)
		pos_t_e = projection_transH_pytorch(pos_t_e, pos_norm)
		neg_h_e = projection_transH_pytorch(neg_h_e, neg_norm)
		neg_t_e = projection_transH_pytorch(neg_t_e, neg_norm)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
		else:
			pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
		return pos, neg

class MyProcessTransH(multiprocessing.Process):
	def __init__(self, L, tripleDict, ent_embeddings, 
		rel_embeddings, norm_embeddings, L1_flag, filter, queue=None, head=0):
		super(MyProcessTransH, self).__init__()
		self.L = L
		self.queue = queue
		self.tripleDict = tripleDict
		self.ent_embeddings = ent_embeddings
		self.rel_embeddings = rel_embeddings
		self.norm_embeddings = norm_embeddings
		self.L1_flag = L1_flag
		self.filter = filter
		self.head = head

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.norm_embeddings, self.L1_flag, self.filter, self.L, self.head)
			except:
				time.sleep(5)
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.norm_embeddings, self.L1_flag, self.filter, self.L, self.head)
			self.queue.task_done()

	def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
		norm_embeddings, L1_flag, filter, L, head):

		hit10Count, totalRank, tripleCount = evaluation_transH_helper(testList, tripleDict, ent_embeddings, 
			rel_embeddings, norm_embeddings, L1_flag, filter, head)

		L.append((hit10Count, totalRank, tripleCount))


class marginLoss(nn.Module):
	def __init__(self):
		super(marginLoss, self).__init__()

	def forward(self, pos, neg, margin):
		zero_tensor = floatTensor(pos.size())
		zero_tensor.zero_()
		zero_tensor = autograd.Variable(zero_tensor)
		return torch.sum(torch.max(pos - neg + margin, zero_tensor))
"""
Description of model training parameters:

learning_rate: The initial learning rate applied during optimization.

early_stopping_round: Number of times the learning rate can be reduced. If set to 0, the learning rate remains constant.

num_batches: Number of batches to process in each training epoch.

train_times: Maximum number of epochs to train the model.

L1_flag: Indicates whether to use L1 distance (True) or L2 distance (False) as the dissimilarity measure.

embedding_size: Dimensionality of the embeddings for both entities and relations.

margin: Margin value used in the MarginLoss function.

filter: Determines whether to exclude false negative samples during training.

momentum: Momentum factor used by the optimizer to accelerate convergence.

optimizer: Optimization algorithm to use (e.g., SGD, Adam).

loss_function: Loss function used during training (typically margin-based loss).

entity_total: Total number of unique entities in the dataset.

relation_total: Total number of unique relations in the dataset.

batch_size: Number of training instances included in each batch.
"""

class Model_config(object):
	def __init__(self):
		self.learning_rate = 0.001
		self.early_stopping_round = 0
		self.num_batches = 250
		self.train_times = 1200
		self.L1_flag = True
		self.embedding_size = 200
		self.margin = 1.0
		self.filter = True
		self.momentum = 0.9
		self.optimizer = optim.Adam
		self.loss_function = marginLoss()
		self.entity_total = 0
		self.relation_total = 0
		self.batch_size = 0

	def loadTriple(inPath, fileName):
		with open(os.path.join(inPath, fileName), 'r') as fr:
			i = 0
			tripleList = []
			for line in fr:
				if i == 0:
					tripleTotal = int(line)
					i += 1
				else:
					line_split = line.split()
					head = int(line_split[0])
					tail = int(line_split[1])
					rel = int(line_split[2])
					tripleList.append(Triple(head, tail, rel))

		tripleDict = {}
		for triple in tripleList:
			tripleDict[(triple.h, triple.t, triple.r)] = True

		return tripleTotal, tripleList, tripleDict
	# Gets the number of entities/relations/triples
	def getAnythingTotal(inPath, fileName):
		with open(os.path.join(inPath, fileName), 'r') as fr:
			for line in fr:
				return int(line)
	
	# Split the tripleList into #num_batches batches
	def getBatchList(tripleList, num_batches):
		batchSize = len(tripleList) // num_batches
		batchList = [0] * num_batches
		for i in range(num_batches - 1):
			batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
		batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
		return batchList

	# Use all the tripleList,
	# and generate negative samples by corrupting head or tail with equal probabilities,
	# with checking whether false negative samples exist.
	def getBatch_filter_all(tripleList, entityTotal, tripleDict):
		newTripleList = [corrupt_head_filter(triple, entityTotal, tripleDict) if random.random() < 0.5 
			else corrupt_tail_filter(triple, entityTotal, tripleDict) for triple in tripleList]
		ph, pt ,pr = getThreeElements(tripleList)
		nh, nt, nr = getThreeElements(newTripleList)
		return ph, pt, pr, nh, nt, nr

	# Sample a batch of #batchSize triples from tripleList,
	# and generate negative samples by corrupting head or tail with equal probabilities,
	# with checking whether false negative samples exist.
	def getBatch_filter_random(tripleList, batchSize, entityTotal, tripleDict):
		oldTripleList = random.sample(tripleList, batchSize)
		newTripleList = [corrupt_head_filter(triple, entityTotal, tripleDict) if random.random() < 0.5 
			else corrupt_tail_filter(triple, entityTotal, tripleDict) for triple in oldTripleList]
		ph, pt ,pr = getThreeElements(oldTripleList)
		nh, nt, nr = getThreeElements(newTripleList)
		return ph, pt, pr, nh, nt, nr

	def normLoss(embeddings, dim=1):
		norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
		return torch.sum(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))
	

	def evaluation_transH_helper(testList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=0):
		# embeddings are torch tensor like (No Variable!)
		# Only one kind of relation

		headList = [triple.h for triple in testList]
		tailList = [triple.t for triple in testList]
		relList = [triple.r for triple in testList]

		h_e = ent_embeddings[headList]
		t_e = ent_embeddings[tailList]
		r_e = rel_embeddings[relList]
		this_rel = relList[0]
		this_norm_emb = norm_embeddings[this_rel]
		this_proj_all_e = projection_transH_pytorch(ent_embeddings, this_norm_emb)
		this_proj_all_e = this_proj_all_e.cpu().numpy()

		if head == 1:
			proj_t_e = projection_transH_pytorch(t_e, this_norm_emb)
			c_h_e = proj_t_e - r_e
			c_h_e = c_h_e.cpu().numpy()

			if L1_flag == True:
				dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
			else:
				dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

			rankArrayHead = np.argsort(dist, axis=1)
			if filter == False:
				rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
			else:
				rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
								for elem in zip(headList, tailList, relList, rankArrayHead)]

			isHit10ListHead = [x for x in rankListHead if x < 10]

			totalRank = sum(rankListHead)
			hit10Count = len(isHit10ListHead)
			tripleCount = len(rankListHead)

		elif head == 2:
			proj_h_e = projection_transH_pytorch(h_e, this_norm_emb)
			c_t_e = proj_h_e + r_e
			c_t_e = c_t_e.cpu().numpy()

			if L1_flag == True:
				dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
			else:
				dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

			rankArrayTail = np.argsort(dist, axis=1)
			if filter == False:
				rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
			else:
				rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
								for elem in zip(headList, tailList, relList, rankArrayTail)]

			isHit10ListTail = [x for x in rankListTail if x < 10]

			totalRank = sum(rankListTail)
			hit10Count = len(isHit10ListTail)
			tripleCount = len(rankListTail)

		else:
			proj_h_e = projection_transH_pytorch(h_e, this_norm_emb)
			c_t_e = proj_h_e + r_e
			proj_t_e = projection_transH_pytorch(t_e, this_norm_emb)
			c_h_e = proj_t_e - r_e

			c_t_e = c_t_e.cpu().numpy()
			c_h_e = c_h_e.cpu().numpy()

			if L1_flag == True:
				dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
			else:
				dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

			rankArrayTail = np.argsort(dist, axis=1)
			if filter == False:
				rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
			else:
				rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
								for elem in zip(headList, tailList, relList, rankArrayTail)]

			isHit10ListTail = [x for x in rankListTail if x < 10]

			if L1_flag == True:
				dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
			else:
				dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

			rankArrayHead = np.argsort(dist, axis=1)
			if filter == False:
				rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
			else:
				rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
								for elem in zip(headList, tailList, relList, rankArrayHead)]

			isHit10ListHead = [x for x in rankListHead if x < 10]

			totalRank = sum(rankListTail) + sum(rankListHead)
			hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
			tripleCount = len(rankListTail) + len(rankListHead)

		return hit10Count, totalRank, tripleCount
		
			
		
		
	def evaluation_transH(testList, tripleDict, ent_embeddings, rel_embeddings, 
		norm_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
		# embeddings are torch tensor like (No Variable!)

		if k > len(testList):
			testList = random.choices(testList, k=k)
		elif k > 0:
			testList = random.sample(testList, k=k)

		# Split the testList according to the relation
		testList.sort(key=lambda x: (x.r, x.h, x.t))
		grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

		ent_embeddings = ent_embeddings.cpu()
		rel_embeddings = rel_embeddings.cpu()
		norm_embeddings = norm_embeddings.cpu()

		with multiprocessing.Manager() as manager:
			L = manager.list()
			queue = multiprocessing.JoinableQueue()
			workerList = []
			for i in range(num_processes):
				worker = MyProcessTransH(L, tripleDict, ent_embeddings, rel_embeddings,
					norm_embeddings, L1_flag, filter, queue=queue, head=head)
				workerList.append(worker)
				worker.daemon = True
				worker.start()

			for k, subList in grouped:
				queue.put(subList)

			queue.join()

			resultList = list(L)

			for worker in workerList:
				worker.terminate()

		if head == 1 or head == 2:
			hit10 = sum([elem[0] for elem in resultList]) / len(testList)
			meanrank = sum([elem[1] for elem in resultList]) / len(testList)
		else:
			hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
			meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

		print('Meanrank: %.6f' % meanrank)
		print('Hit@10: %.6f' % hit10)

		return hit10, meanrank
if __name__ == "__main__":


	trainTotal, trainList, trainDict = loadTriple('./data/FB15k', 'train2id.txt')
	validTotal, validList, validDict = loadTriple('./data/FB15k', 'valid2id.txt')
	tripleTotal, tripleList, tripleDict = loadTriple('./data/FB15k', 'triple2id.txt')
	model_options = Model_config()
	
	#model_options.optimizer = optim.SGD
	model_options.optimizer = optim.Adam
	#model_options.optimizer = optim.RMSprop

	model_options.entity_total = model_options.getAnythingTotal('./data/FB15k', 'entity2id.txt')
	model_options.relation_total = model_options.getAnythingTotal('./data/FB15k', 'relation2id.txt')
	model_options.batch_size = trainTotal // model_options.num_batches

	loss_function = model_options.loss_function()
	model = TransHModel(model_options)

	
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

	optimizer = model_options.optimizer(model.parameters(), lr=model_options.learning_rate)
	margin = autograd.Variable(floatTensor([model_options.margin]))

	start_time = time.time()

	filename = '_'.join(
		['l', str(model_options.learning_rate),
		 'es', str(model_options.early_stopping_round),
		 'L', str(model_options.L1_flag),
		 'em', str(model_options.embedding_size),
		 'nb', str(model_options.num_batches),
		 'n', str(model_options.train_times),
		 'm', str(model_options.margin),
		 'f', str(model_options.filter),
		 'mo', str(model_options.momentum),
		 's', str(model_options.seed),
		 'op', str(model_options.optimizer),
		 'lo', str(model_options.loss_type),]) + '_TransH.ckpt'

	trainBatchList = model_options.getBatchList(trainList, model_options.num_batches)

	for epoch in range(model_options.train_times):
		total_loss = floatTensor([0.0])
		random.shuffle(trainBatchList)
		for batchList in trainBatchList:
			pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = model_options.getBatch_filter_all(batchList,model_options.entity_total, tripleDict)
			

			batch_entity_set = set(pos_h_batch + pos_t_batch + neg_h_batch + neg_t_batch)
			batch_relation_set = set(pos_r_batch + neg_r_batch)
			batch_entity_list = list(batch_entity_set)
			batch_relation_list = list(batch_relation_set)

			pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
			pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
			pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
			neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
			neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
			neg_r_batch = autograd.Variable(longTensor(neg_r_batch))

			model.zero_grad()
			pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch)

			
			losses = loss_function(pos, neg, margin)
			
			ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
			rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
			norm_embeddings = model.norm_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
			losses = losses + model_options.normLoss(ent_embeddings) + model_options.normLoss(rel_embeddings)
			
			losses.backward()
			optimizer.step()
			total_loss += losses.data

			normalize_norm_emb = F.normalize(model.norm_embeddings.weight.data[longTensor(batch_relation_list)], p=2, dim=1)
			model.norm_embeddings.weight.data[longTensor(batch_relation_list)] = normalize_norm_emb

		

		if epoch % 10 == 0:
			now_time = time.time()
			print(now_time - start_time)
			print("Train total loss: %d %f" % (epoch, total_loss[0]))

		if epoch % 10 == 0:
			pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = model_options.getBatch_filter_random(validList,model_options.batch_size, model_options.entity_total, tripleDict)
		
			pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
			pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
			pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
			neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
			neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
			neg_r_batch = autograd.Variable(longTensor(neg_r_batch))

			pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch)

			
			losses = loss_function(pos, neg, margin)
			
			ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
			rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
			norm_embeddings = model.norm_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
			losses = losses + model_options.normLoss(ent_embeddings) + model_options.normLoss(rel_embeddings)

			print("Valid batch loss: %d %f" % (epoch, losses.data[0]))


		if model_options.early_stopping_round > 0:
			if epoch == 0:
				ent_embeddings = model.ent_embeddings.weight.data
				rel_embeddings = model.rel_embeddings.weight.data
				norm_embeddings = model.norm_embeddings.weight.data
				L1_flag = model.L1_flag
				filter = model.filter	
				hit10, best_meanrank = model_options.evaluation_transH(validList, tripleDict, ent_embeddings, rel_embeddings, 
					norm_embeddings, L1_flag, filter, k=model_options.batch_size)
				torch.save(model, os.path.join('./model/FB15k' , filename))	
				best_epoch = 0
				meanrank_not_decrease_time = 0
				lr_decrease_time = 0

			# Evaluate on validation set for every 5 epochs
			elif epoch % 5 == 0:
				ent_embeddings = model.ent_embeddings.weight.data
				rel_embeddings = model.rel_embeddings.weight.data
				norm_embeddings = model.norm_embeddings.weight.data
				L1_flag = model.L1_flag
				filter = model.filter
				hit10, now_meanrank = model_options.evaluation_transH(validList, tripleDict, ent_embeddings, rel_embeddings,norm_embeddings, L1_flag, filter, k=model_options.batch_size)
				if now_meanrank < best_meanrank:
					meanrank_not_decrease_time = 0
					best_meanrank = now_meanrank
					torch.save(model, os.path.join('./model/FB15k', filename))
				else:
					meanrank_not_decrease_time += 1
					# If the result hasn't improved for consecutive 5 evaluations, decrease learning rate
					if meanrank_not_decrease_time == 5:
						lr_decrease_time += 1
						if lr_decrease_time == model_options.early_stopping_round:
							break
						else:
							optimizer.param_groups[0]['lr'] *= 0.5
							meanrank_not_decrease_time = 0
				#if USE_CUDA:
					#model.cuda()

		elif (epoch + 1) % 10 == 0 or epoch == 0:
			torch.save(model, os.path.join('./model/FB15k', filename))


	testTotal, testList, testDict = loadTriple('./data/FB15k', 'test2id.txt')
	oneToOneTotal, oneToOneList, oneToOneDict = loadTriple('./data/FB15k', 'one_to_one.txt')
	oneToManyTotal, oneToManyList, oneToManyDict = loadTriple('./data/FB15k', 'one_to_many.txt')
	manyToOneTotal, manyToOneList, manyToOneDict = loadTriple('./data/FB15k', 'many_to_one.txt')
	manyToManyTotal, manyToManyList, manyToManyDict = loadTriple('./data/FB15k', 'many_to_many.txt')
	
	ent_embeddings = model.ent_embeddings.weight.data
	rel_embeddings = model.rel_embeddings.weight.data
	norm_embeddings = model.norm_embeddings.weight.data
	L1_flag = model.L1_flag
	filter = model.filter

	hit10Test, meanrankTest = model_options.evaluation_transH(testList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=0)

	hit10OneToOneHead, meanrankOneToOneHead = model_options.evaluation_transH(oneToOneList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=1)
	hit10OneToManyHead, meanrankOneToManyHead = model_options.evaluation_transH(oneToManyList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=1)
	hit10ManyToOneHead, meanrankManyToOneHead = model_options.evaluation_transH(manyToOneList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=1)
	hit10ManyToManyHead, meanrankManyToManyHead = model_options.evaluation_transH(manyToManyList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=1)

	hit10OneToOneTail, meanrankOneToOneTail = model_options.evaluation_transH(oneToOneList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=2)
	hit10OneToManyTail, meanrankOneToManyTail = model_options.evaluation_transH(oneToManyList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=2)
	hit10ManyToOneTail, meanrankManyToOneTail = model_options.evaluation_transH(manyToOneList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=2)
	hit10ManyToManyTail, meanrankManyToManyTail = model_options.evaluation_transH(manyToManyList, tripleDict, ent_embeddings, rel_embeddings, norm_embeddings, L1_flag, filter, head=2)	

	writeList = [filename, 
		'testSet', '%.6f' % hit10Test, '%.6f' % meanrankTest, 
		'one_to_one_head', '%.6f' % hit10OneToOneHead, '%.6f' % meanrankOneToOneHead, 
		'one_to_many_head', '%.6f' % hit10OneToManyHead, '%.6f' % meanrankOneToManyHead, 
		'many_to_one_head', '%.6f' % hit10ManyToOneHead, '%.6f' % meanrankManyToOneHead, 
		'many_to_many_head', '%.6f' % hit10ManyToManyHead, '%.6f' % meanrankManyToManyHead,
		'one_to_one_tail', '%.6f' % hit10OneToOneTail, '%.6f' % meanrankOneToOneTail, 
		'one_to_many_tail', '%.6f' % hit10OneToManyTail, '%.6f' % meanrankOneToManyTail, 
		'many_to_one_tail', '%.6f' % hit10ManyToOneTail, '%.6f' % meanrankManyToOneTail, 
		'many_to_many_tail', '%.6f' % hit10ManyToManyTail, '%.6f' % meanrankManyToManyTail,]

	# Write the result into file
	with open(os.path.join('./result/', 'FB15k_TRANSH.txt'), 'a') as fw:
		fw.write('\t'.join(writeList) + '\n')
