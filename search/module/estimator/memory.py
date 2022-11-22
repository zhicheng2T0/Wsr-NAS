import logging
import random
from collections import namedtuple, deque

import torch


#Experience = namedtuple('Experience', ['weights', 'loss'])
Experience = namedtuple('Experience', ['weights', 'loss', 'ae_loss0', 'ae_loss1', 'ae_loss2', 'ae_loss3', 'ae_loss4', 'ae_loss5'])

'''
class Memory(object):
    """Memory"""

    def __init__(self, limit=128, batch_size=64, is_gae=False):
        assert limit >= batch_size, 'limit (%d) should not less than batch size (%d)' % (limit, batch_size)
        super(Memory, self).__init__()
        self.limit = limit
        self.batch_size = batch_size
        self.memory = deque(maxlen=limit)
        self.is_gae = is_gae

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.limit, \
            'require batch_size (%d) exceeds memory limit, should be less than %d' % (batch_size, self.limit)
        length = len(self)
        if batch_size > length:
            logging.warning('required batch_size (%d) is larger than memory size (%d)', batch_size, length)

        indices = [i for i in range(length)]
        random.shuffle(indices)

        weights_normal = []
        weights_reduce = []
        weights = []
        loss = []
        batch = []
        #print(len(self.memory))
        #print(len(self.memory[0]))
        for idx in indices:
            if self.is_gae:
                weights_normal.append(self.memory[idx].weights[0])
                weights_reduce.append(self.memory[idx].weights[1])
            else:
                #weights.append(self.memory[idx].weights)
                weights.append(torch.cat(self.memory[idx].weights,0))
            loss.append(self.memory[idx].loss)
            if len(loss) >= batch_size:
                if self.is_gae:
                    batch.append(((torch.stack(weights_normal), torch.stack(weights_reduce)), torch.stack(loss)))
                else:
                    # print(weights[0])
                    # print('weights0-----------')
                    # print(weights[1])
                    # print('weights1-----------')
                    # print(weights[0][0].shape,weights[0][1].shape)
                    # print('weights-----------')
                    # #print(loss)
                    # #print('loss--------------')
                    batch.append((torch.stack(weights), torch.stack(loss)))
                weights_normal = []
                weights_reduce = []
                weights = []
                loss = []
        if len(loss) > 0:
            if self.is_gae:
                batch.append(((torch.stack(weights_normal), torch.stack(weights_reduce)), torch.stack(loss)))
            else:
                batch.append((torch.stack(weights), torch.stack(loss)))
        return batch

    def append(self, weights, loss):
        self.memory.append(Experience(weights=weights, loss=loss))

    def state_dict(self):
        return {'limit': self.limit,
                'batch_size': self.batch_size,
                'memory': self.memory,
                'is_gae': self.is_gae}

    def load_state_dict(self, state_dict):
        self.limit = state_dict['limit']
        self.batch_size = state_dict['batch_size']
        self.memory = state_dict['memory']
        self.is_gae = state_dict['is_gae']

    def __len__(self):
        return len(self.memory)
'''

class Memory(object):
    """Memory"""

    def __init__(self, limit=128, batch_size=64, is_gae=False):
        assert limit >= batch_size, 'limit (%d) should not less than batch size (%d)' % (limit, batch_size)
        super(Memory, self).__init__()
        self.limit = limit
        self.batch_size = batch_size
        self.memory = deque(maxlen=limit)
        self.is_gae = is_gae

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.limit, \
            'require batch_size (%d) exceeds memory limit, should be less than %d' % (batch_size, self.limit)
        length = len(self)
        if batch_size > length:
            logging.warning('required batch_size (%d) is larger than memory size (%d)', batch_size, length)

        indices = [i for i in range(length)]
        random.shuffle(indices)

        weights_normal = []
        weights_reduce = []
        weights = []
        loss = []
        ae_loss0 = []
        ae_loss1 = []
        ae_loss2 = []
        ae_loss3 = []
        ae_loss4 = []
        ae_loss5 = []
        batch = []
        #print(len(self.memory))
        #print(len(self.memory[0]))
        for idx in indices:
            if self.is_gae:
                weights_normal.append(self.memory[idx].weights[0])
                weights_reduce.append(self.memory[idx].weights[1])
                loss.append(torch.cat(self.memory[idx].loss,0))

                ae_loss0.append(self.memory[idx].ae_loss0)
                ae_loss1.append(self.memory[idx].ae_loss1)
                ae_loss2.append(self.memory[idx].ae_loss2)
                ae_loss3.append(self.memory[idx].ae_loss3)
                ae_loss4.append(self.memory[idx].ae_loss4)
                ae_loss5.append(self.memory[idx].ae_loss5)
            else:
                #weights.append(self.memory[idx].weights)
                weights.append(torch.cat(self.memory[idx].weights,0).cuda())
                loss.append(self.memory[idx].loss.cuda())

                ae_loss0.append(self.memory[idx].ae_loss0.cuda())
                ae_loss1.append(self.memory[idx].ae_loss1.cuda())
                ae_loss2.append(self.memory[idx].ae_loss2.cuda())
                ae_loss3.append(self.memory[idx].ae_loss3.cuda())
                ae_loss4.append(self.memory[idx].ae_loss4.cuda())
                ae_loss5.append(self.memory[idx].ae_loss5.cuda())
            #loss.append(self.memory[idx].loss)
            if len(loss) >= batch_size:
                if self.is_gae:
                    batch.append(((torch.stack(weights_normal), torch.stack(weights_reduce)), torch.stack(loss)))
                else:
                    batch.append((torch.stack(weights), torch.stack(loss), torch.stack(ae_loss0), torch.stack(ae_loss1), torch.stack(ae_loss2), torch.stack(ae_loss3), torch.stack(ae_loss4), torch.stack(ae_loss5)))
                weights_normal = []
                weights_reduce = []
                weights = []
                loss = []
                ae_loss0 = []
                ae_loss1 = []
                ae_loss2 = []
                ae_loss3 = []
                ae_loss4 = []
                ae_loss5 = []
        if len(loss) > 0:
            if self.is_gae:
                batch.append(((torch.stack(weights_normal), torch.stack(weights_reduce)), torch.stack(loss)))
            else:
                batch.append((torch.stack(weights), torch.stack(loss), torch.stack(ae_loss0), torch.stack(ae_loss1), torch.stack(ae_loss2), torch.stack(ae_loss3), torch.stack(ae_loss4), torch.stack(ae_loss5)))
        return batch

    def append(self, weights, loss, ae_loss0, ae_loss1, ae_loss2, ae_loss3, ae_loss4, ae_loss5):
        self.memory.append(Experience(weights=weights, loss=loss, ae_loss0=ae_loss0, ae_loss1=ae_loss1, ae_loss2=ae_loss2, ae_loss3=ae_loss3, ae_loss4=ae_loss4, ae_loss5=ae_loss5))

    def state_dict(self):
        return {'limit': self.limit,
                'batch_size': self.batch_size,
                'memory': self.memory,
                'is_gae': self.is_gae}

    def load_state_dict(self, state_dict):
        self.limit = state_dict['limit']
        self.batch_size = state_dict['batch_size']
        self.memory = state_dict['memory']
        self.is_gae = state_dict['is_gae']

    def __len__(self):
        return len(self.memory)