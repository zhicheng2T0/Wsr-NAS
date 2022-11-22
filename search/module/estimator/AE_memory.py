import logging
import random
from collections import namedtuple, deque

import torch


batch_data = namedtuple('batch_data', ['image','source_noises', 'target_epsilon', 'delta_label', 'batch_y'])


class AE_Memory(object):
    """Memory"""

    def __init__(self, limit=128, batch_size=64):
        assert limit >= batch_size, 'limit (%d) should not less than batch size (%d)' % (limit, batch_size)
        super(AE_Memory, self).__init__()
        self.limit = limit
        self.batch_size = batch_size
        self.memory = deque(maxlen=limit)

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.limit, \
            'require batch_size (%d) exceeds memory limit, should be less than %d' % (batch_size, self.limit)
        length = len(self)

        indices = [i for i in range(length)]
        random.shuffle(indices)

        source_noises_list=[]
        image_list=[]
        target_epsilon_list=[]
        delta_label_list=[]
        batch_y_list=[]

        batch=[]

        #print('batch size: ',self.batch_size)
        for idx in indices:
            image_list.append(self.memory[idx].image)
            source_noises_list.append(self.memory[idx].source_noises)
            target_epsilon_list.append(self.memory[idx].target_epsilon)
            delta_label_list.append(self.memory[idx].delta_label)
            batch_y_list.append(self.memory[idx].batch_y)

            if len(batch_y_list) >= batch_size:
                batch.append((torch.stack(image_list),
                            torch.stack(source_noises_list),
                            torch.stack(target_epsilon_list),
                            torch.stack(delta_label_list),
                            torch.stack(batch_y_list)))
                source_noises_list=[]
                image_list=[]
                target_epsilon_list=[]
                delta_label_list=[]
                batch_y_list=[]
                
        return batch

    def append(self, image,source_noises,target_epsilon,delta_label,batch_y):
        self.memory.append(batch_data(image=image,source_noises=source_noises,target_epsilon=target_epsilon, delta_label=delta_label,batch_y=batch_y))


    def __len__(self):
        return len(self.memory)
