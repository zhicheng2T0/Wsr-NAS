import logging
import random
from collections import namedtuple, deque

import torch
import time

batch_data = namedtuple('batch_data', ['image','source_noises', 'target_epsilon', 'delta_label', 'batch_y','index_list'])

'''
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
        #index_l_list=[]

        batch=[]
        output2=[]

        #print('----indices--------',indices)
        for idx in indices:
            image_list.append(self.memory[idx].image)
            source_noises_list.append(self.memory[idx].source_noises)
            target_epsilon_list.append(self.memory[idx].target_epsilon)
            delta_label_list.append(self.memory[idx].delta_label)
            batch_y_list.append(self.memory[idx].batch_y)
            index_l_list=self.memory[idx].index_list
            #print('------index_l_list--------',index_l_list)

            if len(batch_y_list) >= batch_size:
                batch.append((torch.stack(image_list),
                            torch.stack(source_noises_list),
                            torch.stack(target_epsilon_list),
                            torch.stack(delta_label_list),
                            torch.stack(batch_y_list)))
                output2=index_l_list
                source_noises_list=[]
                image_list=[]
                target_epsilon_list=[]
                delta_label_list=[]
                batch_y_list=[]
                index_l_list=[]
        return batch,output2

    def append(self, image,source_noises,target_epsilon,delta_label,batch_y,index_list):
        start=time.time()
        for i in range(image.shape[0]):
            self.memory.append(batch_data(image=image[i],source_noises=source_noises[i],target_epsilon=target_epsilon[i], delta_label=delta_label[i],batch_y=batch_y[i],index_list=index_list))
        end=time.time()
        #print('end-start',end-start)

    def __len__(self):
        return len(self.memory)
'''

class AE_Memory(object):
    """Memory"""

    def __init__(self, limit=128, batch_size=64):
        #assert limit >= batch_size, 'limit (%d) should not less than batch size (%d)' % (limit, batch_size)
        super(AE_Memory, self).__init__()
        self.limit = limit
        self.batch_size = batch_size
        self.memory = deque(maxlen=limit)

    def get_batch(self):
        length = len(self)

        indices = [i for i in range(length)]
        random.shuffle(indices)


        batch=[]
        output2=[]

        #print('----indices--------',indices)
        for idx in indices:
            #print('---',idx)
            batch.append((self.memory[idx].image,
                        self.memory[idx].source_noises,
                        self.memory[idx].target_epsilon,
                        self.memory[idx].delta_label,
                        self.memory[idx].batch_y))
            output2.append(self.memory[idx].index_list)

        return batch,output2

    def append(self, image,source_noises,target_epsilon,delta_label,batch_y,index_list):
        #start=time.time()
        self.memory.append(batch_data(image=image,source_noises=source_noises,target_epsilon=target_epsilon, delta_label=delta_label,batch_y=batch_y,index_list=index_list))
        #end=time.time()
        #print('end-start',end-start)

    def __len__(self):
        return len(self.memory)

