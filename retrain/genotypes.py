from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

RACL = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), 
                        ('sep_conv_5x5', 0), ('skip_connect', 1), 
                        ('sep_conv_3x3', 0), ('skip_connect', 3), 
                        ('sep_conv_3x3', 3), ('skip_connect', 4)],
               normal_concat=[2, 3, 4, 5],
                reduce=[('sep_conv_3x3',0), ('sep_conv_5x5', 1),
                       ('avg_pool_3x3', 0), ('dil_conv_3x3', 1),
                        ('sep_conv_3x3', 0), ('sep_conv_5x5',1),
                         ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
               reduce_concat=[2, 3, 4, 5])



cna_r=Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 4), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))
cna_m = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))
cna_n=Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 2), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
cna_m_1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))


ADVRUSH = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), 
                            ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), 
                            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), 
                            ('skip_connect', 0), ('sep_conv_3x3', 1)], 
                    normal_concat=range(2, 6),
                    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), 
                            ('skip_connect', 0), ('dil_conv_3x3', 2),
                             ('skip_connect', 0), ('avg_pool_3x3', 1), 
                             ('skip_connect', 0), ('skip_connect', 2)], 
                    reduce_concat=range(2, 6))


WsrNet_Plus = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))


