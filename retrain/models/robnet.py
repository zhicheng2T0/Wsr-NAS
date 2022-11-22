from .basic_model import Network, Network_INT


def robnet(genotype_list, **kwargs):
    return Network(genotype_list=genotype_list, **kwargs)

def robnet_INT(genotype_list, **kwargs):
    return Network_INT(genotype_list=genotype_list, **kwargs)
