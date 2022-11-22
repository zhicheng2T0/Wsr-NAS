from models.robnet import *


def model_entry(config, genotype_list):
    return globals()['robnet'](genotype_list, **config.model_param)

def model_entry_INT(config, genotype_list):
    print('in correct model_entry')
    return globals()['robnet_INT'](genotype_list, **config.model_param)
