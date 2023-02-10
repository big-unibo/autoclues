from policies.union import Union

def initiate(config):
    policies = {
        'union': Union
    }
    return policies['union'](config)