num_features = 6

def params_GenericSPEC():
    return {
        'k': [2] if num_features in [2, 3] else list(range(2, num_features))
    }

def params_NormalizedCut():
    return {
        'k': [2] if num_features in [2, 3] else list(range(2, num_features))
    }

def params_WKMeans():
    return {
        'k': [2] if num_features in [2, 3] else list(range(2, num_features))
    }

def params_PearsonThreshold():
    return {
        'threshold': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    }