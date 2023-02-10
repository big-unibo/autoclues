num_features = 6

def params_GenericSPEC():
    return {
        'k': list(range(1, num_features))
    }

def params_NormalizedCut():
    return {
        'k': list(range(1, num_features))
    }

def params_WKMeans():
    return {
        'k': list(range(1, num_features))
    }