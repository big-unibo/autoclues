# from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
# from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    KBinsDiscretizer,
    Binarizer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from pipeline.feature_selectors import PearsonThreshold

from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from pipeline.PrototypeSingleton import PrototypeSingleton
from pipeline.outlier_detectors import (
    LocalOutlierDetector,
    IsolationOutlierDetector,
)  # , SGDOutlierDetector

from fsfc.generic import GenericSPEC, NormalizedCut, WKMeans


def get_baseline():
    baseline = {}
    for k in PrototypeSingleton.getInstance().getPrototype().keys():
        baseline[k] = ("{}_NoneType".format(k), {})
    return baseline


def pipeline_conf_to_full_pipeline(args, algorithm, seed, algo_config):
    if args == {}:
        args = get_baseline()
    op_to_class = {"pca": PCA, "selectkbest": SelectKBest}
    operators = []
    for part in PrototypeSingleton.getInstance().getParts():
        item = args[part]
        # print(item, PrototypeSingleton.getInstance().getFeatures())
        if "NoneType" in item[0]:
            continue
        else:
            params = {k.split("__", 1)[-1]: v for k, v in item[1].items()}
            transformation_param = item[0].split("_", 1)[0]
            operator_param = item[0].split("_", 1)[-1]
            if transformation_param == "normalize":
                (
                    numerical_features,
                    categorical_features,
                ) = PrototypeSingleton.getInstance().getFeatures()
                operator = ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("normalizing", globals()[operator_param](**params))
                                ]
                            ),
                            numerical_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[("identity_categorical", FunctionTransformer())]
                            ),
                            categorical_features,
                        ),
                    ]
                )
                PrototypeSingleton.getInstance().applyColumnTransformer()
            elif transformation_param == "features":
                operator = globals()[operator_param](**params)
                X = PrototypeSingleton.getInstance().getX()
                operator.fit(X)
                indeces = operator.get_support()
                PrototypeSingleton.getInstance().applyFeaturesEngineering(indeces)
            else:
                operator = globals()[operator_param](**params)
            operators.append((part, operator))

    PrototypeSingleton.getInstance().resetFeatures()
    if "random_state" in algorithm().get_params():
        clf = algorithm(random_state=seed, **algo_config)
    else:
        clf = algorithm(**algo_config)
    return Pipeline(operators + [("classifier", clf)]), operators
