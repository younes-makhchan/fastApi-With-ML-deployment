import pickle
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve(strict=True).parent



classes = [
    "L",
    "M",
    "H",
]

with open(f"{BASE_DIR}/ModelStudentPerformance-0.1.0.pkl", "rb") as f, open(f"{BASE_DIR}/label_encoders-0.1.0.pkl", "rb") as transformer_file:
    model = pickle.load(f)
    label_encoders = pickle.load(transformer_file)


def transform_features(features):
    transformed = {}
    categorical_features = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

    for feature_name, feature_value in features.dict().items():
        if isinstance(feature_value, (int, float)):
            transformed[feature_name] = feature_value
        elif feature_name in categorical_features:
            
            transformed[feature_name] = label_encoders[feature_name].transform([feature_value])[0]
        else:
            transformed[feature_name] = feature_value

    return transformed

def predict_pipeline(features):
    # transform the features first
    transformed=transform_features(features)

    feature_names = list(transformed.keys())
    feature_vector = [transformed[feature] for feature in feature_names]
  
    pred = model.predict([feature_vector])
    return pred[0]


