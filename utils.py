import pandas as pd

def prepare_input(data, model):
    """
    Convert JSON input into DataFrame matching model features.
    """
    input_df = pd.DataFrame([data])

    # Add missing columns required by the heuristic/model
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in correct order
    return input_df[model.feature_names_in_]
