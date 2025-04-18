import json
from typing import List, Optional, Tuple, Set, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_meta_data(raw_file_path: str) -> pd.DataFrame:
    """
    Load and process provided meta information for the datasets.

    :param raw_file_path: Path to the raw metadata file.
    :return: Processed metadata
    """
    meta = pd.read_excel(raw_file_path, header=1)
    meta.columns = meta.columns.str.lower()
    meta.drop(columns="unnamed: 0", inplace=True)

    meta[["attribute", "description"]] = meta[["attribute", "description"]].ffill()
    meta["attribute"] = meta["attribute"].str.lower()

    meta["meaning"] = meta["meaning"].ffill()

    meta["value"] = [v.split(", ") if type(v) == str and "," in v else v for v in meta["value"]]
    meta = meta.explode("value")

    print("Metadata loaded.")

    return meta


def rectify_meta_attributes(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Align the attribute names from the metadata to the other datasets.

    :param meta: Metadata with attributes to be rectified.
    :return: Rectified metadata
    """
    meta["attribute"] = meta["attribute"].replace(r"_rz$", "", regex=True)

    attribute_renaming = {
        "d19_buch": "d19_buch_cd",
        "d19_kk_kundentyp": "kk_kundentyp",
        "kba13_ccm_1400_2500": "kba13_ccm_1401_2500",
        "soho_flag": "soho_kz"
    }

    meta["attribute"] = meta["attribute"].replace(attribute_renaming)

    print("Metadata rectified.")

    return meta


def convert_unknown_values_to_null(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Replace unknown values with null values either based on the metadata or other column-specific rules.

    :param df: Dataset with unknown values that are not null.
    :param meta: Metadata with information about attribute specific unknown values.
    :return: Dataset with unknown values replaced by null values
    """
    cameo_columns = ["cameo_intl_2015", "cameo_deug_2015", "cameo_deu_2015"]
    df[cameo_columns] = df[cameo_columns].replace(["X", "XX"], np.nan)
    df[cameo_columns[:-1]] = df[cameo_columns[:-1]].astype(float)

    unknown_values = meta[meta["meaning"].str.contains("unknown")]

    for attribute, unknown_value in zip(unknown_values["attribute"], unknown_values["value"]):
        if attribute in df:
            df[attribute] = df[attribute].replace(float(unknown_value), np.nan)

    print(f"Converted {len(unknown_values) + 3} unknown values to null.")

    return df


def identify_invalid_out_of_range_values(meta: pd.DataFrame, df: pd.DataFrame, log_details: bool = False) -> pd.DataFrame:
    """
    Identify out-of-range values in the dataset based on metadata.

    Compares actual values in the dataset against expected values defined in the metadata.

    :param meta: Metadata with valid attribute values.
    :param df: Dataset to check for invalid values.
    :param log_details: Whether to print details of each out-of-range value found.
    :return: Attributes and their out-of-range values.
    """
    categorical_values = meta[meta["value"] != "…"][["attribute", "value"]]
    categorical_values = categorical_values.rename(columns={"value": "meta_value"})

    total_invalid_oor_values = []

    for attribute in categorical_values["attribute"].unique():
        if attribute in df:
            attribute_value_counts = df[attribute].value_counts().reset_index()
            attribute_value_counts = attribute_value_counts.merge(
                categorical_values[categorical_values["attribute"] == attribute]["meta_value"],
                left_on=attribute,
                right_on="meta_value",
                how="left",
            )

            invalid_oor_values = attribute_value_counts[attribute_value_counts["meta_value"].isnull()]
            if len(invalid_oor_values) > 0:
                for out_of_range_value in invalid_oor_values.itertuples():
                    if log_details:
                        print(f"attribute: {attribute}, invalid value: {out_of_range_value[1]}, "
                              f"affected rows: {out_of_range_value.count}")
                    total_invalid_oor_values.append([attribute, out_of_range_value[1]])

    return pd.DataFrame(data=total_invalid_oor_values, columns=["attribute", "value"])


def convert_invalid_values_to_null(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Replace invalid values in the dataset with null values.

    Handles attribute specific cases and uses metadata to identify values that are out of range.

    :param df: dataset with invalid values.
    :param meta: Metadata used to detect out-of-range values.
    :return: Cleaned dataset with invalid values replaced by null values.
    """
    df.loc[df["geburtsjahr"] == 0, "geburtsjahr"] = np.nan
    df.loc[df["anz_personen"] == 0, "anz_personen"] = np.nan

    invalid_oor_values = identify_invalid_out_of_range_values(meta, df, log_details=False)
    for attribute, invalid_oor_value in zip(invalid_oor_values["attribute"], invalid_oor_values["value"]):
        if attribute in df:
            df[attribute] = df[attribute].replace(invalid_oor_value, np.nan)

    print(f"Converted {len(invalid_oor_values) + 2} invalid values to null.")

    return df


def remove_records_with_insufficient_data(df: pd.DataFrame, max_ratio: float = .33) -> pd.DataFrame:
    """
    Remove records with too many missing feature values.

    Rows with missing value ratio greater than or equal to the specified threshold are dropped.

    :param df: Dataset with records that have too many missing values.
    :param max_ratio: Maximum allowed ratio of missing values per record.
    :return: Dataset without rows that have too many missing feature values.
    """
    insufficient_records = df.isnull().sum(axis=1) / len(df.columns) >= max_ratio
    df = df[~insufficient_records]

    print(f"Removed {sum(insufficient_records)} records with over {max_ratio * 100}% missing feature values.")

    return df


def remove_features_with_high_missing_values_ratio(df: pd.DataFrame, max_ratio: float = .2) -> pd.DataFrame:
    """
    Remove features (columns) with a high ratio of missing values.

    Columns with a missing value ratio greater than or equal to the specified threshold are dropped.

    :param df: Dataset with features that have too many missing values.
    :param max_ratio: Maximum allowed ratio of missing values per feature.
    :return: Dataset without features over the missing ratio threshold.
    """
    missing_ratios = pd.Series(df.isnull().sum() / len(df))
    columns_with_high_ratio = list(missing_ratios[missing_ratios >= max_ratio].index)

    df = df.drop(columns=columns_with_high_ratio)
    print(f"Removed {len(columns_with_high_ratio)} features with over {max_ratio * 100}% missing values:"
          f"\n{", ".join(columns_with_high_ratio)}")

    return df


def impute_features_with_median_or_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using the median for numeric features and mode for categorical features.

    :param df: Dataset with missing values.
    :return: Dataset with missing values imputed.
    """
    columns_with_missing_values = df.columns[df.isnull().any()]

    for column in columns_with_missing_values:
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].median()
        elif pd.api.types.is_object_dtype(df[column]):
            fill_value = df[column].mode()[0]
        else:
            raise TypeError(f"Column {column} has unsupported dtype: {df[column].dtype}")

        df[column] = df[column].fillna(fill_value)

    print(f"Replaced missing values in {len(columns_with_missing_values) + 4} features.")

    return df


def impute_or_remove_features_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature-specific imputation rules, removes features with high missing ratios,
    and fills remaining missing values with median or mode.

    :param df: Dataset with missing values.
    :return: Dataset with missing values handled.
    """
    median_ager_type = df["ager_typ"].median()
    df["ager_typ"] = [median_ager_type if pd.isnull(t) and birth_year < 1960 else t
                      for t, birth_year in zip(df["ager_typ"], df["geburtsjahr"])]
    df["ager_typ"] = df["ager_typ"].fillna(-1)

    df["titel_kz"] = [1 if pd.notnull(flag) else 0 for flag in df["titel_kz"]]

    for child_age_column in ["alter_kind1", "alter_kind2"]:
        df[child_age_column] = df[child_age_column].fillna(0)

    df = remove_features_with_high_missing_values_ratio(df)

    df = impute_features_with_median_or_mode(df)

    return df


def split_formative_youth_years(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the decade and cultural orientation out of the formative youth years feature 'praegende_jugendjahre'.

    :param df: Dataset with formative youth years.
    :param meta: Metadata with meanings for each formative youth year value.
    :returns: Dataset with 'youth_decade' and 'youth_orientation' features.
    """
    youth_mappings = {
        "decade": {},
        "orientation": {},
    }

    youth_meanings = meta[meta["attribute"] == "praegende_jugendjahre"]

    for row in youth_meanings.itertuples(index=False):
        value = row.value
        meaning = row.meaning

        if meaning != "unknown":
            youth_mappings["decade"][value] = int(meaning.split(" - ")[0][:2])
            youth_mappings["orientation"][value] = meaning.split("(")[1].split(",")[0].lower()

    new_youth_data = {}
    for feature, feature_mapping in youth_mappings.items():
        new_youth_data[f"youth_{feature}"] = df["praegende_jugendjahre"].map(feature_mapping)

    df = pd.concat([df, pd.DataFrame(new_youth_data)], axis=1)

    df = df.drop(columns=["praegende_jugendjahre"])

    print("Extracted decade and cultural orientation as separate features.")

    return df


def split_international_cameo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts wealth and life phase out of the international cameo classification feature.

    :param df: Dataset with international cameo classification.
    :return: Dataset with 'cameo_wealth' and 'cameo_life_phase' features.
    """
    df["cameo_wealth"] = [int(str(v)[0]) if v != -1 and pd.notnull(v) else v for v in df["cameo_intl_2015"]]
    df["cameo_life_phase"] = [int(str(v)[1]) if v != -1 and pd.notnull(v) else v for v in df["cameo_intl_2015"]]

    df = df.drop(columns=["cameo_intl_2015"])

    print("Extracted wealth and life phase as separate features.")

    return df


def remove_features(df: pd.DataFrame, feature_configs: Dict[str, List[str]], removal_types: List[str]) -> pd.DataFrame:
    """
    Removes features from the dataset that were defined in a separate config.

    :param df: Dataset with features to remove.
    :param feature_configs: Config with specified feature lists.
    :param removal_types: reasons for the removal which also specifies the corresponding lists from the config
    :return: Dataset with defined features removed.
    """
    for removal_type in removal_types:
        features = feature_configs[removal_type]
        df = df.drop(columns=features)
        print(f"Removed {len(features)} {removal_type} features.")

    return df


def encode_binary_features(df: pd.DataFrame, binary_features: List[str]) -> pd.DataFrame:
    """
    Encodes specified binary features in the dataset to numerical (0/1) representation.

    :param df: Dataset containing the binary features to encode.
    :param binary_features: Column names in the dataset that are considered binary.
    :returns: Dataset with the specified binary features encoded numerically.
    """
    for feature in binary_features:
        unique_values = sorted(df[feature].unique())
        if len(unique_values) == 1:
            df[feature] = df[feature].map({unique_values[0]: 0})
        elif len(unique_values) == 2:
            df[feature] = df[feature].map({unique_values[0]: 0, unique_values[1]: 1})
        else:
            raise ValueError(f"Feature {feature} has more than two unique values: {unique_values}")

    print(f"Encoded {len(binary_features)} binary features.")

    return df


def encode_categorical_features(df: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encodes specified categorical features in the dataset using one-hot encoding.

    :param df: Dataset containing the categorical features to encode.
    :param categorical_features: Column names in the dataset that are considered categorical.
    :returns: - Dataset with the original categorical features replaced by the generated dummy variables.
              - Names of the created dummy variables.
    """
    for feature in categorical_features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = df[feature].astype(int)

    non_categorical_columns = set(df).difference(categorical_features)

    df = pd.get_dummies(
        df,
        columns=categorical_features,
        prefix=categorical_features,
        prefix_sep="_",
        drop_first=True,
        dtype=int
    )

    dummy_variables = list(set(df).difference(non_categorical_columns))

    print(f"One-hot encoded {len(categorical_features)} categorical features "
          f"into {len(dummy_variables)} dummy variables.")

    return df, dummy_variables


def scale_numeric_features(
        df: pd.DataFrame, numeric_features: List[str], df_name: str, scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scales specified numeric features in the dataset using StandardScaler.

    If a scaler is provided, it is used to transform the data. If no scaler is
    provided, a new StandardScaler is fitted to the numeric features and then used
    to transform them. The original numeric columns are dropped from the dataset
    and replaced with the scaled versions.

    :param df: Dataset containing the numeric features to scale.
    :param numeric_features: Column names in the dataset that are numeric.
    :param df_name: Name of the dataset for logging.
    :param scaler: An optional pre-fitted StandardScaler.
    :returns: - Dataset with the original numeric features replaced by the scaled versions.
              - StandardScaler used for scaling (either the provided one or a newly fitted one).
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])

    scaled_features = scaler.transform(df[numeric_features])
    scaled_features = pd.DataFrame(scaled_features, columns=numeric_features, index=df.index)

    df = df.drop(columns=list(numeric_features))
    df = pd.concat([df, scaled_features], axis=1)

    print(f"Scaled {len(numeric_features)} numeric features from {df_name} dataset.")

    return df, scaler


def clean_data(
        df: pd.DataFrame, meta: pd.DataFrame, feature_config: Dict[str, List[str]], df_name: str
) -> pd.DataFrame:
    """
    Clean the dataset by removing unnecessary columns, handling missing data and extracting new features.

    :param df: Dataset with uncleaned data attributes.
    :param meta: Metadata to handle null values.
    :param feature_config: Config for specific feature handling.
    :param df_name: Name of the dataset for logging.
    :return: Cleaned dataset.
    """
    print(f"\nStart cleaning {df_name} dataset...")

    df = remove_features(df, feature_config, ["irrelevant"])

    df = convert_unknown_values_to_null(df, meta)
    df = convert_invalid_values_to_null(df, meta)

    df = remove_records_with_insufficient_data(df)
    df = impute_or_remove_features_with_missing_values(df)

    df = split_formative_youth_years(df, meta)
    df = split_international_cameo(df)

    df = remove_features(df, feature_config, ["redundant", "uncertain"])

    return df


def encode_data(
        df: pd.DataFrame, feature_config: Dict[str, List[str]], df_name: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode categorical features by converting binary variables to 0/1 and using one-hot encoding
    for features with multiple categories.

    :param df: Dataset containing categorical features.
    :param feature_config: Config to identify categorical features.
    :param df_name: Name of the dataset for logging.
    :return: Dataset with encoded categorical features.
    """
    print(f"\nStart encoding {df_name} dataset...")

    df = encode_binary_features(df, feature_config["binary"])
    df, dummy_variables = encode_categorical_features(df, feature_config["categorical"])

    return df, dummy_variables


def complete_dummy_variables(df: pd.DataFrame, dummy_cols: List[str], other_dummy_cols: List[str]) -> pd.DataFrame:
    """
    Add dummy variables to the dataset that are completely set to 0 if the dummy variable is only present
    in the other dataset.

    :param df: Dataset possibly with missing dummy variables.
    :param dummy_cols: Dummy variables in the dataset.
    :param other_dummy_cols: Dummy variables of the other dataset.
    :return: Dataset with completed dummy variables.
    """
    missing_dummy_variables = set(other_dummy_cols).difference(set(dummy_cols))

    for missing_dummy_variable in missing_dummy_variables:
        df[missing_dummy_variable] = 0

    return df


def align_features(
        df1: pd.DataFrame, df2: pd.DataFrame, dummy_cols_df1: List[str], dummy_cols_df2: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Align features to ensure that the exact same features are present in the same order in both datasets.

    :param df1: First dataset.
    :param df2: Second dataset.
    :param dummy_cols_df1: Dummy variables in the first dataset.
    :param dummy_cols_df2: Dummy variables in the second dataset.
    :return: - First dataset with aligned features.
             - Second dataset with aligned features.
             - Unique Dummy variables unified from both datasets.
    """
    df1 = complete_dummy_variables(df1, dummy_cols_df1, dummy_cols_df2)
    df2 = complete_dummy_variables(df2, dummy_cols_df2, dummy_cols_df1)

    print("\nAligned categorical features after one-hot encoding.")

    dummy_variables = set(dummy_cols_df1) | (set(dummy_cols_df2))

    feature_intersection = sorted(list(set(df1) & set(df2)))
    df1 = df1[feature_intersection]
    df2 = df2[feature_intersection]

    print("Aligned numerical features.")

    return df1, df2, dummy_variables


def preprocess_data(df1: pd.DataFrame, df2: pd.DataFrame, names: Tuple[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the datasets through a series of data cleaning, feature engineering,
    encoding, and scaling steps.

    Both datasets will be aligned for the same feature space.

    :param df1: First raw input dataset to be preprocessed.
    :param df2: Second raw input dataset to be preprocessed.
    :param names: Names of the datasets for logging.
    :returns: - First preprocessed dataset.
              - Second preprocessed dataset.
    """
    meta = load_meta_data(raw_file_path="../data/meta/dias_values.xlsx")
    meta = rectify_meta_attributes(meta)

    with open("feature_config.json", "r") as f:
        feature_config = json.load(f)
    print("Feature config loaded.")

    df1 = clean_data(df1, meta, feature_config, names[0])
    df2 = clean_data(df2, meta, feature_config, names[1])

    df1, dummy_variables_df1 = encode_data(df1, feature_config, names[0])
    df2, dummy_variables_df2 = encode_data(df2, feature_config, names[1])

    df1, df2, dummy_variables = align_features(df1, df2, dummy_variables_df1, dummy_variables_df2)

    print(f"\nStart scaling datasets...")
    non_numeric_features = set(dummy_variables) | set(feature_config["binary"])
    numeric_features = [feature for feature in df1.columns if feature not in non_numeric_features]
    df1, scaler = scale_numeric_features(df1, numeric_features, names[0])
    df2, _ = scale_numeric_features(df2, numeric_features, names[1], scaler)

    print(f"Finish preprocessing for both datasets.")

    return df1, df2


if __name__=="__main__":
    customer = pd.read_csv("../data/Udacity_CUSTOMERS_052018.csv", sep=";", nrows=100000)
    customer.columns = customer.columns.str.lower()
    customer = customer.drop(columns=["customer_group", "online_purchase", "product_group"])

    population = pd.read_csv("../data/Udacity_AZDIAS_052018.csv", sep=";", nrows=100000)
    population.columns = population.columns.str.lower()

    population, customer = preprocess_data(population, customer, ("population", "customer"))