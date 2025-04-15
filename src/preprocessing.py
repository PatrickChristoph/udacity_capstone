import json
from typing import List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_meta_data(raw_file_path: str) -> pd.DataFrame:
    meta = pd.read_excel(raw_file_path, header=1)
    meta.columns = meta.columns.str.lower()
    meta.drop(columns="unnamed: 0", inplace=True)

    meta[["attribute", "description"]] = meta[["attribute", "description"]].ffill()
    meta["attribute"] = meta["attribute"].str.lower()

    meta["meaning"] = meta["meaning"].ffill()

    meta["value"] = [v.split(", ") if type(v) == str and "," in v else v for v in meta["value"]]
    meta = meta.explode("value")

    return meta


def rectify_meta_attributes(meta: pd.DataFrame) -> pd.DataFrame:
    meta["attribute"] = meta["attribute"].replace(r"_rz$", "", regex=True)

    attribute_renaming = {
        "d19_buch": "d19_buch_cd",
        "d19_kk_kundentyp": "kk_kundentyp",
        "kba13_ccm_1400_2500": "kba13_ccm_1401_2500",
        "soho_flag": "soho_kz"
    }

    meta["attribute"] = meta["attribute"].replace(attribute_renaming)

    return meta


def prepare_cameo_classifications(df: pd.DataFrame) -> pd.DataFrame:
    cameo_columns = df.columns[df.columns.str.startswith("cameo")]
    df[cameo_columns] = df[cameo_columns].replace(["X", "XX"], np.nan)

    for cameo_column in cameo_columns:
        if cameo_column != "cameo_deu_2015":
            df[cameo_column] = df[cameo_column].astype(float)

    return df


def convert_unknown_values_to_null(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    unknown_values = meta[meta["meaning"].str.contains("unknown")]

    for attribute, unknown_value in zip(unknown_values["attribute"], unknown_values["value"]):
        if attribute in df:
            df[attribute] = df[attribute].replace(float(unknown_value), np.nan)

    return df


def identify_invalid_oor_values(meta: pd.DataFrame, df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    categorical_values = meta[meta["value"] != "…"][["attribute", "value"]]
    categorical_values = categorical_values.rename(columns={"value": "meta_value"})

    invalid_oor_values = []

    for attribute in categorical_values["attribute"].unique():
        if attribute in df:
            pop_vc = df[attribute].value_counts().reset_index()
            pop_vc = pop_vc.merge(
                categorical_values[categorical_values["attribute"] == attribute]["meta_value"],
                left_on=attribute,
                right_on="meta_value",
                how="left",
            )

            if sum(pop_vc["meta_value"].isnull()) > 0:
                for out_of_range_value in pop_vc[pop_vc["meta_value"].isnull()].itertuples():
                    if log:
                        print(f"attribute: {attribute}, invalid value: {out_of_range_value[1]}, "
                              f"affected rows: {out_of_range_value.count}")
                    invalid_oor_values.append([attribute, out_of_range_value[1]])

    return pd.DataFrame(data=invalid_oor_values, columns=["attribute", "value"])


def convert_invalid_values_to_null(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["geburtsjahr"] == 0, "geburtsjahr"] = np.nan
    df.loc[df["anz_personen"] == 0, "anz_personen"] = np.nan

    invalid_oor_values = identify_invalid_oor_values(meta, df, log=True)
    for attribute, invalid_oor_value in zip(invalid_oor_values["attribute"], invalid_oor_values["value"]):
        if attribute in df:
            df[attribute] = df[attribute].replace(invalid_oor_value, np.nan)

    return df


def remove_records_with_insufficient_data(df: pd.DataFrame, max_missing_values: int = 120) -> pd.DataFrame:
    df =df[~(df.isnull().sum(axis=1) >= max_missing_values)]
    return df


def remove_attributes_with_high_missing_values_ratio(df: pd.DataFrame, max_ratio: float = .2) -> pd.DataFrame:
    missing_ratios = pd.Series(df.isnull().sum() / len(df))
    columns_with_high_ratio = list(missing_ratios[missing_ratios >= max_ratio].index)

    df = df.drop(columns=columns_with_high_ratio)
    print(f"\nRemoved {len(columns_with_high_ratio)} attributes due to high missing ratio:"
          f"\n{", ".join(columns_with_high_ratio)}")

    return df


def impute_attributes_with_median_or_mode(df: pd.DataFrame) -> pd.DataFrame:
    columns_with_missing_values = df.columns[df.isnull().any()]

    for column in columns_with_missing_values:
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].median()
        elif pd.api.types.is_object_dtype(df[column]):
            fill_value = df[column].mode()[0]
        else:
            raise TypeError(f"Column {column} has unsupported dtype: {df[column].dtype}")

        df[column] = df[column].fillna(fill_value)

    print(f"\nReplaced missing values in {len(columns_with_missing_values)} attributes:"
          f"\n{", ".join(columns_with_missing_values)}")

    return df


def impute_or_remove_attributes_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    median_ager_type = df["ager_typ"].median()
    df["ager_typ"] = [median_ager_type if pd.isnull(t) and birth_year < 1960 else t
                      for t, birth_year in zip(df["ager_typ"], df["geburtsjahr"])]
    df["ager_typ"] = df["ager_typ"].fillna(-1)

    df["titel_kz"] = [1 if pd.notnull(flag) else 0 for flag in df["titel_kz"]]

    for child_age_column in ["alter_kind1", "alter_kind2"]:
        df[child_age_column] = df[child_age_column].fillna(0)

    df = remove_attributes_with_high_missing_values_ratio(df)

    df = impute_attributes_with_median_or_mode(df)

    return df


def split_formative_youth_years(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:

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

    return df


def split_international_cameo(df: pd.DataFrame) -> pd.DataFrame:
    df["cameo_wealth"] = [int(str(v)[0]) if v != -1 and pd.notnull(v) else v for v in df["cameo_intl_2015"]]
    df["cameo_life_phase"] = [int(str(v)[1]) if v != -1 and pd.notnull(v) else v for v in df["cameo_intl_2015"]]

    df = df.drop(columns=["cameo_intl_2015"])

    return df


def remove_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df = df.drop(columns=features)
    print(f"Removed {len(features)} features: {", ".join(features)}")

    return df


def encode_binary_features(df: pd.DataFrame, binary_features: List[str]) -> pd.DataFrame:
    for feature in binary_features:
        unique_values = sorted(df[feature].unique())
        if len(unique_values) == 1:
            df[feature] = df[feature].map({unique_values[0]: 0})
        elif len(unique_values) == 2:
            df[feature] = df[feature].map({unique_values[0]: 0, unique_values[1]: 1})
        else:
            raise ValueError(f"Feature {feature} has more than two unique values: {unique_values}")

    print(f"Encoded {len(binary_features)} binary features: {", ".join(binary_features)}")

    return df


def encode_categorical_features(df: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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

    print(f"One-hot encoded {len(categorical_features)} categorical features into {len(dummy_variables)} dummy "
          f"variables: {", ".join(categorical_features)}")

    return df, dummy_variables


def scale_numeric_features(
        df: pd.DataFrame, numeric_features: List[str], scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])

    scaled_features = scaler.transform(df[numeric_features])
    scaled_features = pd.DataFrame(scaled_features, columns=numeric_features, index=df.index)

    df = df.drop(columns=list(numeric_features))
    df = pd.concat([df, scaled_features], axis=1)

    print(f"Scaled {len(numeric_features)} numeric features: {", ".join(numeric_features)}")

    return df, scaler



def preprocess_data(df: pd.DataFrame, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:

    meta = load_meta_data(raw_file_path="../data/meta/dias_values.xlsx")
    meta = rectify_meta_attributes(meta)

    with open("feature_config.json", "r") as f:
        feature_config = json.load(f)

    df = remove_features(df, feature_config["worthless"])

    df = prepare_cameo_classifications(df)

    df = convert_unknown_values_to_null(df, meta)
    df = convert_invalid_values_to_null(df, meta)

    df = remove_records_with_insufficient_data(df)
    df = impute_or_remove_attributes_with_missing_values(df)

    df = split_formative_youth_years(df, meta)
    df = split_international_cameo(df)

    df = remove_features(df, feature_config["redundant"])

    df = encode_binary_features(df, feature_config["binary"])
    df, dummy_variables = encode_categorical_features(df, feature_config["categorical"])

    df = remove_features(df, feature_config["uncertain"])

    non_numeric_features = set(dummy_variables) | set(feature_config["binary"])
    numeric_features = [feature for feature in df.columns if feature not in non_numeric_features]

    df, scaler = scale_numeric_features(df, numeric_features, scaler)

    return df, scaler