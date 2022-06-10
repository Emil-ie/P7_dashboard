#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

pd.set_option("display.max_column", 999)
pd.set_option("display.max_row", 999)
import os
import pickle

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import shap
from PIL import Image

path = os.getcwd()


def get_data(path):

    data = pd.read_pickle(os.path.join(path, "data/dashboard_data.pkl"), index_col=0).drop(
        columns=["TARGET"]
    )
    return data


def get_info(SK_ID_CURR_user, data):
    return (
        data[data.SK_ID_CURR == SK_ID_CURR_user]
        .drop(["index", "SK_ID_CURR"], axis=1)
        .dropna(axis="columns")
    )


def show_logo():
    return Image.open(os.path.join(path, "img/LOGO.png"))


def get_model_and_score(info):
    model = pickle.load(open(os.path.join(path, "model/model_pkl2"), "rb"))
    seuil = 0.31

    # import des données
    X_test = pd.read_pickle(os.path.join(path, "data/test.pkl"), index_col=0)
    prb = model.predict_proba(X_test)
    prb = pd.DataFrame(prb)
    prb_score = prb[1]
    prb_score.index = X_test.index

    score = prb_score.loc[info.index].iloc[0]

    return seuil, score, model, X_test


def convertUpperAndLowerBoundAndThreshold(
    value, oldMin, oldMax, oldThreshold, newMin, newMax, newThreshold
):

    if value < oldThreshold:
        oldMax = oldThreshold
        newMax = newThreshold
    else:
        oldMin = oldThreshold
        newMin = newThreshold

    return ((value - oldMin) * ((newMax - newMin) / (oldMax - oldMin))) + newMin


def gauge_chart(score, minScore, maxScore, threshold):

    localThreshold = 0
    convertedScore = convertUpperAndLowerBoundAndThreshold(
        value=score,
        oldMin=minScore,
        oldMax=maxScore,
        oldThreshold=threshold,
        newMin=-1,
        newMax=1,
        newThreshold=localThreshold,
    )

    color = "RebeccaPurple"
    if convertedScore < localThreshold:
        color = "darkred"
    else:
        color = "green"
    fig = go.Figure(
        go.Indicator(
            domain={"x": [0, 0.9], "y": [0, 0.9]},
            value=convertedScore,
            mode="gauge+number+delta",
            title={"text": "Score"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": color},
                "steps": [
                    {"range": [-1, -0.8], "color": "#ff0000"},
                    {"range": [-0.8, -0.6], "color": "#ff4d00"},
                    {"range": [-0.6, -0.4], "color": "#ff7400"},
                    {"range": [-0.4, -0.2], "color": "#ff9a00"},
                    {"range": [-0.2, 0], "color": "#ffc100"},
                    {"range": [0, 0.2], "color": "#c5ff89"},
                    {"range": [0.2, 0.4], "color": "#b4ff66"},
                    {"range": [0.4, 0.6], "color": "#a3ff42"},
                    {"range": [0.6, 0.8], "color": "#91ff1e"},
                    {"range": [0.8, 1], "color": "#80f900"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 8},
                    "thickness": 0.75,
                    "value": convertedScore,
                },
            },
            #  delta = {'reference': 0.5, 'increasing': {'color': "RebeccaPurple"}}
        )
    )
    return fig


# feature importance du modèle
def get_global_feat_imp(model, X_test):
    feat_imp = model.feature_importances_
    feat_importance = pd.DataFrame(columns=["Feature Name", "Global Feature Importance"])
    feat_importance["Feature Name"] = pd.Series(X_test.columns)
    feat_importance["Global Feature Importance"] = pd.Series(feat_imp)
    top_features = feat_importance.sort_values(
        by="Global Feature Importance", ascending=False
    ).head(10)

    return top_features


def get_local_feat_imp(model, X_test, info):
    shap_values = shap.TreeExplainer(model).shap_values(X_test.loc[info.index])[0]
    dfShap = pd.DataFrame([shap_values], columns=X_test.columns)
    serieSignPositive = dfShap.iloc[0, :].apply(lambda col: True if col >= 0 else False)
    serieValues = dfShap.iloc[0, :]
    serieAbsValues = abs(serieValues)
    local = (
        pd.DataFrame(
            {
                "Local Feature Importance": serieValues,
                "absValues": serieAbsValues,
                "positive": serieSignPositive,
                "color": map(lambda x: "red" if x else "blue", serieSignPositive),
            }
        )
        .sort_values(by="absValues", ascending=False)
        .iloc[:12, :]
        .drop("absValues", axis=1)
    )
    local.reset_index(inplace=True)
    local.rename(columns={"index": "Feature Name"}, inplace=True)
    return local


def plot_local(local):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Variable locale importante", fontsize=20)
    sns.barplot(data=local, x="Local Feature Importance", y="Feature Name", palette=local.color)
    return fig


def plot_global(top_features):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Variable globale importante", fontsize=20)
    sns.barplot(
        data=top_features, x="Global Feature Importance", y="Feature Name", palette="crest"
    )
    return fig
