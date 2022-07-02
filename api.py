import json
import os
import pickle
import sys
from copyreg import pickle

import pandas as pd
import xgboost
from flask import Flask, jsonify, render_template, request

import utils

app = Flask(__name__)


@app.route("/home", methods=["GET"])
def homepage():
    return """<h1> 'Welcome on the credit score prediction home page' </h1>"""


@app.route("/prediction/<ID>", methods=["GET"])
def model(ID):
    data = utils.get_data()
    info = utils.get_info(int(ID), data)
    infos = json.loads(info.to_json(orient="table", index=False))
    seuil, score, model, X_test = utils.get_model_and_score(info)
    return """<h1> Le score pour le client {} est de {} </h1>""".format(ID, round(float(score), 2))


@app.route("/importance_loc/<ID>", methods=["GET"])
def importance_locale(ID):
    data = utils.get_data()
    info = utils.get_info(int(ID), data)
    seuil, score, model, X_test = utils.get_model_and_score(info)
    imp_loc = utils.get_local_feat_imp(model, X_test, info)
    return imp_loc.iloc[:, 0:2].to_html(header=True, table_id="importance locale")


@app.route("/importance_glob/<ID>", methods=["GET"])
def importance_globale(ID):
    data = utils.get_data()
    info = utils.get_info(int(ID), data)
    seuil, score, model, X_test = utils.get_model_and_score(info)
    imp_glob = utils.get_global_feat_imp(model, X_test)
    return (imp_glob.to_html(header=True, table_id="importance globale"),)


if __name__ == "__main__":
    app.run(debug=True)
