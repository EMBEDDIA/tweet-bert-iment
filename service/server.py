import argparse
import logging.handlers
import sys
from pathlib import Path
import random
from typing import Any, Dict, List

import bottle
import yaml
from bottle import Bottle, request, response, run
from bottle_swagger import SwaggerPlugin
from transformers import pipeline

#
# START INIT
#

# CLI parameters
parser = argparse.ArgumentParser(description="Run the sentiment analysis server.")
parser.add_argument("port", type=int, default=8080, help="port number to attach to")
args = parser.parse_args()
sys.argv = sys.argv[0:1]

log = logging.getLogger("root")
log.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

# Bottle
app = Bottle()

# Swagger
with open(Path(__file__).parent / "swagger.yml", "r") as file_handle:
    swagger_def = yaml.load(file_handle, Loader=yaml.FullLoader)
app.install(
    SwaggerPlugin(swagger_def, serve_swagger_ui=True, swagger_ui_suburl="/documentation/", validate_requests=False)
)

log.info("here")


def allow_cors(opts):
    def decorator(func):
        """ this is a decorator which enables CORS for specified endpoint """

        def wrapper(*args, **kwargs):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(opts)
            response.headers["Access-Control-Allow-Headers"] = (
                "Origin, Accept, Content-Type, X-Requested-With, " "X-CSRF-Token"
            )

            # Only respond with body for non-OPTIONS
            if bottle.request.method != "OPTIONS":
                return func(*args, **kwargs)

        return wrapper

    return decorator

model_path_sl = "EMBEDDIA/sloberta-tweetsentiment"
model_path_hr = "EMBEDDIA/bertic-tweetsentiment"
model_path_ru = "EMBEDDIA/rubert-tweetsentiment"
model_path_en = "EMBEDDIA/english-tweetsentiment"
sentiment_sl = pipeline("sentiment-analysis", model=model_path_sl, tokenizer=model_path_sl, use_fast=False)
sentiment_hr = pipeline("sentiment-analysis", model=model_path_hr, tokenizer=model_path_hr)
sentiment_ru = pipeline("sentiment-analysis", model=model_path_ru, tokenizer=model_path_ru)
sentiment_en = pipeline("sentiment-analysis", model=model_path_en, tokenizer=model_path_en)

#
# END INIT
#


def analyze(comments: List[str], lang) -> List[float]:
    sentiment_task = {'slovene': sentiment_sl, 'croatian': sentiment_hr, 'slovenian': sentiment_sl, 'english': sentiment_en, 'russian': sentiment_ru, 'sl': sentiment_sl, 'hr': sentiment_hr, 'en': sentiment_en, 'ru': sentiment_ru}
    results = sentiment_task[lang.lower()](comments)
    #label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
    #results = [(label_map[res["label"]], res["score"]) for res in results]
    results = [(res["label"], res["score"]) for res in results]
    #results = [(label, -likelihood if label == "Negative" else likelihood) for (label, likelihood) in results]
    #results = [(label, 0 if label == "Neutral" else likelihood) for (label, likelihood) in results]
    results = [label for (label, likelihood) in results]
    return results


@app.route("/analyze", method=["POST", "OPTIONS"])
@allow_cors(["POST", "OPTIONS"])
def api_generate_json() -> Dict[str, Any]:
    parameters = request.json
    #print(parameters)
    if not parameters:
        response.status = 400
        return {"errors": ["Missing or empty request body"]}
    comments = parameters.get("comments")
    if not comments:
        response.status = 400
        return {"errors": ["Invalid or missing comment list."]}
    return {"sentiments": analyze(comments, parameters.get("lang"))}


@app.route("/health", method=["GET", "OPTIONS"])
@allow_cors(["GET", "OPTIONS"])
def health() -> Dict[str, Any]:
    return {"version": "1.0.0"}


def main() -> None:
    log.info(f"Starting server at {args.port}")
    run(app, server="meinheld", host="0.0.0.0", port=args.port)
    log.info("Stopping")


if __name__ == "__main__":
    main()
