import argparse
import json
import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def test_deployed_app_post_request(args):
    app_url = args.url + "/predict"
    logger.info(f"Testing the live api -> {app_url}...")

    expected_output = 0

    input_data = {
        "age": 26,
        "workclass": " Private",
        "fnlgt": 222539,
        "education": " Some-college",
        "education-num": 10,
        "marital-status": " Married-civ-spouse",
        "occupation": " Craft-repair",
        "relationship": " Husband",
        "race": " White",
        "sex": " Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": " United-States",
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(app_url, data=json.dumps(input_data), headers=headers)
    assert r.status_code == 200
    assert r.json()["prediction"] == expected_output
    response = {"prediction": r.json()["prediction"], "status": r.status_code}
    return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Census Bureau Render App Predictions Test CLI"
    )

    parser.add_argument(
        "url",
        type=str,
        help="url[:port] of the app to test inferences (e.g. http://127.0.0.1:8000)",
    )

    args = parser.parse_args()
    # Call live testing function
    res = test_deployed_app_post_request(args)
    logger.info(f"Result of the post request: {res}")
