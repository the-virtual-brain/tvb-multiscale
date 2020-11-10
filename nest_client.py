import logging

import requests

LOG = logging.getLogger(__name__)


class NestClient:
    def __init__(self, rest_server):
        self.api_url = rest_server
        self.available_calls = self.get_available_calls()

    def get_available_calls(self):
        url = self.api_url + "/api/"
        res = requests.get(url, headers={"Accept": "application/json"})
        res.raise_for_status()
        return res.json()

    def nest_call(self, method, data=None):
        assert method in self.available_calls
        if data is None:
            data = {}
        url = self.api_url + "/api/{}".format(method)
        res = requests.post(url, headers={"Accept": "application/json"}, json=data)
        if not res.ok:
            LOG.error(res.text)
        res.raise_for_status()
        return res.json()


if __name__ == '__main__':
    nest_client = NestClient("http://localhost:5000")
    print(nest_client.available_calls)

    models = nest_client.nest_call("Models")
    print("Available models: {}".format(models))

    data = {
        "model": "aeif_cond_alpha",
        "n": 2
    }
    response = nest_client.nest_call("Create", data)
    print("Create response: {}".format(response))

    # Invalid requests
    nest_client.nest_call("Create")
    nest_client.nest_call("Blabla")
