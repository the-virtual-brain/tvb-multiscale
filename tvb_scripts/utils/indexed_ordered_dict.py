# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import sort_dict, ensure_list, is_integer

LOG = initialize_logger(__name__)


class IndexedOrderedDict(object):
    _dict = OrderedDict()

    # TODO: Find out error when initializing with no argument to __init__
    def __init__(self, input_dict=OrderedDict({})):
        if not isinstance(input_dict, OrderedDict):
            if isinstance(input_dict, dict):
                # LOG.warning("input_dict is not an Ordered dict!\n"
                #             "Ordering it by sorting its keys!")
                input_dict = sort_dict(input_dict)
            else:
                raise ValueError("input node_ordered_dict is not a dict!:\n%s" % str(input_dict))

        self._dict = input_dict

    def __getitem__(self, keys):
        output = []
        if isinstance(keys, slice):
            try:
                output = (np.array(list(self._dict.values()))[keys]).tolist()
            except:
                LOG.warning("keys %s not found in IndexedOrderedDict %s!\n"
                            "Returning None!" % (str(keys), str(self._dict)))
        else:
            output = []
            for key in ensure_list(keys):
                try:
                    if is_integer(key):
                        output.append(list(self._dict.values())[key])
                    else:  # assuming key string
                        output.append(self._dict[key])
                except:
                    LOG.warning("key %s not found in IndexedOrderedDict %s!\n"
                                "Returning None!" % (str(key), str(self._dict)))
        n_outputs = len(output)
        if n_outputs == 0:
            return None
        elif n_outputs == 1:
            return output[0]
        else:
            return output

    def __setitem__(self, keys, values):
        if isinstance(keys, slice):
            keys = (np.array(list(self._dict.keys()))[keys]).tolist()
        for key, val in zip(ensure_list(keys), ensure_list(values)):
            try:
                if is_integer(key):
                    list(self._dict.values())[key] = val
                else:  # assuming key string
                    self._dict[key] = val
            except:
                LOG.warning("key %s not found in IndexedOrderedDict %s!\n"
                            "Skipping item setting with value %s!" % (str(key), str(self._dict), str(val)))

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def update(self, *args, **kwargs):
        if isinstance(args[0], IndexedOrderedDict):
            self._dict.update(args[0]._dict)
        else:
            self._dict.update(*args, **kwargs)

    def __delitem__(self, key):
        del self._dict[key]

    def pop(self, key):
        return self._dict.pop(key, None)

    def get(self, key):
        return self._dict.get(key, None)

    def __getattr__(self, attr):
        getattr(self._dict, attr)
