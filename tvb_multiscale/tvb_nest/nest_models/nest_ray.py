# -*- coding: utf-8 -*-

from six import string_types
import json
import ray

import numpy
import pandas

from tvb_multiscale.tvb_nest.config import CONFIGURED

from tvb_multiscale.core.utils.data_structures_utils import is_iterable

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, dicts_of_lists_to_lists_of_dicts, list_of_dicts_to_dict_of_tuples


#@serve.deployment
class RayNESTServer(object):

    def __init__(self, config=CONFIGURED):
        from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel
        self.nest_instance = configure_nest_kernel(load_nest(config=config), config=config)

    def _node_collection_to_gids(self, node_collection):
        return node_collection.global_id

    def NodeCollection(self, node_gids, *args, **kwargs):
        output = self.nest_instance.NodeCollection(tuple(ensure_list(node_gids)), *args, **kwargs)
        return output

    def _synapse_collection_to_dict(self, synapse_collection):
        return synapse_collection.get()

    def SynapseCollection(self, elements):
        synapse_model = numpy.unique(elements["synapse_model"])
        if numpy.all(synapse_model == synapse_model[0]):
            synapse_model = synapse_model[0]
        else:
            synapse_model = None
        return self.nest_instance.GetConnections(source=self.NodeCollection(numpy.unique(elements["source"])),
                                                 target=self.NodeCollection(numpy.unique(elements["target"])),
                                                 synapse_model=synapse_model)

    @ray.method(num_returns=1)
    def nest(self, attr, *args, **kwargs):
        # For all calls that do not have NodeCollection or SynapseCollection as inputs or outputs
        return getattr(self.nest_instance, attr)(*args, **kwargs)

    def _nodes_or_synapses(self, elements):
        if isinstance(elements, dict):
            # assuming these are connections:
            return self.SynapseCollection(elements)
        elif isinstance(elements, (tuple, list)):
            # assuming these are nodes:
            return self.NodeCollection(elements)
        else:
            return elements

    @ray.method(num_returns=1)
    def get(self, nodes, *params, **kwargs):
        if len(nodes):
            return self._nodes_or_synapses(nodes).get(*params, **kwargs)
        else:
            return ()

    @ray.method(num_returns=1)
    def set(self, nodes, params=None, **kwargs):
        if len(nodes):
            return self._nodes_or_synapses(nodes).set(params, **kwargs)
        else:
            return None

    @ray.method(num_returns=1)
    def GetStatus(self, nodes, attrs=None, output=None):
        if len(nodes):
            output = self.nest_instance.GetStatus(self._nodes_or_synapses(nodes), attrs, output)
        else:
            return ()

    @ray.method(num_returns=1)
    def SetStatus(self, nodes, params, val):
        if len(nodes):
            return self.nest_instance.SetStatus(self._nodes_or_synapses(nodes), params, val)
        else:
            return None

    @ray.method(num_returns=1)
    def install_nest(self, *args, **kwargs):
        from pynestml.frontend.pynestml_frontend import install_nest
        return install_nest(*args, **kwargs)

    @ray.method(num_returns=1)
    def help(self, obj=None, return_text=True):
        # TODO: a warning for when return_text = False
        return self.nest_instance.help(obj=self._nodes_or_synapses(obj), return_text=True)

    @ray.method(num_returns=1)
    def Create(self, model, n=1, params=None, positions=None):
        return self._node_collection_to_gids(
            self.nest_instance.Create(model, n=n, params=params, positions=positions))

    @ray.method(num_returns=1)
    def GetNodes(self, properties={}, local_only=False):
        return self._node_collection_to_gids(
            self.nest_instance.GetNodes(properties=properties, local_only=local_only))

    @ray.method(num_returns=1)
    def GetLocalNodeCollection(self, node_inds):
        if len(node_inds):
            return self._node_collection_to_gids(
                self.nest_instance.GetLocalNodeCollection(self.NodeCollection(node_inds)))
        else:
            return ()

    @ray.method(num_returns=1)
    def Connect(self, pre, post, conn_spec=None, syn_spec=None, return_synapsecollection=False):
        if return_synapsecollection:
            return self._synapse_collection_to_dict(
                self.nest_instance.Connect(self.NodeCollection(pre),
                                           self.NodeCollection(post),
                                           conn_spec=conn_spec, syn_spec=syn_spec,
                                           return_synapsecollection=True))
        else:
            return self.nest_instance.Connect(self.NodeCollection(pre),
                                              self.NodeCollection(post),
                                              conn_spec=conn_spec, syn_spec=syn_spec,
                                              return_synapsecollection=False)

    @ray.method(num_returns=1)
    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse'):
        return self.nest_instance.Disconnect(self.NodeCollection(pre),
                                             self.NodeCollection(post),
                                             conn_spec=conn_spec, syn_spec=syn_spec)

    @ray.method(num_returns=1)
    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None):
        if source is not None:
            source = self.NodeCollection(source)
        if target is not None:
            target = self.NodeCollection(target)
        return self._synapse_collection_to_dict(
            self.nest_instance.GetConnections(source=source, target=target,
                                              synapse_model=synapse_model, synapse_label=synapse_label))

    @ray.method(num_returns=1)
    def CopyModel(self, existing, new, params=None):
        self.nest_instance.CopyModel(existing, new, params)
        return new


def serializable(data):
    """Make data serializable for JSON.
       Modified from pynest utils.

    Parameters
    ----------
    data : any

    Returns
    -------
    data_serialized : str, int, float, list, dict
        Data can be encoded to JSON
    """

    if isinstance(data, (numpy.ndarray, RayNodeCollection)):
        return data.tolist()
    if isinstance(data, RaySynapseCollection):
        # Get full information from SynapseCollection
        return serializable(data.todict())
    if isinstance(data, (list, tuple)):
        return [serializable(d) for d in data]
    if isinstance(data, dict):
        return dict([(key, serializable(value)) for key, value in data.items()])
    return data


def to_json(data, **kwargs):
    """Serialize data to JSON.
       Modified from pynest utils.

    Parameters
    ----------
    data : any
    kwargs : keyword argument pairs
        Named arguments of parameters for `json.dumps` function.

    Returns
    -------
    data_json : str
        JSON format of the data
    """

    data_serialized = serializable(data)
    data_json = json.dumps(data_serialized, **kwargs)
    return data_json


class RayNodeCollectionIterator(object):
    """
    Iterator class for `RayNodeCollectionIterator`.
    Modifying the corresponding class for NodeCollectionIterator of pynest.

    Returns
    -------
    `NodeCollection`:
        Single node ID `RayNodeCollectionIterator` of respective iteration.
    """

    def __init__(self, nc):
        self._nc = nc
        self._increment = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._increment > len(self._nc) - 1:
            raise StopIteration

        val = self._nc[self._increment]
        self._increment += 1
        return val


class RayNodeCollection(object):
    """
    Class for `RayNodeCollection`.
    Modifying the corresponding class for NodeCollection of pynest.
    """

    nest_instance = None

    gids = ()

    def __init__(self, nest_instance, gids=()):
        self.nest_instance = nest_instance
        self.gids = tuple(numpy.unique(gids))

    def __getattr__(self, attr):
        if self.nest_instance:
            return self.get(attr)
        raise AttributeError("type object %s has no attribute %s" % (self.__class__.__name__, attr))

    def __setattr__(self, attr, value):
        if attr in ["nest_instance", "gids"]:
            super(RayNodeCollection, self).__setattr__(attr, value)
        elif self.nest_instance:
            return self.set({"attr": value})

    def __getstate__(self):
        return {"gids": self.gids, "nest_instance": self.nest_instance}

    def __setstate__(self, d):
        super(RayNodeCollection, self).__setattr__("gids", d.get("gids", ()))
        super(RayNodeCollection, self).__setattr__("nest_instance", d.get("nest_instance", None))

    def __iter__(self):
        return RayNodeCollectionIterator(self)

    def __add__(self, other):
        if not isinstance(other, RayNodeCollection):
            raise NotImplementedError()

        return self.__class__(self.nest_instance, self.gids + other.gids)

    def __getitem__(self, key):
        return self.__class__(self.nest_instance, tuple(ensure_list(numpy.array(self.gids)[key])))

    def __contains__(self, node_id):
        return node_id in self.gids

    def __eq__(self, other):
        if not isinstance(other, RayNodeCollection):
            raise NotImplementedError('Cannot compare NodeCollection to {}'.format(type(other).__name__))

        if self.__len__() != other.__len__():
            return False

        return self.gids == other.gids

    def __neq__(self, other):
        if not isinstance(other, RayNodeCollection):
            raise NotImplementedError()

        return not self == other

    def __len__(self):
        return len(self.gids)

    def __str__(self):
        return "%s: %s" % (self.__class__.__name__, str(self.gids))

    def __repr__(self):
        return "%s:\nnest_instance=%s\ngids=%s" % (self.__class__.__name__, str(self.nest_instance), str(self.gids))

    def get(self, *params, **kwargs):
        return self.nest_instance.get(self, *params, **kwargs)

    def set(self, params=None, **kwargs):
        return self.nest_instance.set(self, params, **kwargs)

    def tolist(self):
        """
        Convert `NodeCollection` to list.
        """
        return list(self.gids)

    def index(self, node_id):
        """
        Find the index of a node ID in the `NodeCollection`.

        Parameters
        ----------
        node_id : int
            Global ID to be found.

        Raises
        ------
        ValueError
            If the node ID is not in the `NodeCollection`.
        """
        return list(self.gids).index(node_id)

    def __bool__(self):
        """Converts the NodeCollection to a bool. False if it is empty, True otherwise."""
        return len(self) > 0

    def __array__(self, dtype=None):
        """Convert the NodeCollection to a NumPy array."""
        return numpy.array(self.tolist(), dtype=dtype)


class RaySynapseCollectionIterator(object):
    """
    Iterator class for RaySynapseCollection.
    Modifying the corresponding class for SynapseCollectionIterator of pynest.
    """

    def __init__(self, synapse_collection):
        self.nest_instance = synapse_collection.nest_instance
        self._iter = iter(dicts_of_lists_to_lists_of_dicts(synapse_collection.todict()))

    def __iter__(self):
        return self

    def __next__(self):
        return RaySynapseCollection(self.nest_instance, next(self._iter))


class RaySynapseCollection(object):
    """
    Class for RaySynapseCollection.
    Modifying the corresponding class for SynapseCollection of pynest.

    """

    nest_instance = None

    _attributes = ["source", "target", "weight", "delay", "receptor",
                   "synapse_model", "synapse_id", "port", "target_thread"]
    source = ()
    target = ()
    weight = ()
    delay = ()
    receptor = ()
    target_thread = ()
    port = ()
    synapse_id = ()
    synapse_model = ()

    def __init__(self, nest_instance, conns={}):
        self.nest_instance = nest_instance
        self.fromdict(conns)

    def fromdict(self, conns={}):
        for key, val in conns.items():
            if key not in self._attributes:
                self._attributes.append(key)
            setattr(self, key, tuple(ensure_list(val)))
        n = len(self.source)
        assert [len(getattr(self, attr)) == n for attr in self._attributes[1:]]

    def todict(self, keys=None):
        if keys is None:
            keys = self._attributes
        else:
            for key in keys:
                if key not in self._attributes:
                    raise ValueError("%s is not an attribute of %s!" % (key, self.__class__.__name__))
        output = {}
        for attr in keys:
            output[attr] = getattr(self, attr)
        return output

    def __getstate__(self):
        d = self.todict()
        d["nest_instance"] = self.nest_instance
        d["_attributes"] = self._attributes
        return d

    def __setstate__(self, d):
        super(RaySynapseCollection, self).__setattr__("nest_instance", d.pop("nest_instance", None))
        super(RaySynapseCollection, self).__setattr__("_attributes", d.pop("_attributes", self._attributes))
        self.fromdict(d)

    def __iter__(self):
        return RaySynapseCollectionIterator(self)

    def __len__(self):
        return len(self.source)

    def __eq__(self, other):
        if not isinstance(other, RaySynapseCollection):
            raise NotImplementedError()

        if self.__len__() != other.__len__():
            return False

        for attr in ['source', 'target', 'target_thread', 'synapse_id', 'port']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __neq__(self, other):
        if not isinstance(other, RaySynapseCollection):
            raise NotImplementedError()
        return not self == other

    def __getitem__(self, key):
        return self.__class__(self.nest_instance,
                              list_of_dicts_to_dict_of_tuples(list(numpy.array(self.todict())[key])))

    def __str__(self):
        """
        Printing a `SynapseCollection` returns something of the form:
            *--------*-------------*
            | source | 1, 1, 2, 2, |
            *--------*-------------*
            | target | 1, 2, 1, 2, |
            *--------*-------------*
        """
        srcs = self.source
        trgt = self.target

        if isinstance(srcs, int):
            srcs = [srcs]
        if isinstance(trgt, int):
            trgt = [trgt]

        # 35 is arbitrarily chosen.
        if len(srcs) < 35:
            source = '| source | ' + ''.join(str(e)+', ' for e in srcs) + '|'
            target = '| target | ' + ''.join(str(e)+', ' for e in trgt) + '|'
        else:
            source = ('| source | ' + ''.join(str(e)+', ' for e in srcs[:15]) +
                      '... ' + ''.join(str(e)+', ' for e in srcs[-15:]) + '|')
            target = ('| target | ' + ''.join(str(e)+', ' for e in trgt[:15]) +
                      '... ' + ''.join(str(e)+', ' for e in trgt[-15:]) + '|')

        borderline_s = '*--------*' + '-'*(len(source) - 12) + '-*'
        borderline_t = '*--------*' + '-'*(len(target) - 12) + '-*'
        borderline_m = max(borderline_s, borderline_t)

        result = (borderline_s + '\n' + source + '\n' + borderline_m + '\n' +
                  target + '\n' + borderline_t)
        return result

    def sources(self):
        """Returns iterator containing the source node IDs of the `SynapseCollection`."""
        if not isinstance(self.sources, (list, tuple)):
            self.sources = (self.sources,)
        return iter(self.sources)

    def targets(self):
        """Returns iterator containing the target node IDs of the `SynapseCollection`."""
        if not isinstance(self.targets, (list, tuple)):
            self.targets = (self.targets,)
        return iter(self.targets)

    def get(self, keys=None, output=''):
        """
        Return a parameter dictionary of the connections.

        If `keys` is a string, a list of values is returned, unless we have a
        single connection, in which case the single value is returned.
        `keys` may also be a list, in which case a dictionary with a list of
        values is returned.

        Parameters
        ----------
        keys : str or list, optional
            String or a list of strings naming model properties. get
            then returns a single value or a dictionary with lists of values
            belonging to the given `keys`.
        output : str, ['pandas','json'], optional
            If the returned data should be in a Pandas DataFrame or in a
            JSON serializable format.

        Returns
        -------
        dict:
            All parameters, or, if keys is a list of strings, a dictionary with
            lists of corresponding parameters
        type:
            If keys is a string, the corrsponding parameter(s) is returned


        Raises
        ------
        TypeError
            If input params are of the wrong form.
        KeyError
            If the specified parameter does not exist for the connections.
        """
        pandas_output = output == 'pandas'

        # Return empty tuple if we have no connections or if we have done a
        # nest.ResetKernel()
        if self.nest_instance:
            num_conn = self.nest_instance.GetKernelStatus('num_connections')
        else:
            num_conn = 0
        if self.__len__() == 0 or num_conn == 0:
            return ()

        if keys is None:
            keys = self._attributes
        elif isinstance(keys, string_types):
            keys = keys.split(" ")
        elif not is_iterable(keys):
            raise TypeError("keys should be either a string or an iterable")

        result = self.todict(keys)

        if pandas_output:
            index = (self.source if self.__len__() > 1 else
                     (self.source,))
            result = pandas.DataFrame(result, index=index)
        elif output == 'json':
            result = to_json(result)

        return result

    def set(self, params=None, **kwargs):
        """
        Set the parameters of the connections to `params`.

        NB! This is almost the same implementation as SetStatus

        If `kwargs` is given, it has to be names and values of an attribute as keyword argument pairs. The values
        can be single values or list of the same size as the `SynapseCollection`.

        Parameters
        ----------
        params : str or dict or list
            Dictionary of parameters or list of dictionaries of parameters of
            same length as the `SynapseCollection`.
        kwargs : keyword argument pairs
            Named arguments of parameters of the elements in the `SynapseCollection`.

        Raises
        ------
        TypeError
            If input params are of the wrong form.
        KeyError
            If the specified parameter does not exist for the connections.
        """
        if self.nest_instance:
            num_conn = self.nest_instance.GetKernelStatus('num_connections')
            if num_conn == 0:
                return
            self.nest_instance.set(self, params=params, **kwargs)
            self.fromdict(self.nest_instance.get())


class RayNESTClient(object):

    run_task_ref_obj = None

    def __init__(self, nest_server):
        self.nest_server = nest_server

    def __getstate__(self):
        return {"nest_server": self.nest_server, "_run_ref": self.run_task_ref_obj}

    def __setstate__(self, d):
        self.nest_server = d.get("nest_server", None)
        self.run_task_ref_obj = d.get("_run_ref", None)

    def _gids(self, gids):
        return tuple(ensure_list(gids))

    def _node_collection_to_gids(self, node_collection):
        return node_collection.gids

    def NodeCollection(self, gids):
        return RayNodeCollection(self, self._gids(gids))

    def SynapseCollection(self, conns_dict):
        return RaySynapseCollection(self, conns_dict)

    def _synapse_collection_to_source_target_dict(self, synapse_collection):
        return {"source": synapse_collection.source, "target": synapse_collection.target,
                "synapse_model": synapse_collection.synapse_model}

    def get(self, nodes,  *params, **kwargs):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.get.remote(
                    self._node_collection_to_gids(nodes),  *params, **kwargs))
        else:
            return ray.get(
                self.nest_server.get.remote(
                    self._synapse_collection_to_source_target_dict(nodes),  *params, **kwargs))

    def set(self, nodes, params=None, **kwargs):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.set.remote(
                    self._node_collection_to_gids(nodes), params, **kwargs))
        else:
            return ray.get(
                self.nest_server.set.remote(
                    self._synapse_collection_to_source_target_dict(nodes), params, **kwargs))

    def GetStatus(self, nodes, attrs=None, output=None):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.GetStatus.remote(
                    self._node_collection_to_gids(nodes), attrs, output))
        else:
            return ray.get(
                self.nest_server.GetStatus.remote(
                    self._synapse_collection_to_source_target_dict(nodes), attrs, output))

    def SetStatus(self, nodes, params, val=None):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.SetStatus.remote(
                    self._node_collection_to_gids(nodes), params, val))
        else:
            return ray.get(
                self.nest_server.SetStatus.remote(
                    self._synapse_collection_to_source_target_dict(nodes), params, val))

    def Create(self, model, n=1, params=None, positions=None):
        return self.NodeCollection(
            ray.get(self.nest_server.Create.remote(model, n=n, params=params, positions=positions)))

    def Connect(self, pre, post, conn_spec=None, syn_spec=None, return_synapsecollection=False):
        if return_synapsecollection:
            return self.SynapseCollection(ray.get(
                self.nest_server.Connect.remote(self._node_collection_to_gids(pre),
                                                self._node_collection_to_gids(post),
                                                conn_spec=conn_spec, syn_spec=syn_spec,
                                                return_synapsecollection=True)))
        else:
            return ray.get(
                self.nest_server.Connect.remote(self._node_collection_to_gids(pre),
                                                self._node_collection_to_gids(post),
                                                conn_spec=conn_spec, syn_spec=syn_spec,
                                                return_synapsecollection=False))

    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse'):
        return ray.get(
            self.nest_server.Disconnect.remote(self._node_collection_to_gids(pre),
                                               self._node_collection_to_gids(post),
                                               conn_spec=conn_spec, syn_spec=syn_spec))

    def GetLocalNodeCollection(self, node_collection):
        if len(node_collection):
            return self.NodeCollection(
                self.nest_server.GetLocalNodeCollection.remote(
                    self._node_collection_to_gids(node_collection)))
        else:
            return self.NodeCollection(())

    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None):
        if source is not None:
            source = self._node_collection_to_gids(source)
        if target is not None:
            target = self._node_collection_to_gids(target)
        return self.SynapseCollection(
            ray.get(self.nest_server.GetConnections.remote(source=source, target=target,
                                                           synapse_model=synapse_model,
                                                           synapse_label=synapse_label)))

    def GetNodes(self, properties={}, local_only=False):
        return self.NodeCollection(
            self.nest_server.GetNodes.remote(properties=properties, local_only=local_only))

    def Models(self):
        return ray.get(self.nest_server.nest.remote("Models"))

    def GetDefaults(self, model, keys=None, output=''):
        return ray.get(self.nest_server.nest.remote("GetDefaults", model, keys=keys, output=output))

    def SetDefaults(self, model, params, val=None):
        return ray.get(self.nest_server.nest.remote("SetDefaults", model, params, val=val))

    def CopyModel(self, existing, new, params=None):
        return ray.get(self.nest_server.CopyModel.remote((existing, new, params)))

    def ConnectionRules(self):
        return ray.get(self.nest_server.nest.remote("ConnectionRules"))

    def DisableStructuralPlasticity(self):
        return ray.get(self.nest_server.nest.remote("DisableStructuralPlasticity"))

    def EnableStructuralPlasticity(self):
        return ray.get(self.nest_server.nest.remote("EnableStructuralPlasticity"))

    def ResetKernel(self):
        return ray.get(self.nest_server.nest.remote("ResetKernel"))

    def GetKernelStatus(self, *args):
        return ray.get(self.nest_server.nest.remote("GetKernelStatus", *args))

    def SetKernelStatus(self, values_dict):
        return ray.get(self.nest_server.nest.remote("SetKernelStatus", values_dict))

    def set_verbosity(self, level):
        return ray.get(self.nest_server.nest.remote("set_verbosity", level))

    def get_verbosity(self):
        return ray.get(self.nest_server.nest.remote("get_verbosity"))

    def sysinfo(self):
        return ray.get(self.nest_server.nest.remote("sysinfo"))

    def help(self, obj=None, return_text=True):
        # TODO: a warning for when return_text = False
        if isinstance(obj, RayNodeCollection):
            return ray.get(self.nest_server.help(obj=self._node_collection_to_gids(obj), return_text=True))
        elif isinstance(obj, RaySynapseCollection):
            return ray.get(self.nest_server.help(obj=self._synapse_collection_to_source_target_dict(obj),
                                                 return_text=True))
        else:
            return ray.get(self.nest_server.help(obj=obj, return_text=True))

    def Install(self, module_name):
        return ray.get(self.nest_server.nest.remote("Install", module_name))

    def PrintNodes(self):
        return ray.get(self.nest_server.nest.remote("PrintNodes"))

    def authors(self):
        return ray.get(self.nest_server.nest.remote("authors"))

    def get_argv(self):
        return ray.get(self.nest_server.nest.remote("get_argv"))

    @property
    def is_running(self):
        if self.run_task_ref_obj is None:
            return False
        else:
            done, running = ray.wait([self.run_task_ref_obj], timeout=0)
            if len(running):
                return True
            else:
                return False

    @property
    def block_run(self):
        if self.is_running:
            ray.get(self.run_task_ref_obj)
        self.run_task_ref_obj = None
        return self.run_task_ref_obj

    def _run(self, method, time):
        if not self.is_running:
            if method.lower() == "simulate":
                method = "Simulate"
            else:
                method = "Run"
            run_task_ref_obj = None
            while run_task_ref_obj is None:
                run_task_ref_obj = self.nest_server.nest.remote(method, time)
            self.run_task_ref_obj = run_task_ref_obj
        return self.run_task_ref_obj

    def Prepare(self):
        return ray.get(self.nest_server.nest.remote("Prepare"))

    def Run(self, time):
        return self._run("Run", time)

    def RunLock(self, time, ref_objs=[]):
        if len(ref_objs):
            ray.get(ref_objs)
        return self._run("Run", time)

    def Simulate(self, time):
        return self._run("Simulate", time)

    def Cleanup(self):
        return ray.get(self.nest_server.nest.remote("Cleanup"))
