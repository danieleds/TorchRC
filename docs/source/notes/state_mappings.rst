State mappings
==============

Unlike other Reservoir Computing libraries, in TorchRC the reservoirs and the readouts
are independent modules that can be *composed*. This allows for high flexibility, since
you can 1) swap the readout with anything (even deep networks trained by gradient descent),
and 2) you can perform any kind of transformation on the states before feeding them to the
readout.

In this note we'll show some techniques that you can use by exploiting the flexibility of
TorchRC to implement custom state mappings from the reservoir to the readout.

Last state mapping
------------------

The *last state mapping* is the typical configuration for Reservoir Computing models: the
last state of a sequence is used to produce a prediction.

.. code-block:: python

    h, _ = reservoir(x)
    out = readout(h[-1])


Mean state mapping
------------------

With a *mean state mapping*, the average of the states over all time steps is used to produce
a prediction. In this case, the readout has access to a more explicit view of the whole dynamics
of the states, in a tensor of fixed size.

.. code-block:: python

    h, _ = reservoir(x)
    out = readout(h.mean(dim=0))


Higher-dimensional projection
-----------------------------

Sometimes you may want to project your reservoir states into a higher-dimensional space to improve
linear separability. As before, we can modify the output states before feeding them to the readout.

.. code-block:: python

    h, _ = reservoir(x)
    hw = h[-1] @ projection_matrix
    out = readout(hw)

