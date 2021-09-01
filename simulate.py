import os

import numpy as np
import matplotlib.pyplot as plt

import brian2 as br2
from brian2 import ms, um, cm, mV, nS, siemens, ufarad, amp
from brian2.core.functions import timestep


def run(inp_spikes, scat, weight=.2*nS, VT=-62 * mV, duration=100 * ms):
    """Run a simulation of a granule cell.

    Parameters
    ----------
    inp_spikes: List of arrays*ms, the differebt input spike times
    weight: br2.nS, the weight of synapses
    sat: Boolean, determine if the gs saturates or not
    scatter: Boolean, determine if inputs are scattered or not
    VT : br2.Quantity, sett the threshold of the LIF model

    Returns
    -------
    state_mon: `StateMonitor` of the soma"""
    net = br2.Network()

    # Set the input spiketimes
    indices = []
    for i, time in enumerate(inp_spikes):
        indices += [i for _ in range(len(time))]
    inp = br2.SpikeGeneratorGroup(len(inp_spikes), indices,
                                  [t for spikes in inp_spikes for t in spikes])
    net.add(inp)

    # Set model's morphology
    morpho = br2.Soma(10*um)
    d_l = 10*um

    # Set its biophysical parameters
    eqs = '''
    Im = gL*(EL-v) : amp/meter**2
    Is = gs1*(Es-v) + gs2*(Es-v): amp (point current)
    dgs1/dt = -gs1/taus : siemens
    dgs2/dt = -gs2/taus : siemens
    '''
    parameters = {'Cm': 1 * ufarad * cm ** -2,
                  'gL': 5e-5 * siemens * cm ** -2,
                  'EL': -65 * mV,
                  'VT': VT,
                  # synaptic parameters
                  'Es': 0 * mV,
                  'taus': 1 * ms
                  }
    threshold = 'v > VT'
    # The reset makes the membrane potential jump *up*, as a crude
    # approximation of an action potential (that can then propagate to the
    # connected compartments). It will be reset to the resting potential
    # via a delayed synapse (see below)
    neuron = br2.SpatialNeuron(morphology=morpho, model=eqs,
                               method='exact',
                               threshold=threshold,
                               refractory=threshold,
                               threshold_location=0,  # soma
                               reset='v = -65*mV',
                               namespace=parameters,
                               name='neuron')
    neuron.v = 'EL'
    net.add(neuron)

    # Set the connections
    syn1 = br2.Synapses(inp, neuron, 'w : siemens (constant)', on_pre='gs1 = gs1 + w ; gs1 = clip(gs1, 0*nS, 0.1*nS)', name='syn1')
    syn1.connect(i=0, j=morpho)
    syn1.w = weight/2
    net.add(syn1)

    if scat:
        syn2 = br2.Synapses(inp, neuron, 'w : siemens (constant)', on_pre='gs2 = gs2 + w ; gs2 = clip(gs2, 0*nS, 0.1*nS)', name='syn2')
        syn2.connect(i=0, j=morpho)
        syn2.w = weight/2
        net.add(syn2)
        syn1.w = weight/2

    else:
        syn1.w = weight

    net.add(syn1)

    # Record the voltage and the spikes
    mon = br2.StateMonitor(neuron, 'v', record=True)
    spike_mon = br2.SpikeMonitor(neuron)
    net.add(mon, spike_mon)
    net.run(duration, namespace={})
    return mon, spike_mon


def adjust_spines(ax, spines):
    """Remove the spines from a matplotlib graphics.
    taken from matplotlib gallery anonymous author.

    parameters
    ----------
    ax: a matplolib axes object
        handler of the object to work with
    spines: list of char
        location of the spines

    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            pass
        else:
            spine.set_color('none')
            # don't draw spine
            # turn off ticks where there is no spine

    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


if __name__ == "__main__":
    # Prepare the system for storing the Fig
    figdir = "../Figs/"
    if not os.path.exists(figdir):
        os.mkdir(figdir)

    #Running the simulation
    n_spikes = []
    inp = [np.array([15*i for i in range(10)])*ms]
    colors = ['black', 'grey']
    fig, ax = plt.subplots(figsize=(3.5, 2))
    for i, c_sat in enumerate([True, False]):
        mon, spike_mon = run(inp, c_sat)
        n_spikes += [len(spike_mon)]
        ax.plot(mon.t/ms, mon.v[0]/mV, color=colors[i], lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    adjust_spines(ax, ["left", "bottom"])
    plt.tight_layout()
    plt.savefig(figdir + "Fig1E.png",dpi=300)
    plt.close

    n_spikes = 2*[n_spikes[1]*10, n_spikes[1]*10] + 2*[n_spikes[0]*10, n_spikes[1]*10]
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.bar([0.1, 0.9, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9], n_spikes, color=colors*4)
    adjust_spines(ax, ["left", "bottom"])
    ax.set_ylim(10,90)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Out. Frequency')
    plt.savefig(figdir + "Fig1D.png",dpi=300)
