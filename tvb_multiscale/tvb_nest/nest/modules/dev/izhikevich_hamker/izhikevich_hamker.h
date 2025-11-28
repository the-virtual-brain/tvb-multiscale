/*
 *  izhikevich_hamker.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef IZHIKEVICH_HAMKER_H
#define IZHIKEVICH_HAMKER_H

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

namespace nest
{

/** @BeginDocumentation
@ingroup Neurons
@ingroup iaf

Name: izhikevich_hamker - izhikevich_hamker neuron model

Description:
Implementation of the simple spiking neuron model introduced by Izhikevich
[1] and modified in [2]. The default parameters' values follow the CortexI population of [3].
The dynamics are given by:
  @f[
  dv/dt = n2*v^2 + n1*v + n0 - u/C_m + I_e + I - g_AMPA*(v-E_AMPA) - g_GABA_A*(v-E_GABA) - g_L*v \\
  du/dt = a*(b*(v-Vr) - u)]
  tau_rise_AMPA*dg_AMPA/dt = -g_AMPA + spikes_exc(t)      (positively weighted spikes at port 1)
  tau_rise_GABA_A*dg_GABA/dt = -g_GABA_A + spikes_inh(t)  (negatively weighted spikes at port 1)
  tau_rise*dg_L/dt = -g_L + spikes_baseline(t)            (only positively -error otherwise- weighted spikes at port 0)
  @f]

    if  \f$ v >= V_{th} \f$:
      v is set to c
      u is incremented by d

    conductances jump on each spike arrival by the weight of the spike.

As published in [1], the numerics differs from the standard forward Euler
technique in two ways:
1) the new value of u is calculated based on the new value of v, rather than
the previous value
2) the variable v is updated using a time step half the size of that used to
update variable u.

This model offers both forms of integration, they can be selected using the
boolean parameter consistent_integration. To reproduce some results published
on the basis of this model, it is necessary to use the published form of the
dynamics. In this case, consistent_integration must be set to false. For all
other purposes, it is recommended to use the standard technique for forward
Euler integration. In this case, consistent_integration must be set to true
(default).


Parameters:
The following parameters can be set in the status dictionary.

\verbatim embed:rst
======================= =======  ==============================================
 V_m                    mV       Membrane potential
 U_m                    mV       Membrane potential recovery variable
 E_rev_AMPA             mV       AMPA reversal potential
 E_rev_GABA_A           mV       GABA reversal potential
 V_th                   mV       Spike threshold
 V_min                  mV       Absolute lower value for the membrane potential
 V_r                    mV       Reversal potential threshold
 C_m                    real     Membrane capacitance
 g_L                    nS       Baseline conductance variable
 g_AMPA                 nS       AMPA conductance variable
 g_GABA_A               nS       GABA conductance variable
 I_syn_ex               pA       AMPA synaptic current
 I_syn_in               pA       GABA synaptic current
 I_syn                  pA       Total synaptic current
 I_e                    pA       Constant input current (R=1)
 t_ref                  ms       Duration of refractory period in ms.
 tau_rise               ms       Time constant of baseline conductance
 tau_rise_AMPA          ms       Time constant of AMPA synapse
 tau_rise_GABA_A        ms       Time constant of GABA synapse
 n0                     real     Constant coefficient of V_m
 n1                     real     Linear coefficient of V_m
 n2                     real     Square coefficient of V_m
 a                      real     Describes time scale of recovery variable
 b                      real     Sensitivity of recovery variable
 c                      mV       After-spike reset value of V_m
 d                      mV       After-spike reset value of U_m
 current_stimulus_scale real     Float coefficient scaling input current stimuli.
 current_stimulus_mode  long     Transformation of input current stimuli:
                                  If 0 (default), do nothing
                                  if 1, rectification, e.g., std::abs(current),
                                  if 2, square pulse current, 1 if current > 0.0 else 0.0
 consistent_integration boolean  Use standard integration technique
======================= =======  ==============================================
\endverbatim

References:

\verbatim embed:rst
.. [1] Izhikevich EM (2003). Simple model of spiking neurons. IEEE Transactions on
       Neural Networks, 14:1569-1572.
       DOI: https://doi.org/10.1109/TNN.2003.820440
   [2] Baladron J, Nambu A, & Hamker FH (2019). The subthalamic nucleus-external globus pallidus loop
       biases exploratory decisions towards known alternatives: a neuro-computational study.
       European Journal of Neuroscience, 49:754-767.
       DOI: https://doi.org/10.1111/ejn.13666
   [3] Maith O, Escudero V, Dinkelbach HU,Baladron J, Horn A, Irmen F, Kuhn, AA & Hamker FH (2020).
       A computational model‐based analysis of basal ganglia pathway changes in Parkinson’s disease
       inferred from resting‐state fMRI. European Journal of Neuroscience, 00:1-18.
       DOI: https://doi.org/10.1111/ejn.14868
\endverbatim

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

FirstVersion: 2020

Author: Hanuschkin, Morrison, Kunkel and Perdikis, D., for the modification

SeeAlso: izhikevich
*/
class izhikevich_hamker : public ArchivingNode
{

public:
  izhikevich_hamker();
  izhikevich_hamker( const izhikevich_hamker& );

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using Node::handle;
  using Node::handles_test_event;

  void handle( DataLoggingRequest& );
  void handle( SpikeEvent& );
  void handle( CurrentEvent& );

  size_t handles_test_event( DataLoggingRequest&, size_t );
  size_t handles_test_event( SpikeEvent&, size_t );
  size_t handles_test_event( CurrentEvent&, size_t );

  size_t send_test_event( Node&, size_t, synindex, bool );

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  friend class RecordablesMap< izhikevich_hamker >;
  friend class UniversalDataLogger< izhikevich_hamker >;

  void init_state_( const Node& proto );
  void init_buffers_();
  void pre_run_hook() override;

  void update( Time const&, const long, const long );

  /**
   * Minimal spike receptor type.
   * @note Start with 1 so we can forbid port 0 to avoid accidental
   *       creation of connections with no receptor type set.
   */
  static const size_t MIN_SPIKE_RECEPTOR = 0;

  /**
   * Spike receptors.
   */
  enum SpikeSynapseTypes
  {
    ACTIVITY = MIN_SPIKE_RECEPTOR,
    NOISE,
    SUP_SPIKE_RECEPTOR
  };

  static const size_t NUM_SPIKE_RECEPTORS = SUP_SPIKE_RECEPTOR - MIN_SPIKE_RECEPTOR;

  // ----------------------------------------------------------------

  /**
   * Independent parameters of the model.
   */
  struct Parameters_
  {
    /** Membrane potential */
    /** Threshold */
    double E_rev_AMPA_;
    double E_rev_GABA_A_;
    double V_th_;
    /** Lower bound */
    double V_min_;
    /** Reversal potential threshold */
    double V_r_;

     /** Membrane capacitance */
    double C_m_;

    /** External DC current */
    double I_e_;

    /** Synaptic time constants */
    double t_ref_;
    double tau_rise_;
    double tau_rise_AMPA_;
    double tau_rise_GABA_A_;

    double n0_;
    double n1_;
    double n2_;
    double a_;
    double b_;
    double c_;
    double d_;

    /** Current stimuli **/
    double current_stimulus_scale_;
    long current_stimulus_mode_;

    /** Use standard integration numerics **/
    bool consistent_integration_;

    Parameters_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
    void set( const DictionaryDatum& ); //!< Set values from dictionary
  };

  // ----------------------------------------------------------------

  /**
   * State variables of the model.
   */
  struct State_
  {
    double v_;         // membrane potential
    double u_;         // membrane recovery variable
    double g_L_;       // Membrane conductance
    double g_AMPA_;    // AMPA conductance
    double g_GABA_A_;  // GABA conductance
    double I_syn_ex_;  // total AMPA synaptic current
    double I_syn_in_;  // total GABA synaptic current
    double I_syn_;     // total synaptic current
    double I_;         // input current

    int r_;            // number of refractory steps remaining

    /** Accumulate spikes arriving during refractory period, discounted for
        decay until end of refractory period.
    */

    State_(); //!< Default initialization

    void get( DictionaryDatum&, const Parameters_& ) const;
    void set( const DictionaryDatum&, const Parameters_& );
  };

  // ----------------------------------------------------------------

  /**
   * Buffers of the model.
   */
  struct Buffers_
  {
    /**
     * Buffer for recording
     */
    Buffers_( izhikevich_hamker& );
    Buffers_( const Buffers_&, izhikevich_hamker& );
    UniversalDataLogger< izhikevich_hamker > logger_;

    /** buffers and sums up incoming spikes/currents */
    RingBuffer spikes_exc_;
    RingBuffer spikes_inh_;
    RingBuffer spikes_base_;
    RingBuffer currents_;
  };

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {
    unsigned int refractory_counts_;
  };

  // Access functions for UniversalDataLogger -----------------------

  //! Read out the membrane potential
  double
  get_V_m_() const
  {
    return S_.v_;
  }
  //! Read out the recovery variable
  double
  get_U_m_() const
  {
    return S_.u_;
  }
  //! Read out the g_L variable
  double
  get_g_L_() const
  {
    return S_.g_L_;
  }
  //! Read out the g_AMPA variable
  double
  get_g_AMPA_() const
  {
    return S_.g_AMPA_;
  }
  //! Read out the g_GABA_A variable
  double
  get_g_GABA_A_() const
  {
    return S_.g_GABA_A_;
  }
  //! Read out the Ι variable
  double
  get_I_syn_() const
  {
    return S_.I_syn_;
  }
  //! Read out the Ι variable
  double
  get_I_syn_ex_() const
  {
    return S_.I_syn_ex_;
  }
  //! Read out the Ι variable
  double
  get_I_syn_in_() const
  {
    return S_.I_syn_in_;
  }
  //! Read out the Ι variable
  double
  get_I_() const
  {
    return S_.I_;
  }

  // ----------------------------------------------------------------

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  //! Mapping of recordables names to access functions
  static RecordablesMap< izhikevich_hamker > recordablesMap_;
  /** @} */
};

inline size_t
izhikevich_hamker::send_test_event( Node& target, size_t receptor_type, synindex, bool )
{
  SpikeEvent e;
  e.set_sender( *this );

  return target.handles_test_event( e, receptor_type );
}

inline size_t
izhikevich_hamker::handles_test_event( SpikeEvent&, size_t receptor_type )
{
  if ( receptor_type < MIN_SPIKE_RECEPTOR || receptor_type > SUP_SPIKE_RECEPTOR )
  {
    throw IncompatibleReceptorType( receptor_type, get_name(), "SpikeEvent" );
  }
  return receptor_type - MIN_SPIKE_RECEPTOR;
}

inline size_t
izhikevich_hamker::handles_test_event( CurrentEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline size_t
izhikevich_hamker::handles_test_event( DataLoggingRequest& dlr, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
izhikevich_hamker::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d, P_ );
  ArchivingNode::get_status( d );
  ( *d )[ names::recordables ] = recordablesMap_.get_list();

  /**
   * @todo dictionary construction should be done only once for
   * static member in default c'tor, but this leads to
   * a seg fault on exit, see #328
   */
  DictionaryDatum receptor_dict_ = new Dictionary();
  ( *receptor_dict_ )[ names::activity ] = ACTIVITY;
  ( *receptor_dict_ )[ names::noise ] = NOISE;

  ( *d )[ names::receptor_types ] = receptor_dict_;
}

inline void
izhikevich_hamker::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  ArchivingNode::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace nest

#endif /* #ifndef IZHIKEVICH_HAMKER_H */
