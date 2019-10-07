/*
 *  tvb_rate_ampa_gaba_wongwang.h
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

#ifndef TVB_RATE_AMPA_GABA_WONGWANG_H
#define TVB_RATE_AMPA_GABA_WONGWANG_H

#include "config.h"

// C++ includes:
#include <string>

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "node.h"
#include "normal_randomdev.h"
#include "poisson_randomdev.h"
#include "ring_buffer.h"
#include "recordables_map.h"
#include "universal_data_logger.h"

namespace tvbnest
{

/** @BeginDocumentation
Name: tvb_rate_ampa_gaba_wongwang - rate model implementing the TVB version of an AMPA or GABA Wong-Wang neuron

Description:

tvb_rate_ampa_gaba_wongwang is an implementation of a nonlinear rate model with equations:
dS_i/dt = -(1/tau_syn) * S_i + SUM_k{delta(t-t_k}, where t_k is the time of a spike emitted by neuron i

The spike is emitted when the membrane voltage V_m >= V_th, in which case it is reset to V_reset,
and kept there for refractory time t_ref.

The V_m dynamics is given by the following equations:

dV_m_i/dt = 1/C_m *( -g_m * (V_m_i - E_L)
                     -g_AMPA_ext * (V_m_i - E_ex))*SUM_j_in_AMPA_ext{w_ij*S_j}  // input from external AMPA neurons
                     -g_AMPA_rec * (V_m_i - E_ex))*SUM_j_in_AMPA_rec{w_ij*S_j} // input from recursive AMPA neurons
                     -g_NMDA / (1 + lambda_NMDA * exp(-beta*V_m_i)) * (V_m_i - E_ex))*SUM_j_in_NMDA{w_ij*S_j} // input from recursive NMDA neurons
                     -g_GABA * (V_m_i - E_in))*SUM_j_in_GAB{w_ij*S_j} // input from recursive GABA neurons

The model supports connections to other identical models with either zero or
non-zero delay, and uses the secondary_event concept introduced with
the gap-junction framework.

Parameters:

The following parameters can be set in the status dictionary.

Default parameter values follow reference [3]
  The following parameters can be set in the status dictionary.
  C_m [pF]  200 for Inh, Capacity of the membrane
  g_m [nS]  20nS for Inh, Membrane leak conductance
  E_L [mV]  Resting potential.
  V_th [mV]  Threshold
  V_reset [mV]  Reset value of the membrane potential
  t_ref [ms]  1ms for Inh, Refractory period.
  g_AMPA_ext [nS]  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
  g_AMPA_rec [nS]  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
  g_NMDA [nS]  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
  g_GABA [nS]  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
  E_ex [mV]  Excitatory reversal potential
  E_in [mV]  Inhibitory reversal potential
  tau_syn [ms]  2.0ms for AMPA and 10.0 ms for GABA synapse decay time
  tau_AMPA [ms]  AMPA synapse decay time
  tau_NMDA_rise [ms]  NMDA synapse rise time
  tau_NMDA_decay [ms]  NMDA synapse decay time
  tau_GABA [ms]  GABA synapse decay time
  beta [real]
  lamda_NMDA [real]
  I_e [pA]  External current.
  rectify_output        bool - Flag to constrain synaptic gating variable (S) in the interval [0, 1].
                             true (default): If the S < 0 it is set to S = 0.0 after each time step.
                                             If the S > 1 it is set to S = 1.0 after each time step.
                             false : No constraint.
    consistent_integration bool - Flag to select integrator.
                              true (default): Exponential Euler integrator.
                              false: Euler - Maruyama integrator.

  Dynamic state variables:
  r [integer]  counts number of tick during the refractory period

  Initial values:
  V_m [mV]  membrane potential

References:

[1] Kong-Fatt Wong and Xiao-Jing Wang, A Recurrent Network
    Mechanism of Time Integration in Perceptual Decisions.
    Journal of Neuroscience 26(4), 1314-1328, 2006.

[2] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
    Romani, Patric Hagmann and Maurizio Corbetta. Resting-State
    Functional Connectivity Emerges from Structurally and
    Dynamically Shaped Slow Linear Fluctuations. The Journal of
    Neuroscience 32(27), 11239-11252, 2013.

[3] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann, Gian Luca
    Romani,  Dante Mantini, and Maurizio Corbetta. How Local
    Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics.
    The Journal of Neuroscience 34(23), 7886-7898, 2014.

[4] Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
    Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
    The Virtual Brain: a simulator of primate brain network dynamics.
    Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

[5] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A.,
    Bolten, M., Helias, M. and Diesmann, M. (2017).
    Integration of Continuous-Time Dynamics in a
    Spiking Neural Network Simulator.
    Front. Neuroinform. 11:34. doi: 10.3389/fninf.2017.00034

[6] Hahne, J., Helias, M., Kunkel, S., Igarashi, J.,
    Bolten, M., Frommer, A. and Diesmann, M. (2015).
    A unified framework for spiking and gap-junction interactions
    in distributed neuronal network simulations.
    Front. Neuroinform. 9:22. doi: 10.3389/fninf.2015.00022

Sends: InstantaneousRateConnectionEvent, DelayedRateConnectionEvent

Receives: InstantaneousRateConnectionEvent, DelayedRateConnectionEvent,
DataLoggingRequest

Author: Dionysios Perdikis, following previous code of Mario Senden, Jan Hahne, Jannis Schuecker

SeeAlso: rate_connection_instantaneous, rate_connection_delayed, lin_rate, tanh_rate, threshold_lin_rate
 */
class tvb_rate_ampa_gaba_wongwang : public nest::Archiving_Node
{

public:
  typedef nest::Node base;

  tvb_rate_ampa_gaba_wongwang();
  tvb_rate_ampa_gaba_wongwang( const tvb_rate_ampa_gaba_wongwang& );

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using nest::Node::handle;
  using nest::Node::sends_secondary_event;

  void handle( nest::InstantaneousRateConnectionEvent& );
  void handle( nest::DelayedRateConnectionEvent& );
  void handle( nest::DataLoggingRequest& );

  nest::port handles_test_event( nest::InstantaneousRateConnectionEvent&, nest::rport );
  nest::port handles_test_event( nest::DelayedRateConnectionEvent&, nest::rport );
  nest::port handles_test_event( nest::DataLoggingRequest&, nest::rport );

  void
  sends_secondary_event( nest::InstantaneousRateConnectionEvent& )
  {
  }
  void
  sends_secondary_event( nest::DelayedRateConnectionEvent& )
  {
  }

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:

/**
  * Synapse types to connect to
  * @note Excluded upper and lower bounds are defined as INF_, SUP_.
  *       Excluding port 0 avoids accidental connections.
  */
  enum SynapseTypes
    {
      INF_RECEPTOR = 0,
      AMPA_EXT ,
      AMPA_REC ,
      NMDA ,
      GABA ,
      SUP_RECEPTOR
    };

  void init_state_( const nest::Node& proto );
  void init_buffers_();
  void calibrate();

  /** This is the actual update function. The additional boolean parameter
   * determines if the function is called by update (false) or wfr_update (true)
   */
  bool update_( nest::Time const&, const long, const long, const bool );

  void update( nest::Time const&, const long, const long );
  bool wfr_update( nest::Time const&, const long, const long );

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< tvb_rate_ampa_gaba_wongwang >;
  friend class nest::UniversalDataLogger< tvb_rate_ampa_gaba_wongwang >;

  // ----------------------------------------------------------------

  /**
   * Independent parameters of the model.
   */
  struct Parameters_
  {
    //!  200 for Inh, Capacity of the membrane
    double C_m_;

    //!  Threshold
    double V_th_;

    //!  Reset value of the membrane potential
    double V_reset_;

    //!  Resting potential.
    double E_L_;

    //!  Excitatory reversal potential
    double E_ex_;

    //!  Inhibitory reversal potential
    double E_in_;

    //!  1ms for Inh, Refractory period.
    double t_ref_;

     //!  2.0ms for AMPA and 10.0 ms for GABA synapse decay time
    double tau_syn_;

    //!  AMPA synapse decay time
    double tau_AMPA_;

    //!  NMDA synapse rise time
    double tau_NMDA_rise_;

    //!  NMDA synapse decay time
    double tau_NMDA_decay_;

    //!  GABA synapse decay time
    double tau_GABA_;

    //!  20nS for Inh, Membrane leak conductance
    double g_m_;

    //!  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
    double g_AMPA_ext_;

    //!  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
    double g_AMPA_rec_;

    //!  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
    double g_NMDA_;

    //!  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
    double g_GABA_;

    //!
    double beta_;

    //!
    double lamda_NMDA_;

    //!  External current.
    double I_e_;

    //!  Spike amplitude.
    double spike_amplitude_;

    /** Noise parameters. */
    double sigma_;
    double sigma_S_;
    double sigma_V_m_;

    /** Should the synaptic gating variable (S) be constrained in the interval [0, 1]?.
        true (default): If the S < 0 it is set to 0.0 - S after each time step.
                        If the S > 1 it is set to 1.0 - S after each time step.
        false : No constraint.
    **/
    bool rectify_output_;

    /** Flag to select integrator.
        true (default): Exponential Euler integrator.
        false: Euler - Maruyama integrator.
    **/
    bool consistent_integration_;

    Parameters_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary

    void set( const DictionaryDatum& );
  };

  // ----------------------------------------------------------------

  /**
   * State variables of the model.
   */
  struct State_
  {
    double S_;  //!< Synaptic gating variable restricted in [0, 1]
    double noise_S_;  //!< Noise for S

    long r; // Refractory time steps' counter

    double V_m_; // Membrane voltage
    double noise_V_m_;  //!< Noise for V_m

    double spike_; // Spike amplitude

    double s_AMPA_ext_; // Total external AMPA synaptic input

    double s_AMPA_rec_; // Total recursive AMPA synaptic input

    double s_NMDA_; // Total recursive NMDA synaptic input

    double s_GABA_; // Total recursive GABA synaptic input

    State_(); //!< Default initialization

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum& );
  };

  // ----------------------------------------------------------------

  /**
   * Buffers of the model.
   */
  struct Buffers_
  {
    Buffers_( tvb_rate_ampa_gaba_wongwang& );
    Buffers_( const Buffers_&, tvb_rate_ampa_gaba_wongwang& );

    std::vector<long> receptor_types_;

    std::vector< nest::RingBuffer delayed_S_inputs(SUP_RECEPTOR-1);
    inline nest::RingBuffer& get_delayed_S_AMPA_ext() {  return delayed_S_inputs[AMPA_EXT - 1]; }
    inline nest::RingBuffer& get_delayed_S_AMPA_rec() {  return delayed_S_inputs[AMPA_REC - 1]; }
    inline nest::RingBuffer& get_delayed_S_NMDA() {  return delayed_S_inputs[NMDA - 1]; }
    inline nest::RingBuffer& get_delayed_S_GABA() {  return delayed_S_inputs[GABA - 1]; }

    std::vector<std::vector< double >>
      instant_S_inputs_(SUP_RECEPTOR-1);
    inline std::vector< double >& get_instant_S_AMPA_ext() {  return instant_S_inputs_[AMPA_EXT - 1]; }
    inline std::vector< double >& get_instant_S_AMPA_rec() {  return instant_S_inputs_[AMPA_REC - 1]; }
    inline std::vector< double >& get_instant_S_NMDA() {  return instant_S_inputs_[NMDA - 1]; }
    inline std::vector< double >& get_instant_S_GABA() {  return instant_S_inputs_[GABA - 1]; }

    // by RateConnectionInstantaneous from inhibitory neurons
    std::vector< double >
      last_S_values; //!< remembers y_values from last wfr_update
     std::vector< double >
      last_V_m_values; //!< remembers y_values from last wfr_update
    std::vector<std::vector< double >> random_numbers(2); //!< remembers the random_numbers in
    // order to apply the same random
    // numbers in each iteration when wfr
    // is used
    nest::UniversalDataLogger< tvb_rate_ampa_gaba_wongwang >
      logger_; //!< Logger for all analog data
  };

  inline nest::RingBuffer& get_delayed_S_AMPA_ext() {return B_.get_delayed_S_AMPA_ext();};
  inline nest::RingBuffer& get_delayed_S_AMPA_rec() {return B_.get_delayed_S_AMPA_rec();};
  inline nest::RingBuffer& get_delayed_S_NMDA() {return B_.get_delayed_S_NMDA();};
  inline nest::RingBuffer& get_delayed_S_GABA() {return B_.get_delayed_S_GABA();};

  inline nest::RingBuffer& get_instant_S_AMPA_ext() {return B_.get_instant_S_AMPA_ext();};
  inline nest::RingBuffer& get_instant_S_AMPA_rec() {return B_.get_instant_S_AMPA_rec();};
  inline nest::RingBuffer& get_instant_S_NMDA() {return B_.get_instant_S_NMDA();};
  inline nest::RingBuffer& get_instant_S_GABA() {return B_.get_instant_S_GABA();};

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {

    // propagators
    std::vector<double> P1_(2);
    std::vector<double> P2_(2);
    double P2_spike_;
    double g_m_E_L_;

    // propagator for noise
    std::vector<double> input_noise_factor_(2);

    librandom::RngPtr rng_;
    librandom::PoissonRandomDev poisson_dev_; //!< random deviate generator
    librandom::NormalRandomDev normal_dev_;   //!< random deviate generator

    //!  refractory time in steps
    long RefractoryCounts;

    // Effective noise strength
    std::vector<double> sigma_(2);

  };
  //! Read out the synaptic gating variable
  inline double get_S_() const
  {
    return S_.S_;
  }

  //! Read out the S noise
  inline double get_noise_S_() const
  {
    return S_.noise_S_;
  }

  inline long get_r() const {
    return S_.r;
  }
  inline void set_r(const long __v) {
    S_.r = __v;
  }

  inline double get_V_m_() const
  {
    return S_.V_m_;
  }
  //! Read out the V_m noise
  inline double get_noise_V_m_() const
  {
    return S_.noise_V_m_;
  }

  inline double get_spike_() const
  {
    return S_.spike_;
  }

  inline double get_s_AMPA_ext() const
  {
    return S_.s_AMPA_ext;
  }

  inline double get_s_AMPA_rec() const
  {
    return S_.s_AMPA_rec;
  }

  inline double get_s_NMDA() const
  {
    return S_.s_NMDA;
  }

  inline double get_s_GABA() const
  {
    return S_.s_GABA;
  }

  inline double get_I_leak() const {
    return P_.g_m * (S_.V_m_ - P_.E_L);
  }

  inline double get_I_AMPA_ext() const {
    return P_.g_AMPA_ext * (S_.V_m_ - P_.E_ex) * S_.s_AMPA_ext];
  }

  inline double get_I_AMPA_rec() const {
    return P_.g_AMPA_rec * (S_.V_m_ - P_.E_ex) * S_.s_AMPA_rec;
  }

  inline double get_I_NMDA() const {
    return P_.g_NMDA * (S_.V_m_ - P_.E_ex)/(1 + P_.lambda_NMDA * exp(-P_.beta_ * S_.V_m_) * S_.s_NMDA;
  }

  inline double get_I_GABA() const {
    return P_.g_GABA * (S_.V_m_ - P_.E_in) * S_.s_GABA;
  }

  inline long get_RefractoryCounts() const {
    return V_.RefractoryCounts;
  }
  inline void set_RefractoryCounts(const long __v) {
    V_.RefractoryCounts = __v;
  }

  // ----------------------------------------------------------------

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< tvb_rate_ampa_gaba_wongwang > recordablesMap_;
};

inline void
tvb_rate_ampa_gaba_wongwang::update( nest::Time const& origin, const long from, const long to )
{
  update_( origin, from, to, false );
}

inline bool
tvb_rate_ampa_gaba_wongwang::wfr_update( nest::Time const& origin, const long from, const long to )
{
  State_ old_state = S_; // save state before wfr update
  const bool wfr_tol_exceeded = update_( origin, from, to, true );
  S_ = old_state; // restore old state

  return not wfr_tol_exceeded;
}

inline nest::port
tvb_rate_ampa_gaba_wongwang::handles_test_event(
  nest::InstantaneousRateConnectionEvent&,
  nest::rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline nest::port
tvb_rate_ampa_gaba_wongwang::handles_test_event(
  nest::DelayedRateConnectionEvent&,
  nest::rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline nest::port
tvb_rate_ampa_gaba_wongwang::handles_test_event(
  nest::DataLoggingRequest& dlr,
  nest::rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}


inline void
tvb_rate_ampa_gaba_wongwang::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  nest::Archiving_Node::get_status( d );
  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
tvb_rate_ampa_gaba_wongwang::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d );         // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  nest::Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace

#endif /* #ifndef tvb_rate_ampa_gaba_wongwang_H */
