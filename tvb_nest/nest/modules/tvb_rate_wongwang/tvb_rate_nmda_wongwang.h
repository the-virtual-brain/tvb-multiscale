/*
 *  tvb_rate_nmda_wongwang.h
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

#ifndef TVB_RATE_NMDA_WONGWANG_H
#define TVB_RATE_NMDA_WONGWANG_H

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
Name: tvb_rate_nmda_wongwang - rate model implementing the TVB version of an AMPA or GABA Wong-Wang neuron

Description:

tvb_rate_nmda_wongwang is an implementation of a nonlinear rate model with equations:
tau_rise_NMDA * dx_i/dt = - x_i + spike_amplitude*SUM_k{delta(t-t_k}, where t_k is the time of a spike emitted by neuron i
tau_decay_NMDA * dS_i/dt = - S_i + alpha * x_i * (1 - S_i)

The spike is emitted when the membrane voltage V_m >= V_th, in which case it is reset to V_reset,
and kept there for refractory time t_ref.

The V_m dynamics is given by the following equations:

dV_m_i/dt = 1/C_m *( -g_L * (V_m_i - E_L)
                     -g_AMPA_ext * (V_m_i - E_ex))*SUM_j_in_AMPA_ext{w_ij*S_j}  // input from external AMPA neurons
                     -g_AMPA_rec * (V_m_i - E_ex))*SUM_j_in_AMPA_rec{w_ij*S_j} // input from recursive AMPA neurons
                     -g_NMDA / (1 + lamda_NMDA * exp(-beta*V_m_i)) * (V_m_i - E_ex))*SUM_j_in_NMDA{w_ij*S_j} // input from recursive NMDA neurons
                     -g_GABA * (V_m_i - E_in))*SUM_j_in_GAB{w_ij*S_j} // input from recursive GABA neurons

The model supports connections to other identical models with either zero or
non-zero delay, and uses the secondary_event concept introduced with
the gap-junction framework.

Parameters:

The following parameters can be set in the status dictionary.

Default parameter values follow reference [3]
  The following parameters can be set in the status dictionary.
  V_th [mV]  Threshold
  V_reset [mV]  Reset value of the membrane potential
  E_L [mV]  Resting potential.
  E_ex [mV]  Excitatory reversal potential
  E_in [mV]  Inhibitory reversal potential
  t_ref [ms]  1ms for Inh, Refractory period.
  tau_rise_NMDA [ms]  100.0 ms for NMDA synapse rise time
  tau_decay_NMDA [ms]  2.0 ms for NMDA synapse decay time
  C_m [pF]  200 for Inh, Capacity of the membrane
  g_L [nS]  20nS for Inh, Membrane leak conductance
  g_AMPA_ext [nS]  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
  g_AMPA_rec [nS]  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
  g_NMDA [nS]  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
  g_GABA [nS]  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
  alpha [real] 0.5 KHz
  beta [real]  0.062
  lamda_NMDA [real]  0.28
  I_e [pA]  External current.
  spike_amplitude [real] Amplitude of spike. Default = tau_rise_NMDA = 2.0ms
  rectify_output        bool - Flag to constrain synaptic gating variable (S) in the interval [0, 1].
                             true (default): If the S < 0 it is set to S = 0.0 at each time step.
                                             If the S > 1 it is set to S = 1.0 at each time step.
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
class tvb_rate_nmda_wongwang : public nest::Archiving_Node
{

public:
  typedef nest::Node base;

  tvb_rate_nmda_wongwang();
  tvb_rate_nmda_wongwang( const tvb_rate_nmda_wongwang& );

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
  friend class nest::RecordablesMap< tvb_rate_nmda_wongwang >;
  friend class nest::UniversalDataLogger< tvb_rate_nmda_wongwang >;

  // ----------------------------------------------------------------

  /**
   * Independent parameters of the model.
   */
  struct Parameters_
  {
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

    //!  20.0 ms for NMDA synapse rise time
    double tau_rise_NMDA_;

    //!  100.0 ms for NMDA synapse decay time
    double tau_decay_NMDA_;

    //!  200 for Inh, Capacity of the membrane
    double C_m_;

    //!  20nS for Inh, Membrane leak conductance
    double g_L_;

    //!  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
    double g_AMPA_ext_;

    //!  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
    double g_AMPA_rec_;

    //!  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
    double g_NMDA_;

    //!  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
    double g_GABA_;

    //!
    double alpha_;

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
    double sigma_x_;
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
    double x_;  //!< Synaptic variable
    double noise_x_;  //!< Noise for x

    double S_;  //!< Synaptic gating variable restricted in [0, 1]
    double noise_S_;  //!< Noise for S

    double V_m_; // Membrane voltage
    double noise_V_m_;  //!< Noise for V_m

    double s_AMPA_ext_; // Total external AMPA synaptic input

    double s_AMPA_rec_; // Total recursive AMPA synaptic input

    double s_NMDA_; // Total recursive NMDA synaptic input

    double s_GABA_; // Total recursive GABA synaptic input

    double I_leak_; // Leak current

    double I_AMPA_ext_; // Total current of external AMPA synaptic input

    double I_AMPA_rec_; // Total current of external AMPA synaptic input

    double I_NMDA_; // Total current of external NMDA synaptic input

    double I_GABA_; // Total current of external GABA synaptic input

    double spike_; // Spike amplitude

    long r; // Refractory time steps' counter

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
    Buffers_( tvb_rate_nmda_wongwang& );
    Buffers_( const Buffers_&, tvb_rate_nmda_wongwang& );

   nest::UniversalDataLogger< tvb_rate_nmda_wongwang >
      logger_; //!< Logger for all analog data

    std::vector< nest::RingBuffer >
        delayed_S_inputs_;
    inline nest::RingBuffer& get_delayed_S_AMPA_ext() {  return delayed_S_inputs_[AMPA_EXT - 1]; }
    inline nest::RingBuffer& get_delayed_S_AMPA_rec() {  return delayed_S_inputs_[AMPA_REC - 1]; }
    inline nest::RingBuffer& get_delayed_S_NMDA() {  return delayed_S_inputs_[NMDA - 1]; }
    inline nest::RingBuffer& get_delayed_S_GABA() {  return delayed_S_inputs_[GABA - 1]; }

    std::vector<std::vector< double >>
      instant_S_inputs_;
    inline std::vector< double >& get_instant_S_AMPA_ext() {  return instant_S_inputs_[AMPA_EXT - 1]; }
    inline std::vector< double >& get_instant_S_AMPA_rec() {  return instant_S_inputs_[AMPA_REC - 1]; }
    inline std::vector< double >& get_instant_S_NMDA() {  return instant_S_inputs_[NMDA - 1]; }
    inline std::vector< double >& get_instant_S_GABA() {  return instant_S_inputs_[GABA - 1]; }

    std::vector<long> receptor_types_;

    // by RateConnectionInstantaneous from inhibitory neurons
    std::vector<std::vector< double >>
      last_y_values; //!< remembers y values from last wfr_update
    std::vector<std::vector< double >> random_numbers; //!< remembers the random_numbers in
    // order to apply the same random
    // numbers in each iteration when wfr
    // is used

  };

  inline nest::RingBuffer& get_delayed_S_AMPA_ext() {return B_.get_delayed_S_AMPA_ext();};
  inline nest::RingBuffer& get_delayed_S_AMPA_rec() {return B_.get_delayed_S_AMPA_rec();};
  inline nest::RingBuffer& get_delayed_S_NMDA() {return B_.get_delayed_S_NMDA();};
  inline nest::RingBuffer& get_delayed_S_GABA() {return B_.get_delayed_S_GABA();};

  inline std::vector< double >& get_instant_S_AMPA_ext() {return B_.get_instant_S_AMPA_ext();};
  inline std::vector< double >& get_instant_S_AMPA_rec() {return B_.get_instant_S_AMPA_rec();};
  inline std::vector< double >& get_instant_S_NMDA() {return B_.get_instant_S_NMDA();};
  inline std::vector< double >& get_instant_S_GABA() {return B_.get_instant_S_GABA();};

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {

    // propagators
    std::vector<double> P1_;
    std::vector<double> P2_;
    double g_L_E_L_;  // g_L * E_L
    double alpha_tau_decay_NMDA_;

    // propagator for noise
    std::vector<double> input_noise_factor_;

    librandom::RngPtr rng_;
    librandom::PoissonRandomDev poisson_dev_; //!< random deviate generator
    librandom::NormalRandomDev normal_dev_;   //!< random deviate generator

    //!  refractory time in steps
    long RefractoryCounts;

    // Effective noise strength
    std::vector<double> sigma_;

  };

  inline double get_x_() const
  {
    return S_.x_;
  }
  inline void set_x_(const double __v) {
    S_.x_ = __v;
  }
  inline double get_noise_x_() const
  {
    return S_.noise_x_;
  }
  inline void set_noise_x_(const double __v) {
    S_.noise_x_ = __v;
  }

  //! Read out the synaptic gating variable
  inline double get_S_() const
  {
    return S_.S_;
  }

  inline void set_S_(const double __v) {
    S_.S_ = __v;
  }

  //! Read out the S noise
  inline double get_noise_S_() const
  {
    return S_.noise_S_;
  }
  inline void set_noise_S_(const double __v) {
    S_.noise_S_ = __v;
  }

  inline double get_V_m_() const
  {
    return S_.V_m_;
  }
  inline void set_V_m_(const double __v) {
    S_.V_m_ = __v;
  }

  //! Read out the V_m noise
  inline double get_noise_V_m_() const
  {
    return S_.noise_V_m_;
  }
  inline void set_noise_V_m_(const double __v) {
    S_.noise_V_m_ = __v;
  }

  inline double get_s_AMPA_ext() const
  {
    return S_.s_AMPA_ext_;
  }
  inline void set_s_AMPA_ext_(const double __v) {
    S_.s_AMPA_ext_ = __v;
  }

  inline double get_s_AMPA_rec() const
  {
    return S_.s_AMPA_rec_;
  }
  inline void set_s_AMPA_rec_(const double __v) {
    S_.s_AMPA_rec_ = __v;
  }

  inline double get_s_NMDA() const
  {
    return S_.s_NMDA_;
  }
  inline void set_s_NMDA_(const double __v) {
    S_.s_NMDA_ = __v;
  }

  inline double get_s_GABA() const
  {
    return S_.s_GABA_;
  }
  inline void set_s_GABA_(const double __v) {
    S_.s_GABA_ = __v;
  }

  inline double get_I_leak() const {
    return S_.I_leak_;
  }
  inline void set_I_leak_(const double __v) {
    S_.I_leak_ = __v;
  }

  inline double get_I_AMPA_ext() const {
    return S_.I_AMPA_ext_;
  }
  inline void set_I_AMPA_ext_(const double __v) {
    S_.I_AMPA_ext_ = __v;
  }

  inline double get_I_AMPA_rec() const {
    return S_.I_AMPA_rec_;
  }
  inline void set_I_AMPA_rec_(const double __v) {
    S_.I_AMPA_rec_ = __v;
  }

  inline double get_I_NMDA() const {
    return S_.I_NMDA_;
  }
  inline void set_I_NMDA_(const double __v) {
    S_.I_NMDA_ = __v;
  }

  inline double get_I_GABA() const {
    return S_.I_GABA_;
  }
  inline void set_I_GABA_(const double __v) {
    S_.I_GABA_ = __v;
  }

  inline double get_spike_() const
  {
    return S_.spike_;
  }
  inline void set_spike_(const double __v) {
    S_.spike_ = __v;
  }

  inline long get_r() const {
    return S_.r;
  }
  inline void set_r(const long __v) {
    S_.r = __v;
  }

  inline double get_I_syn_() const
  {
    return S_.I_leak_ + S_.I_AMPA_ext_ + S_.I_AMPA_rec_ + S_.I_NMDA_ + S_.I_GABA_;
  }

  inline long get_RefractoryCounts() const {
    return V_.RefractoryCounts;
  }

  inline void set_RefractoryCounts(const long __v) {
    V_.RefractoryCounts = __v;
  }

  inline std::vector<double> get_sigma_() const {
    return V_.sigma_;
  }

  inline void set_sigma_(const std::vector<double> __v) {
    V_.sigma_ = __v;
  }

  // ----------------------------------------------------------------

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< tvb_rate_nmda_wongwang > recordablesMap_;
};

inline void
tvb_rate_nmda_wongwang::update( nest::Time const& origin, const long from, const long to )
{
  update_( origin, from, to, false );
}

inline bool
tvb_rate_nmda_wongwang::wfr_update( nest::Time const& origin, const long from, const long to )
{
  State_ old_state = S_; // save state before wfr update
  const bool wfr_tol_exceeded = update_( origin, from, to, true );
  S_ = old_state; // restore old state

  return not wfr_tol_exceeded;
}

inline nest::port
tvb_rate_nmda_wongwang::handles_test_event(
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
tvb_rate_nmda_wongwang::handles_test_event(
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
tvb_rate_nmda_wongwang::handles_test_event(
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
tvb_rate_nmda_wongwang::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  nest::Archiving_Node::get_status( d );
  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
tvb_rate_nmda_wongwang::set_status( const DictionaryDatum& d )
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

#endif /* #ifndef tvb_rate_nmda_wongwang_H */
