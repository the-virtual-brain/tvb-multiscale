/*
 *  tvb_rate_redwongwang_inh.h
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

#ifndef TVB_RATE_REDWONGWANG_INH_H
#define TVB_RATE_REDWONGWANG_INH_H

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
Name: tvb_rate_redwongwang_inh - rate model implementing the TVB version of a reduced Wong-Wang model
of an inhibitory population .

Description:

tvb_rate_redwongwang_inh is an implementation of a nonlinear rate model with equations:
dS_i/dt = -(1/tau) * S_i + g*r_i

where S_i is a synaptic gating variable
and r_i = H(I_syn_i, a, b, d) is the firing rate,

where H(I_syn_i, a, b, d) is a sigmoidal function of the input synaptic currents I_syn_i:
H(I_syn, a, b, d) = (a*I_syn-b) / (1-exp(-d*(a*I_syn-b)))

and  I_syn_i = W_I*Io + w_rec*J_i*S_i + CouplingInput
(i.e., the input transformation is applied to individual inputs).

The model supports connections to other identical models with either zero or
non-zero delay, and uses the secondary_event concept introduced with
the gap-junction framework.

Parameters:

The following parameters can be set in the status dictionary.

Default parameter values follow reference [3]
g                   double - kinetic parameter in s (??; default: 1.0/1000)
tau                 double - Time constant of rate dynamics in ms (default: 10ms (GABA)).
w_rec               double - local synaptic recurrence weight (unitless, default: 1.0)
W_I                 double - external synaptic  weight (unitless, default: 0.7)
a                   double - sigmoidal function parameter in nC^-1 (default: 615nC^-1)
b                   double - sigmoidal function parameter in Hz (default: 177 Hz)
d                   double - sigmoidal function parameter in sec (??; default: 0.087 s)
J_i                 double - synaptic inhibitory GABA coupling current in nA (default: 1.0 nA)
Io                  double - overall effective external input current in nA (default: 0.382 nA)
I_e                 double - external current (e.g., from stimulation) (default: 0.0 nA)
sigma               double - Noise parameter in nA (??; default: 0.01).
rectify_output        bool - Flag to constrain synaptic gating variable (S) in the interval [0, 1].
                             true (default): If the S < 0 it is set to S = 0.0 at each time step.
                                             If the S > 1 it is set to S = 1.0 at each time step.
                             false : No constraint.
consistent_integration bool - Flag to select integrator.
                              true (default): Exponential Euler integrator.
                              false: Euler - Maruyama integrator.

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
class tvb_rate_redwongwang_inh : public nest::Archiving_Node
{

public:
  typedef nest::Node base;

  tvb_rate_redwongwang_inh();
  tvb_rate_redwongwang_inh( const tvb_rate_redwongwang_inh& );

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
  void init_state_( const nest::Node& proto );
  void init_buffers_();
  void calibrate();

  /** This is the actual update function. The additional boolean parameter
   * determines if the function is called by update (false) or wfr_update (true)
   */
  bool update_( nest::Time const&, const long, const long, const bool );

  void update( nest::Time const&, const long, const long );
  bool wfr_update( nest::Time const&, const long, const long );

  // sigmoid helper function
  double sigmoid( double );

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< tvb_rate_redwongwang_inh >;
  friend class nest::UniversalDataLogger< tvb_rate_redwongwang_inh >;

  // ----------------------------------------------------------------

  /**
   * Independent parameters of the model.
   */
  struct Parameters_
  {
    /** Kinetic parameter in s (??). */
    double g_;

    /** Time constant in ms. */
    double tau_;

    /** Local inhibitory synaptic recurrence weight (unitless). */
    double w_rec_;

    /** External synaptic weight (unitless). */
    double W_I_;

    /** Sigmoidal function parameter in nC^-1. */
    double a_;

    /** Sigmoidal function parameter in Hz1. */
    double b_;

    /** Sigmoidal function parameter in s (??). */
    double d_;

    /** Excitatory synaptic coupling current in nA. */
    double J_i_;

    /** Overall effective external input current in nA. */
    double Io_;

    /** External (e.g., stimulus) input current in nA. */
    double I_e_;

    /** Noise parameter. */
    double sigma_;

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
    double noise_;  //!< Noise
    double I_syn_;  // Total synaptic current
    double r_;  // Rate

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
    Buffers_( tvb_rate_redwongwang_inh& );
    Buffers_( const Buffers_&, tvb_rate_redwongwang_inh& );

    nest::RingBuffer delayed_currents_ex_; //!< buffer for current vector received by
    // RateConnectionDelayed from excitatory neurons
    nest::RingBuffer delayed_currents_in_; //!< buffer for current vector received by
    // RateConnectionDelayed from inhibitory neurons
    std::vector< double >
      instant_currents_ex_; //!< buffer for current vector received
    // by RateConnectionInstantaneous from excitatory neurons
    std::vector< double >
      instant_currents_in_; //!< buffer for current vector received
    // by RateConnectionInstantaneous from inhibitory neurons
    std::vector< double >
      last_y_values; //!< remembers y_values from last wfr_update
    std::vector< double > random_numbers; //!< remembers the random_numbers in
    // order to apply the same random
    // numbers in each iteration when wfr
    // is used
    nest::UniversalDataLogger< tvb_rate_redwongwang_inh >
      logger_; //!< Logger for all analog data
  };

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {

    // propagators
    double P1_;
    double P2_;
    double W_I_Io_I_e_;  // W_I * Io + I_e
    double w_rec_J_i_;  // w_rec * J_i

    // propagator for noise
    double input_noise_factor_;

    librandom::RngPtr rng_;
    librandom::PoissonRandomDev poisson_dev_; //!< random deviate generator
    librandom::NormalRandomDev normal_dev_;   //!< random deviate generator
  };

  //! Read out the synaptic gating variable
   inline double
  get_S_() const
  {
    return S_.S_;
  }
  inline void
  set_S_(const double __v) {
    S_.S_ = __v;
  }

  inline double
  get_currents_() const
  {
    return S_.S_ * P_.J_i_;
  }

  //! Read out the noise
  inline double
  get_noise_() const
  {
    return S_.noise_;
  }

  inline double
  get_I_syn_() const
  {
    return S_.I_syn_;
  }
  inline void
  set_I_syn_(const double __v) {
    S_.I_syn_ = __v;
  }

  inline double
  get_r_() const
  {
    return S_.r_;
  }
  inline void
  set_r_(const double __v) {
    S_.r_ = __v;
  }

  // ----------------------------------------------------------------

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< tvb_rate_redwongwang_inh > recordablesMap_;
};

inline void
tvb_rate_redwongwang_inh::update( nest::Time const& origin, const long from, const long to )
{
  update_( origin, from, to, false );
}

inline bool
tvb_rate_redwongwang_inh::wfr_update( nest::Time const& origin, const long from, const long to )
{
  State_ old_state = S_; // save state before wfr update
  const bool wfr_tol_exceeded = update_( origin, from, to, true );
  S_ = old_state; // restore old state

  return not wfr_tol_exceeded;
}

inline nest::port
tvb_rate_redwongwang_inh::handles_test_event(
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
tvb_rate_redwongwang_inh::handles_test_event(
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
tvb_rate_redwongwang_inh::handles_test_event(
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
tvb_rate_redwongwang_inh::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  nest::Archiving_Node::get_status( d );
  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
tvb_rate_redwongwang_inh::set_status( const DictionaryDatum& d )
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

#endif /* #ifndef TVB_RATE_REDWONGWANG_INH_H */
