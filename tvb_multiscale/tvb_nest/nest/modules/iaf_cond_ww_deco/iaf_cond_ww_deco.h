/*
 *  iaf_cond_ww_deco.h
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

#ifndef IAF_COND_WW_DECO_H
#define IAF_COND_WW_DECO_H

// Generated includes:
#include "config.h"
#include <sstream>

#ifdef HAVE_GSL

// External includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

namespace nest
{
/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL. Internally, it is
 *       a first-class C++ function, but cannot be a member function
 *       because of the C-linkage.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 * @param void* Pointer to model neuron instance.
 */
extern "C" int
iaf_cond_ww_deco_dynamics( double, const double*, double*, void* );

/** @BeginDocumentation
Name: iaf_cond_ww_deco - Conductance based adaptive exponential
                                     integrate-and-fire neuron model according
                                     to Brette and Gerstner (2005) with
                                     multiple synaptic rise time and decay
                                     time constants, and synaptic conductance
                                     modeled by an alpha function.

Description:

iaf_cond_ww_deco is a conductance-based adaptive exponential
integrate-and-fire neuron model. It allows an arbitrary number of synaptic
time constants. Synaptic conductance is modeled by an alpha function, as
described by A. Roth and M.C.W. van Rossum in Computational Modeling Methods
for Neuroscientists, MIT Press 2013, Chapter 6.

The time constants are supplied by an array, "tau_syn", and the pertaining
synaptic reversal potentials are supplied by the array "E_rev". Port numbers
are automatically assigned in the range from 1 to n_receptors.
During connection, the ports are selected with the property "receptor_type".

The membrane potential is given by the following differential equation:

C dV/dt = -g_L(V-E_L) + g_L*Delta_T*exp((V-V_T)/Delta_T) + I_syn_tot(V, t)
          - w + I_e

where

I_syn_tot(V,t) = \sum_i g_i(t) (V - E_{rev,i}) ,

the synapse i is excitatory or inhibitory depending on the value of E_{rev,i}
and the differential equation for the spike-adaptation current w is:

tau_w * dw/dt = a(V - E_L) - w

When the neuron fires a spike, the adaptation current w <- w + b.

Parameters:

The following parameters can be set in the status dictionary.

Dynamic state variables:
  V_m        double - Membrane potential in mV
  w          double - Spike-adaptation current in pA.

Membrane Parameters:
  C_m        double - Capacity of the membrane in pF
  t_ref      double - Duration of refractory period in ms.
  V_reset    double - Reset value for V_m after a spike. In mV.
  E_L        double - Leak reversal potential in mV.
  g_L        double - Leak conductance in nS.
  I_e        double - Constant external input current in pA.
  Delta_T    double - Slope factor in mV
  V_th       double - Spike initiation threshold in mV
  V_peak     double - Spike detection threshold in mV.

Adaptation parameters:
  a          double - Subthreshold adaptation in nS.
  b          double - Spike-triggered adaptation in pA.
  tau_w      double - Adaptation time constant in ms

Synaptic parameters
  E_rev      double vector - Reversal potential in mV.
  tau_syn    double vector - Time constant of synaptic conductance in ms

Integration parameters
  gsl_error_tol  double - This parameter controls the admissible error of the
                          GSL integrator. Reduce it if NEST complains about
                          numerical instabilities.


Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

Author: Hans Ekkehard Plesser, based on aeif_cond_beta_multisynapse

SeeAlso: iaf_cond_ww_deco
*/
class iaf_cond_ww_deco : public ArchivingNode
{

public:
  iaf_cond_ww_deco();
  iaf_cond_ww_deco( const iaf_cond_ww_deco& );
  virtual ~iaf_cond_ww_deco();

  friend int iaf_cond_ww_deco_dynamics( double,
    const double*,
    double*,
    void* );

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using Node::handle;
  using Node::handles_test_event;

  port send_test_event( Node&, rport, synindex, bool );

  void handle( SpikeEvent& );
  void handle( CurrentEvent& );
  void handle( DataLoggingRequest& );

  port handles_test_event( SpikeEvent&, rport );
  port handles_test_event( CurrentEvent&, rport );
  port handles_test_event( DataLoggingRequest&, rport );

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  void init_state_( const Node& proto );
  void init_buffers_();
  void calibrate();
  void update( Time const&, const long, const long );

  // The next three classes need to be friends to access the State_ class/member
  friend class DynamicRecordablesMap< iaf_cond_ww_deco >;
  friend class DynamicUniversalDataLogger< iaf_cond_ww_deco >;
  friend class DataAccessFunctor< iaf_cond_ww_deco >;

  // ----------------------------------------------------------------

  /**
   * Independent parameters of the model.
   */
  struct Parameters_
  {
    //!  Threshold
    double V_th;

    //!  Reset value of the membrane potential
    double V_reset;

    //!  Resting potential.
    double E_L;

    //!  Excitatory reversal potential
    double E_ex;

    //!  Inhibitory reversal potential
    double E_in;

    //!  1ms for Inh, Refractory period.
    double t_ref;

    //!  AMPA synapse decay time
    double tau_decay_AMPA;

    //!  NMDA synapse rise time
    double tau_rise_NMDA;

    //!  NMDA synapse decay time
    double tau_decay_NMDA;

    //!  GABA synapse decay time
    double tau_decay_GABA_A;

    //!  200 for Inh, Capacity of the membrane
    double C_m;

    //!  20nS for Inh, Membrane leak conductance
    double g_L;

    //!  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
    double g_AMPA_ext;

    //!  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
    double g_AMPA;

    //!  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
    double g_NMDA;

    //!  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
    double g_GABA_A;

    //!  Excitatory synapse weight, initialized as w_EE = 1.55, it should be set by the user as w_IE = 1.0, for inhibitory neurons
    double w_E;

    //!  Inhibitory synapse weight, initialized as w_II = 1.0, it should be set by the user as w_EI = J_i, for excitatory neurons
    double w_I;

    //!  External excitatory synapses' weights
    std::vector< double >  w_E_ext;

    //! Number of excitatory neurons in the population
    //! Also acting as ceiling for s_AMPA and s_NMDA
    int N_E;

    //! Number of inhibitory neurons in the population
    //! Also acting as ceiling for s_GABA
    int N_I;

    //! Ceiling of s_AMPA_ext
    std::vector< double > s_AMPA_ext_max;

    //! Scaling parameter of s_NMDA in the ds_NMDA/dt equation
    double epsilon;

    //!
    double alpha;

    //!
    double beta;

    //!
    double lambda_NMDA;

    //!  External current.
    double I_e;

    double gsl_error_tol; //!< error bound for GSL integrator

    // boolean flag which indicates whether the neuron has connections
    bool has_connections_;

    Parameters_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
    void set( const DictionaryDatum& ); //!< Set values from dictionary

    //! Return the number of receptor ports
    inline size_t
    n_receptors()  const
    {
      return w_E_ext.size();
    }
  };

  // ----------------------------------------------------------------

  /**
   * State variables of the model.
   * @note Copy constructor and assignment operator required because
   *       of C-style arrays.
   */
  struct State_
  {

    /**
     * Enumeration identifying elements in state vector State_::y_.
     * This enum identifies the elements of the vector. It must be public to be
     * accessible from the iteration function. The last two elements of this
     * enum (DG, G) will be repeated
     * n times at the end of the state vector State_::y with n being the number
     * of synapses.
     */
    enum StateVecElems
    {
      V_M = 0,
      // recursive AMPA synaptic gating variable
      S_AMPA,

      // recursive NMDA linear synaptic gating variable
      X_NMDA,

      //  NMDA 2nd order synaptic gating variable
      S_NMDA,

      // recursive GABA synaptic gating variable
      S_GABA,

      // external AMPA synaptic gating variables
      S_AMPA_EXT_0,

      STATE_VECTOR_MIN_SIZE

    };

    static const size_t NUMBER_OF_FIXED_STATES_ELEMENTS = S_AMPA_EXT_0; // 5: V_M, S_AMPA, S_GABA, X_NMDA, S_NMDA

    enum FixedCurrentsElems
    {
      // leak synaptic current
      I_L = NUMBER_OF_FIXED_STATES_ELEMENTS,  // current_index = 5

      // external (stimulation) currents
      I_E, // current_index = 6

      // recursive AMPA synaptic current
      I_AMPA,  // current_index = 7

      // recursive NMDA synaptic current
      I_NMDA, // current_index = 8

      // recursive GABA synaptic current
      I_GABA, // current_index = 9

      FIXED_CURRENTS_ELEMS_END  // 10

    };

    static const size_t NUMBER_OF_FIXED_CURRENTS_ELEMENTS =
        FIXED_CURRENTS_ELEMS_END - NUMBER_OF_FIXED_STATES_ELEMENTS; // 10 -5 = 5

    enum FixedSpikesElems
    {
      // recursive excitatory spikes
      SPIKES_EXC = FIXED_CURRENTS_ELEMS_END,  // spike_index = 10

      // recursive inhibitory spikes
      SPIKES_INH  // spike_index = 11

    };

    static const size_t NUMBER_OF_FIXED_SPIKES_ELEMENTS = 2;

    static const size_t NUMBER_OF_TOTAL_FIXED_ELEMENTS = NUMBER_OF_FIXED_STATES_ELEMENTS +
                                                         NUMBER_OF_FIXED_CURRENTS_ELEMENTS +
                                                         NUMBER_OF_FIXED_SPIKES_ELEMENTS; // = 5 + 5 + 2 = 12

    static const size_t NUMBER_OF_ELEMENTS_PER_RECEPTOR = 3 ;

    enum ReceptorElemsStartIndxs
    {
        // external AMPA synaptic gating variables
        S_AMPA_EXT_START = NUMBER_OF_TOTAL_FIXED_ELEMENTS,  // external_index = 12

        // external AMPA synaptic current
        I_AMPA_EXT_START, // current_index = 13

        // external spikes
        SPIKES_EXC_EXT_START // spike_index = 14
    };

    enum ReceptorElemsTypes
    {
        // external AMPA synaptic gating variables
        S_AMPA_EXT_TYP = 0,

        // external AMPA synaptic current
        I_AMPA_EXT_TYP, // 1

        // external spikes
        SPIKES_EXC_EXT_TYP //  2
    };



    std::vector< double > y_; //!< neuron state
    int r_;                   //!< number of refractory steps remaining

    State_( const Parameters_& ); //!< Default initialization
    State_( const State_& );
    State_& operator=( const State_& );

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum& );

  }; // State_

  // ----------------------------------------------------------------

  /**
   * Buffers of the model.
   */
  struct Buffers_
  {
    Buffers_( iaf_cond_ww_deco& );
    Buffers_( const Buffers_&, iaf_cond_ww_deco& );

    //! Logger for all analog data
    DynamicUniversalDataLogger< iaf_cond_ww_deco > logger_;

    //!< Buffer incoming nSs through delay, as sum
    std::vector< RingBuffer > spikesExc_ext;

    //!< Buffer incoming nSs through delay, as sum
    RingBuffer spikesExc;

    //!< Buffer incoming nSs through delay, as sum
    RingBuffer spikesInh;

    //!< Buffer incoming pAs through delay, as sum
    RingBuffer currents;

    /** GSL ODE stuff */
    gsl_odeiv_step* s_;    //!< stepping function
    gsl_odeiv_control* c_; //!< adaptive stepsize control function
    gsl_odeiv_evolve* e_;  //!< evolution function
    gsl_odeiv_system sys_; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double step_;            //!< simulation step size in ms
    double IntegrationStep_; //!< current integration time step,
                             //!< updated by solver

    /**
     * Input current injected by CurrentEvent.
     * This variable is used to transport the current applied into the
     * _dynamics function computing the derivative of the state vector.
     * It must be a part of Buffers_, since it is initialized once before
     * the first simulation, but not modified before later Simulate calls.
     */

    double I_e;

    //!< neuron spikes received:
    double spikes_exc;
    double spikes_inh;
    std::vector< double > spikes_exc_ext;

  };

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {

    unsigned int refractory_counts_;

    // Some internal variables to speed up computations:

    double g_L_E_L;  // g_L * E_L in pA

    double minus_beta;

    double minus_tau_decay_AMPA;  // - tau_decay_AMPA in 1/ms

    double minus_tau_decay_GABA_A;  // - tau_decay_GABA_A in 1/ms

    double minus_tau_rise_NMDA;  // - tau_rise_NMDA in ms

    double minus_tau_decay_NMDA;  // - tau_decay_NMDA in ms

    double w_E_g_AMPA;  // w_E * g_AMPA

    double w_E_N_E_g_NMDA;  // w_E * N_E *g_NMDA

    double alpha_N_E;   // alpha / N_E

    double w_I_g_GABA_A;  // w_I * g_GABA

    std::vector< double > w_E_ext_g_AMPA_ext;  // w_E_ext * g_AMPA_ext

    double s_EXC_max;   // maximum boundary of s_AMPA and s_NMDA, to be set equal to N_E
    double s_INH_max;   // maximum boundary of s_GABA, to be set equal to N_I

  };

  // Data members -----------------------------------------------------------

  /**
   * @defgroup iaf_cond_ww_deco
   * Instances of private data structures for the different types
   * of data pertaining to the model.
   * @note The order of definitions is important for speed.
   * @{
   */
  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;
  /** @} */

  // Access functions for UniversalDataLogger -------------------------------

  //! Mapping of recordables names to access functions
  DynamicRecordablesMap< iaf_cond_ww_deco > recordablesMap_;

  // Data Access Functor getter
  DataAccessFunctor< iaf_cond_ww_deco > get_data_access_functor(
    size_t elem );

  // Utility function that inserts the S_AMPA_ext to the
  // recordables map

  Name get_s_receptor_name( size_t receptor );
  Name get_I_receptor_name( size_t receptor );
  Name get_spikes_receptor_name( size_t receptor );
  void insert_external_recordables( size_t first = 0 );

  inline size_t recordables_end() {
    return State_::NUMBER_OF_TOTAL_FIXED_ELEMENTS + State_::NUMBER_OF_ELEMENTS_PER_RECEPTOR * P_.n_receptors();
  }

  inline size_t elem_S_AMPA_EXT(size_t receptor) {
    return State_::S_AMPA_EXT_START + receptor * State_::NUMBER_OF_ELEMENTS_PER_RECEPTOR ;
  }

  inline size_t elem_I_AMPA_EXT(size_t receptor) {
    return State_::I_AMPA_EXT_START + receptor * State_::NUMBER_OF_ELEMENTS_PER_RECEPTOR ;
  }


  inline size_t elem_SPIKES_EXC_EXT(size_t receptor) {
    return State_::SPIKES_EXC_EXT_START + receptor * State_::NUMBER_OF_ELEMENTS_PER_RECEPTOR ;
  }

  inline double get_I_L() const {
    return P_.g_L * S_.y_[ State_::V_M ] - V_.g_L_E_L;
  }

  inline double get_I_e() const {
    return P_.I_e + B_.I_e;
  }

  inline double get_I_AMPA() const {
    return V_.w_E_g_AMPA * (S_.y_[ State_::V_M ] - P_.E_ex) * S_.y_[State_::S_AMPA];
  }

  inline double get_I_NMDA() const {
    return V_.w_E_N_E_g_NMDA /
                ( 1 + P_.lambda_NMDA * std::exp(V_.minus_beta * S_.y_[ State_::V_M ] ) ) *
                (S_.y_[ State_::V_M ] - P_.E_ex) * S_.y_[State_::S_NMDA];
  }

  inline double get_I_GABA() const {
    return V_.w_I_g_GABA_A * (S_.y_[ State_::V_M ] - P_.E_in) * S_.y_[State_::S_GABA];
  }

  inline double get_I_AMPA_ext( size_t receptor ) const {
    return V_.w_E_ext_g_AMPA_ext[ receptor ] * (S_.y_[ State_::V_M ] - P_.E_ex) * S_.y_[State_::S_AMPA_EXT_0 + receptor];
  }

  inline double get_spikes_exc() {
    double temp = B_.spikes_exc;
    B_.spikes_exc = 0.0;
    return temp;
  }

  inline double get_spikes_inh() {
    double temp = B_.spikes_inh;
    B_.spikes_inh = 0.0;
    return temp;
  }

  inline double get_spikes_exc_ext( size_t receptor ) {
    double temp = B_.spikes_exc_ext[ receptor ];
    B_.spikes_exc_ext[ receptor ] = 0.0;
    return temp;
  }

  inline double
  get_state_element( size_t elem )
  {

    if (elem < State_::NUMBER_OF_FIXED_STATES_ELEMENTS)
    {
        return S_.y_[ elem ];
    }
    else if (elem < State_::SPIKES_EXC)
    {
        switch(elem) {
            case State_::I_L: return get_I_L() ;
            case State_::I_E: return get_I_e() ;
            case State_::I_AMPA: return get_I_AMPA() ;
            case State_::I_NMDA: return get_I_NMDA() ;
            case State_::I_GABA: return get_I_GABA() ;
            default: throw nest::BadProperty("Non existent current index!");
        }
    }
    else if (elem < State_::NUMBER_OF_TOTAL_FIXED_ELEMENTS)
    {
        switch(elem) {
            case State_::SPIKES_EXC: return get_spikes_exc() ;
            case State_::SPIKES_INH: return get_spikes_inh() ;
            default: throw nest::BadProperty("Non existent spikes' current index!");
        }
    }
    else if (elem < recordables_end())
    {
        size_t receptor_elem = elem - State_::NUMBER_OF_TOTAL_FIXED_ELEMENTS ;
        size_t elem_type = receptor_elem % State_::NUMBER_OF_ELEMENTS_PER_RECEPTOR ;
        size_t receptor = receptor_elem / State_::NUMBER_OF_ELEMENTS_PER_RECEPTOR ;
        switch(elem_type) {
            case State_::S_AMPA_EXT_TYP: return S_.y_[ State_::S_AMPA_EXT_0 + receptor ] ;
            case State_::I_AMPA_EXT_TYP: return get_I_AMPA_ext( receptor ) ;
            case State_::SPIKES_EXC_EXT_TYP: return get_spikes_exc_ext( receptor ) ;
            default: throw nest::BadProperty("Non existent receptor element type!");
        }
    }
    else
    {
        // TODO: Throw the correct exception here
        throw nest::BadProperty("Too large element index!");
    }
  };

};

inline port
iaf_cond_ww_deco::send_test_event( Node& target,
  rport receptor_type,
  synindex,
  bool )
{
  SpikeEvent e;
  e.set_sender( *this );

  return target.handles_test_event( e, receptor_type );
}

inline port
iaf_cond_ww_deco::handles_test_event( SpikeEvent&,
  rport receptor_type )

{
  if ( receptor_type < 0
    || receptor_type > static_cast< port >(P_.n_receptors() ) )
  {
    throw IncompatibleReceptorType( receptor_type, get_name(), "SpikeEvent" );
  }

  P_.has_connections_ = true;
  return receptor_type;
}

inline port
iaf_cond_ww_deco::handles_test_event( CurrentEvent&,
  rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline port
iaf_cond_ww_deco::handles_test_event( DataLoggingRequest& dlr,
  rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
iaf_cond_ww_deco::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  ArchivingNode::get_status( d );

  ( *d )[ names::recordables ] = recordablesMap_.get_list();
}

} // namespace

#endif // HAVE_GSL
#endif // IAF_COND_WW_DECO_H //
