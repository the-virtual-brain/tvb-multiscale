
/*
*  iaf_cond_deco2014.h
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
*  2019-11-14 14:51:07.199115
*/
#ifndef IAF_COND_DECO2014
#define IAF_COND_DECO2014

#include "config.h"


#ifdef HAVE_GSL

// External includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// forwards the declaration of the function
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
extern "C" inline int iaf_cond_deco2014_dynamics( double, const double y[], double f[], void* pnode );


// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"


// Includes from sli:
#include "dictdatum.h"

/* BeginDocumentation
  Name: iaf_cond_deco2014.

  Description:  
     -*- coding: utf-8 -*-

  Name: iaf_cond_deco2014 - Conductance based leaky integrate-and-fire neuron model
                            with separate excitatory AMPA & NMDA, and inhibitory GABA synapses

  Description:
  iaf_cond_deco2014 is an implementation of a spiking neuron using IAF dynamics with
  conductance-based synapses.

  The spike is emitted when the membrane voltage V_m >= V_th,
  in which case it is reset to V_reset, and kept there for refractory time t_ref.

  The V_m dynamics is given by the following equations:

  dV_m/dt = 1/C_m *( -g_m * (V_m - E_L)
                     -g_AMPA_ext * (V_m - E_ex))*SUM_i{s_AMPA_ext_i)  // input from external AMPA excitatory neurons
                     -g_AMPA * (V_m - E_ex))*s_AMPA          // input from recursive excitatory neurons
                     -g_NMDA / (1 + lambda_NMDA * exp(-beta*V_m)) * (V_m - E_ex))*s_NMDA // input from recursive excitatory neurons
                     -g_GABA * (V_m - E_in))*s_GABA // input from recursive inhibitory neurons

  where
  ds_(AMPA/GABA)/dt = -(1/tau_(AMPA/GABA)) * w_E/I * s_(AMPA/GABA) + w_E/I *SUM_j_in_E/I{SUM_k{delta(t-t_j_k}},
  dx_NMDA/dt = -(1/tau_NMDA_rise) * w_E * x_NMDA + w_E * SUM_j_in_E{SUM_k{delta(t-t_k}},
  where t_j_k is the time of kth spike emitted by jth neuron of the local excitatory (E) or inhibitory (I) population
  ds_NMDA/dt =  -(1/tau_NMDA_decay) * s_NMDA + alpha * x_NMDA * (1 - s_NMDA)

  Similarly:
  ds_AMPA_ext_i/dt = -(1/tau_AMPA_ext) * w_E_ext_i * s_AMPA_ext_i + w_E_ext_i * SUM_j_in_E_ext_i{SUM_k{delta(t-t_j_k}},
  where t_j_k is the time of kth spike emitted by jth neuron of the ith external excitatory (E) AMPA population

  Therefore, port 0 is reserved for local population spikes, either with positive weight (w_E), or negative (w_I),
  whereas external excitatory AMPA spikes from N external populations are received in ports > 0,
  strictly with positive weights.

  Sends: SpikeEvent

  Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

  References:

  [1] Ponce-Alvarez A., Mantini D., Deco G., Corbetta M., Hagmann P., & Romani G. L. (2014).
  How Local Excitation-Inhibition Ratio Impacts the Whole Brain Dynamics. Journal of Neuroscience.
  34(23), 7886-7898. https://doi.org/10.1523/jneurosci.5068-13.2014

  iaf_cond_beta:

  [2] Meffin, H., Burkitt, A. N., & Grayden, D. B. (2004). An analytical
  model for the large, fluctuating synaptic conductance state typical of
  neocortical neurons in vivo. J.  Comput. Neurosci., 16, 159-175.

  [3] Bernander, O., Douglas, R. J., Martin, K. A. C., & Koch, C. (1991).
  Synaptic background activity influences spatiotemporal integration in
  single pyramidal cells.  Proc. Natl. Acad. Sci. USA, 88(24),
  11569-11573.

  [4] Kuhn, Aertsen, Rotter (2004). Neuronal Integration of Synaptic Input in
  the Fluctuation- Driven Regime. Jneurosci 24(10) 2345-2356

  gif_cond_exp_multisynapse:

  [5] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W (2012)
    Parameter extraction and classification of three cortical neuron types
    reveals two distinct adaptation mechanisms. J. Neurophysiol., 107(6),
    1756-1775.

   [6] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015)
   Automated High-Throughput Characterization of Single Neurons by Means of
   Simplified Spiking Models. PLoS Comput. Biol., 11(6), e1004275.

  Author: Dionysios Perdikis

  SeeAlso: iaf_cond_beta, gif_cond_exp_multisynapse



  Parameters:
  The following parameters can be set in the status dictionary.
  V_th [mV]  Threshold
  V_reset [mV]  Reset value of the membrane potential
  E_L [mV]  Resting potential.
  E_ex [mV]  Excitatory reversal potential
  E_in [mV]  Inhibitory reversal potential
  t_ref [ms]  1ms for Inh, Refractory period.
  tau_decay_AMPA [ms]  AMPA synapse decay time
  tau_rise_NMDA [ms]  NMDA synapse rise time
  tau_decay_NMDA [ms]  NMDA synapse decay time
  tau_decay_GABA_A [ms]  GABA synapse decay time
  C_m [pF]  200 for Inh, Capacity of the membrane
  g_L [nS]  20nS for Inh, Membrane leak conductance
  g_AMPA_ext [nS]  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
  g_AMPA [nS]  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
  g_NMDA [nS]  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
  g_GABA_A [nS]  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
  w_E [real]  Excitatory synapse weight, initialized as w_EE = 1.4, it should be set by the user as w_IE = 1.0
  w_I [real]  Inhibitory synapse weight, initialized as w_II = 1.0, it should be set by the user as w_EI
  w_E_ext [real]  External excitatory synapses' weights' mean
  alpha [kHz] 
  beta [real] 
  lambda_NMDA [real] 
  I_e [pA]  External current.
  

  Dynamic state variables:
  r [integer]  counts number of tick during the refractory period
  

  Initial values:
  s_NMDA [nS] nS = 0.0nS
  V_m [mV]  = 0.0mV membrane potential
  

  References: Empty

  Sends: nest::SpikeEvent

  Receives: Spike, Current, DataLoggingRequest
*/
class iaf_cond_deco2014 : public nest::Archiving_Node{
public:
  /**
  * The constructor is only used to create the model prototype in the model manager.
  */
  iaf_cond_deco2014();

  /**
  * The copy constructor is used to create model copies and instances of the model.
  * @node The copy constructor needs to initialize the parameters and the state.
  *       Initialization of buffers and interal variables is deferred to
  *       @c init_buffers_() and @c calibrate().
  */
  iaf_cond_deco2014(const iaf_cond_deco2014 &);

  /**
  * Releases resources.
  */
  ~iaf_cond_deco2014();

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using nest::Node::handles_test_event;
  using nest::Node::handle;

  /**
  * Used to validate that we can send nest::SpikeEvent to desired target:port.
  */
  nest::port send_test_event(nest::Node& target, nest::rport receptor_type, nest::synindex, bool);

  /**
  * @defgroup mynest_handle Functions handling incoming events.
  * We tell nest that we can handle incoming events of various types by
  * defining @c handle() and @c connect_sender() for the given event.
  * @{
  */
  void handle(nest::SpikeEvent &);        //! accept spikes
  void handle(nest::CurrentEvent &);      //! accept input current
  void handle(nest::DataLoggingRequest &);//! allow recording with multimeter

  nest::port handles_test_event(nest::SpikeEvent&, nest::port);
  nest::port handles_test_event(nest::CurrentEvent&, nest::port);
  nest::port handles_test_event(nest::DataLoggingRequest&, nest::port);
  /** @} */

  // SLI communication functions:
  void get_status(DictionaryDatum &) const;
  void set_status(const DictionaryDatum &);

private:
  //! Reset parameters and state of neuron.

  //! Reset state of neuron.
  void init_state_(const Node& proto);

  //! Reset internal buffers of neuron.
  void init_buffers_();

  //! Initialize auxiliary quantities, leave parameters and state untouched.
  void calibrate();

  //! Take neuron through given time interval
  void update(nest::Time const &, const long, const long);

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap<iaf_cond_deco2014>;
  friend class nest::UniversalDataLogger<iaf_cond_deco2014>;

  /**
  * Free parameters of the neuron.
  *
  *
  *
  * These are the parameters that can be set by the user through @c SetStatus.
  * They are initialized from the model prototype when the node is created.
  * Parameters do not change during calls to @c update() and are not reset by
  * @c ResetNetwork.
  *
  * @note Parameters_ need neither copy constructor nor @c operator=(), since
  *       all its members are copied properly by the default copy constructor
  *       and assignment operator. Important:
  *       - If Parameters_ contained @c Time members, you need to define the
  *         assignment operator to recalibrate all members of type @c Time . You
  *         may also want to define the assignment operator.
  *       - If Parameters_ contained members that cannot copy themselves, such
  *         as C-style arrays, you need to define the copy constructor and
  *         assignment operator to copy those members.
  */
  struct Parameters_{

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

    //!  Excitatory synapse weight, initialized as w_EE = 1.4, it should be set by the user as w_IE = 1.0, for inhibitory neurons
    double w_E;

    //!  Inhibitory synapse weight, initialized as w_II = 1.0, it should be set by the user as w_EI = J_i, for excitatory neurons
    double w_I;

    //!  External excitatory synapses' weights
    std::vector< double >  w_E_ext;

    //! 
    double alpha;

    //! 
    double beta;

    //! 
    double lambda_NMDA;

    //!  External current.
    double I_e;

    double __gsl_error_tol;
    /** Initialize parameters to their default values. */

    //! boolean flag which indicates whether the neuron has connections
    bool has_connections_;

    Parameters_();

    //! Return the number of receptor ports
    inline long
    n_receptors() const
    {
      return (long) w_E_ext.size();
    }

  };

  /**
  * Dynamic state of the neuron.
  *
  *
  *
  * These are the state variables that are advanced in time by calls to
  * @c update(). In many models, some or all of them can be set by the user
  * through @c SetStatus. The state variables are initialized from the model
  * prototype when the node is created. State variables are reset by @c ResetNetwork.
  *
  * @note State_ need neither copy constructor nor @c operator=(), since
  *       all its members are copied properly by the default copy constructor
  *       and assignment operator. Important:
  *       - If State_ contained @c Time members, you need to define the
  *         assignment operator to recalibrate all members of type @c Time . You
  *         may also want to define the assignment operator.
  *       - If State_ contained members that cannot copy themselves, such
  *         as C-style arrays, you need to define the copy constructor and
  *         assignment operator to copy those members.
  */
  struct State_{
    //! Symbolic indices to the elements of the state vector y
    enum StateVecElems{
    //  membrane potential
    V_m,

    // recursive AMPA synaptic gating variable
    s_AMPA,

    // recursive GABA synaptic gating variable
    s_GABA,

    // recursive NMDA linear synaptic gating variable
    x_NMDA,

    //  NMDA 2nd order synaptic gating variable
    s_NMDA,

    // external AMPA synaptic gating variables
    s_AMPA_ext,

    STATE_VEC_SIZE
    };

    static const size_t NUMBER_OF_FIXED_STATES_ELEMENTS = 5; //!< V_M, s_AMPA, s_GABA, x_NMDA, s_NMDA
    static const size_t NUM_STATE_ELEMENTS_PER_RECEPTOR = 1; //!< s_AMPA_ext

    //! state vector, must be C-array for GSL solver,
    // but for the moment we define it as std::vector
    std::vector< double > ode_state;

    //!  counts number of ticks during the refractory period
    long r;

    State_( const Parameters_& ); //!< Default initialization
    State_( const State_& );
    State_& operator=( const State_& );
  };

  /**
  * Internal variables of the neuron.
  *
  *
  *
  * These variables must be initialized by @c calibrate, which is called before
  * the first call to @c update() upon each call to @c Simulate.
  * @node Variables_ needs neither constructor, copy constructor or assignment operator,
  *       since it is initialized by @c calibrate(). If Variables_ has members that
  *       cannot destroy themselves, Variables_ will need a destructor.
  */
  struct Variables_ {    

    //!  refractory time in steps
    long RefractoryCounts;

    // Some internal variables to speed up computations:

    double g_L_E_L;  // g_L * E_L in pA

    double minus_beta;

    double minus_w_E_tau_decay_AMPA;  // -w_E / tau_decay_AMPA in 1/ms

    double minus_w_I_tau_decay_GABA_A;  // -w_I / tau_decay_GABA_A in 1/ms

    double minus_w_E_tau_rise_NMDA;  // -w_E / tau_rise_NMDA in 1/ms

    std::vector< double > minus_w_E_ext_tau_decay_AMPA;  // -w_E_ext / tau_decay_AMPA in 1/ms

  };

  /**
    * Buffers of the neuron.
    * Ususally buffers for incoming spikes and data logged for analog recorders.
    * Buffers must be initialized by @c init_buffers_(), which is called before
    * @c calibrate() on the first call to @c Simulate after the start of NEST,
    * ResetKernel or ResetNetwork.
    * @node Buffers_ needs neither constructor, copy constructor or assignment operator,
    *       since it is initialized by @c init_nodes_(). If Buffers_ has members that
    *       cannot destroy themselves, Buffers_ will need a destructor.
    */
  struct Buffers_ {
    Buffers_(iaf_cond_deco2014 &);
    Buffers_(const Buffers_ &, iaf_cond_deco2014 &);

    /** Logger for all analog data */
    nest::UniversalDataLogger<iaf_cond_deco2014> logger_;

    inline std::vector<nest::RingBuffer> & get_spikesExc_ext() {return spikesExc_ext;}
    //!< Buffer incoming nSs through delay, as sum
    std::vector< nest::RingBuffer > spikesExc_ext;
    std::vector<double> spikesExc_ext_grid_sum_;
    
    inline nest::RingBuffer& get_spikesExc() {return spikesExc;}
    //!< Buffer incoming nSs through delay, as sum
    nest::RingBuffer spikesExc;
    double spikesExc_grid_sum_;
    
    inline nest::RingBuffer& get_spikesInh() {return spikesInh;}
    //!< Buffer incoming nSs through delay, as sum
    nest::RingBuffer spikesInh;
    double spikesInh_grid_sum_;
    
    //!< Buffer incoming pAs through delay, as sum
    nest::RingBuffer currents;
    inline nest::RingBuffer& get_currents() {return currents;}
    double currents_grid_sum_;
    /** GSL ODE stuff */
    gsl_odeiv_step* __s;    //!< stepping function
    gsl_odeiv_control* __c; //!< adaptive stepsize control function
    gsl_odeiv_evolve* __e;  //!< evolution function
    gsl_odeiv_system __sys; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double __step;             //!< step size in ms
    double __integration_step; //!< current integration time step, updated by GSL
    };

  inline long get_r() const {
    return S_.r;
  }
  inline void set_r(const long __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Refractory count r must be >= 0." );
    }
    S_.r = __v;
  }

  inline double get_V_m() const {
    return S_.ode_state[State_::V_m];
  }
  inline void set_V_m(const double __v) {
    S_.ode_state[State_::V_m] = __v;
  }

  inline double get_s_AMPA() const {
    return S_.ode_state[State_::s_AMPA];
  }
  inline void set_s_AMPA(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Synaptic gating variable s_AMPA must be >= 0." );
    }
    S_.ode_state[State_::s_AMPA] = __v;
  }

  inline double get_s_GABA() const {
    return S_.ode_state[State_::s_GABA];
  }
  inline void set_s_GABA(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Synaptic gating variable s_GABA must be >= 0." );
    }
    S_.ode_state[State_::s_GABA] = __v;
  }

  inline double get_x_NMDA() const {
    return S_.ode_state[State_::x_NMDA];
  }
  inline void set_x_NMDA(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Synaptic gating variable x_NMDA must be >= 0." );
    }
    S_.ode_state[State_::x_NMDA] = __v;
  }

  inline double get_s_NMDA() const {
    return S_.ode_state[State_::s_NMDA];
  }
  inline void set_s_NMDA(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Synaptic gating variable s_NMDA must be >= 0." );
    }
    S_.ode_state[State_::s_NMDA] = __v;
  }

  inline double get_s_AMPA_ext_sum() const {
    const std::vector< double > s_AMPA_ext;
    return std::accumulate(s_AMPA_ext.begin(), s_AMPA_ext.end(), 0.0);
    // const std::vector< double > s_AMPA_ext = get_s_AMPA_ext();
    // double s_AMPA_ext_sum = 0.0;
    // for ( long i = 0 ; i < P_.n_receptors() ; ++i ) {
    //    s_AMPA_ext_sum += S_.ode_state[State_::s_AMPA_ext + i];
    //}
    //return s_AMPA_ext_sum;
  }

  inline std::vector< double > get_s_AMPA_ext() const {
    std::vector< double > s_AMPA_ext;
    s_AMPA_ext.resize(P_.n_receptors(), 0.0);
    for ( long i = 0; i < P_.n_receptors() ; ++i ) {
        s_AMPA_ext[i]= S_.ode_state[State_::s_AMPA_ext + i];
    }
    return s_AMPA_ext;
  }

  inline void set_s_AMPA_ext(const std::vector< double > __v) {
    for ( long i = 0 ; i < P_.n_receptors() ; ++i ) {
        if ( __v[i] < 0 )
        {
            throw nest::BadProperty( "Synaptic gating variable s_AMPA_ext must be >= 0." );
        } else
        {
            S_.ode_state[State_::s_AMPA_ext + i] = __v[i];
        }
    }
  }

  inline double get_V_th() const {
    return P_.V_th;
  }
  inline void set_V_th(const double __v) {
    P_.V_th = __v;
  }

  inline double get_V_reset() const {
    return P_.V_reset;
  }
  inline void set_V_reset(const double __v) {
    P_.V_reset = __v;
  }

  inline double get_E_L() const {
    return P_.E_L;
  }
  inline void set_E_L(const double __v) {
    P_.E_L = __v;
  }

  inline double get_E_ex() const {
    return P_.E_ex;
  }
  inline void set_E_ex(const double __v) {
    P_.E_ex = __v;
  }

  inline double get_E_in() const {
    return P_.E_in;
  }
  inline void set_E_in(const double __v) {
    P_.E_in = __v;
  }

  inline double get_t_ref() const {
    return P_.t_ref;
  }
  inline void set_t_ref(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Time constant t_ref must be > 0." );
    }
    P_.t_ref = __v;
  }

  inline double get_tau_decay_AMPA() const {
    return P_.tau_decay_AMPA;
  }
  inline void set_tau_decay_AMPA(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Time constant tau_decay_AMPA must be > 0." );
    }
    P_.tau_decay_AMPA = __v;
  }

  inline double get_tau_rise_NMDA() const {
    return P_.tau_rise_NMDA;
  }
  inline void set_tau_rise_NMDA(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Time constant tau_rise_NMDA must be > 0." );
    }
    P_.tau_rise_NMDA = __v;
  }

  inline double get_tau_decay_NMDA() const {
    return P_.tau_decay_NMDA;
  }
  inline void set_tau_decay_NMDA(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Time constant tau_decay_NMDA must be > 0." );
    }
    P_.tau_decay_NMDA = __v;
  }

  inline double get_tau_decay_GABA_A() const {
    return P_.tau_decay_GABA_A;
  }
  inline void set_tau_decay_GABA_A(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Time constant tau_decay_GABA_A must be > 0." );
    }
    P_.tau_decay_GABA_A = __v;
  }

  inline double get_C_m() const {
    return P_.C_m;
  }
  inline void set_C_m(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Membrane capacitance C_m must be > 0." );
    }
    P_.C_m = __v;
  }

  inline double get_g_L() const {
    return P_.g_L;
  }
  inline void set_g_L(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Membrane leak conductance g_L must be > 0." );
    }
    P_.g_L = __v;
  }

  inline double get_g_AMPA_ext() const {
    return P_.g_AMPA_ext;
  }
  inline void set_g_AMPA_ext(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Membrane AMPA_ext conductance g_AMPA_ext must be > 0." );
    }
    P_.g_AMPA_ext = __v;
  }

  inline double get_g_AMPA() const {
    return P_.g_AMPA;
  }
  inline void set_g_AMPA(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Membrane AMPA conductance g_AMPA must be > 0." );
    }
    P_.g_AMPA = __v;
  }

  inline double get_g_NMDA() const {
    return P_.g_NMDA;
  }
  inline void set_g_NMDA(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Membrane NMDA conductance g_NMDA must be > 0." );
    }
    P_.g_NMDA = __v;
  }

  inline double get_g_GABA_A() const {
    return P_.g_GABA_A;
  }
  inline void set_g_GABA_A(const double __v) {
    if ( __v <= 0 )
    {
        throw nest::BadProperty( "Membrane GABA_A conductance g_GABA_A must be > 0." );
    }
    P_.g_GABA_A = __v;
  }

  inline double get_w_E() const {
    return P_.w_E;
  }
  inline void set_w_E(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Excitatory population coupling w_E must be >= 0." );
    }
    P_.w_E = __v;
  }

  inline double get_w_I() const {
    return P_.w_I;
  }
  inline void set_w_I(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "Inhibitory population coupling w_I must be >= 0." );
    }
    P_.w_I = __v;
  }

  inline std::vector< double >  get_w_E_ext() const {
    return P_.w_E_ext;
  }
  inline void set_w_E_ext(const std::vector< double >  __v) {
    const long old_n_receptors = P_.n_receptors();
    if ( (long) __v.size() < old_n_receptors && P_.has_connections_ )  {
        throw nest::BadProperty(
        "The neuron has connections, therefore the number of ports cannot be "
        "reduced." );
    }
    for ( long i = 0 ; i < P_.n_receptors() ; ++i ) {
        if ( __v[i] < 0 )
        {
            throw nest::BadProperty( "External excitatory population coupling w_E_ext must be >= 0." );
        }
    }
    P_.w_E_ext = __v;
  }

  inline double get_alpha() const {
    return P_.alpha;
  }
  inline void set_alpha(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "alpha frequency must be >= 0." );
    }
    P_.alpha = __v;
  }

  inline double get_beta() const {
    return P_.beta;
  }
  inline void set_beta(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "beta must be >= 0." );
    }
    P_.beta = __v;
  }

  inline double get_lambda_NMDA() const {
    return P_.lambda_NMDA;
  }
  inline void set_lambda_NMDA(const double __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "lambda_NMDA must be >= 0." );
    }
    P_.lambda_NMDA = __v;
  }

  inline double get_I_e() const {
    return P_.I_e;
  }
  inline void set_I_e(const double __v) {
    P_.I_e = __v;
  }

  inline long get_RefractoryCounts() const {
    return V_.RefractoryCounts;
  }
  inline void set_RefractoryCounts(const long __v) {
    if ( __v < 0 )
    {
        throw nest::BadProperty( "RefractoryCounts must be >= 0." );
    }
    V_.RefractoryCounts = __v;
  }

  inline long get_n_receptors() const {
    return P_.n_receptors();
  }

  inline double get_I_L() const {
    return P_.g_L * S_.ode_state[State_::V_m] - V_.g_L_E_L;
  }

  inline double get_I_AMPA() const {
    return P_.g_AMPA * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_AMPA];
  }

  inline double get_I_GABA() const {
    return P_.g_GABA_A * (S_.ode_state[State_::V_m] - P_.E_in) * S_.ode_state[State_::s_GABA];
  }

  inline double get_I_NMDA() const {
    return P_.g_NMDA / ( 1 + P_.lambda_NMDA * std::exp(-V_.minus_beta * S_.ode_state[State_::V_m]) ) *
          (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_NMDA];
  }

  inline std::vector< double > get_I_AMPA_ext() const {
    std::vector< double > I_AMPA_ext;
    const double V_m_E_ex = S_.ode_state[State_::V_m] - P_.E_ex;
    I_AMPA_ext.resize(P_.n_receptors(), 0.0);
    for ( long i = 0 ; i < P_.n_receptors() ; ++i ) {
        I_AMPA_ext[i] = P_.g_AMPA * V_m_E_ex * S_.ode_state[State_::s_AMPA_ext + i];
    }
    return I_AMPA_ext;
  }

  inline double get_I_AMPA_ext_sum() const {
    const std::vector< double > I_AMPA_ext = get_I_AMPA_ext();
    return std::accumulate(I_AMPA_ext.begin(), I_AMPA_ext.end(), 0.0);
    // double I_AMPA_ext = 0.0;
    // for ( long i = 0 ; i < P_.n_receptors() ; ++i ) {
    //     I_AMPA_ext +=
    //         P_.g_AMPA * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_AMPA_ext + i];
    // }
    // return I_AMPA_ext;
  }

  inline std::vector<nest::RingBuffer> & get_spikesExc_ext() {return B_.get_spikesExc_ext();};
  
  inline nest::RingBuffer& get_spikesExc() {return B_.get_spikesExc();};
  
  inline nest::RingBuffer& get_spikesInh() {return B_.get_spikesInh();};
  
  inline nest::RingBuffer& get_currents() {return B_.get_currents();};
  

  // Generate function header
  
  /**
  * @defgroup pif_members Member variables of neuron model.
  * Each model neuron should have precisely the following four data members,
  * which are one instance each of the parameters, state, buffers and variables
  * structures. Experience indicates that the state and variables member should
  * be next to each other to achieve good efficiency (caching).
  * @note Devices require one additional data member, an instance of the @c Device
  *       child class they belong to.
  * @{
  */
  Parameters_ P_;  //!< Free parameters.
  State_      S_;  //!< Dynamic state.
  Variables_  V_;  //!< Internal Variables
  Buffers_    B_;  //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap<iaf_cond_deco2014> recordablesMap_;

  friend int iaf_cond_deco2014_dynamics( double, const double y[], double f[], void* pnode );
  
/** @} */
}; /* neuron iaf_cond_deco2014 */

inline nest::port iaf_cond_deco2014::send_test_event(
    nest::Node& target, nest::rport receptor_type, nest::synindex, bool){
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c nest::SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender(*this);
  return target.handles_test_event(e, receptor_type);
}

inline nest::port iaf_cond_deco2014::handles_test_event(nest::SpikeEvent&, nest::port receptor_type){
  if ( receptor_type < 0 || receptor_type > static_cast< nest::port >( get_n_receptors()) ) {
        // TODO refactor me. The code assumes that there is only one. Check by coco.
        throw nest::IncompatibleReceptorType( receptor_type, get_name(), "SpikeEvent" );
    }
  P_.has_connections_ = true;
  return receptor_type;
}



inline nest::port iaf_cond_deco2014::handles_test_event(
    nest::CurrentEvent&, nest::port receptor_type){
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if (receptor_type != 0)
  throw nest::UnknownReceptorType(receptor_type, get_name());
  return 0;
}

inline nest::port iaf_cond_deco2014::handles_test_event(
    nest::DataLoggingRequest& dlr, nest::port receptor_type){
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if (receptor_type != 0)
  throw nest::UnknownReceptorType(receptor_type, get_name());

  return B_.logger_.connect_logging_device(dlr, recordablesMap_);
}

// TODO call get_status on used or internal components
inline void iaf_cond_deco2014::get_status(DictionaryDatum &__d) const{  
  def<double>(__d, nest::names::V_th, get_V_th());
      
  def<double>(__d, nest::names::V_reset, get_V_reset());
      
  def<double>(__d, nest::names::E_L, get_E_L());
      
  def<double>(__d, nest::names::E_ex, get_E_ex());
      
  def<double>(__d, nest::names::E_in, get_E_in());
      
  def<double>(__d, nest::names::t_ref, get_t_ref());
      
  def<double>(__d, nest::names::tau_decay_AMPA, get_tau_decay_AMPA());
      
  def<double>(__d, nest::names::tau_rise_NMDA, get_tau_rise_NMDA());
      
  def<double>(__d, nest::names::tau_decay_NMDA, get_tau_decay_NMDA());
      
  def<double>(__d, nest::names::tau_decay_GABA_A, get_tau_decay_GABA_A());
      
  def<double>(__d, nest::names::C_m, get_C_m());
      
  def<double>(__d, nest::names::g_L, get_g_L());
      
  def<double>(__d, "g_AMPA_ext", get_g_AMPA_ext());
      
  def<double>(__d, nest::names::g_AMPA, get_g_AMPA());
      
  def<double>(__d, nest::names::g_NMDA, get_g_NMDA());
      
  def<double>(__d, nest::names::g_GABA_A, get_g_GABA_A());
      
  def<double>(__d, "w_E", get_w_E());
      
  def<double>(__d, "w_I", get_w_I());
      
  def<std::vector< double > >(__d, "w_E_ext", get_w_E_ext());
      
  def<double>(__d, nest::names::alpha, get_alpha());
      
  def<double>(__d, nest::names::beta, get_beta());
      
  def<double>(__d, "lambda_NMDA", get_lambda_NMDA());
      
  def<double>(__d, nest::names::I_e, get_I_e());
      
  def<long>(__d, "r", get_r());
      
  def<double>(__d, "s_NMDA", get_s_NMDA());
      
  def<double>(__d, nest::names::V_m, get_V_m());
    

  (*__d)[nest::names::recordables] = recordablesMap_.get_list();
  
  def< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
  if ( P_.__gsl_error_tol <= 0. ){
    throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
  }
  

}

inline void iaf_cond_deco2014::set_status(const DictionaryDatum &__d){

  double tmp_V_th = get_V_th();
  updateValue<double>(__d, nest::names::V_th, tmp_V_th);


  double tmp_V_reset = get_V_reset();
  updateValue<double>(__d, nest::names::V_reset, tmp_V_reset);


  double tmp_E_L = get_E_L();
  updateValue<double>(__d, nest::names::E_L, tmp_E_L);


  double tmp_E_ex = get_E_ex();
  updateValue<double>(__d, nest::names::E_ex, tmp_E_ex);


  double tmp_E_in = get_E_in();
  updateValue<double>(__d, nest::names::E_in, tmp_E_in);


  double tmp_t_ref = get_t_ref();
  updateValue<double>(__d, nest::names::t_ref, tmp_t_ref);


  double tmp_tau_decay_AMPA = get_tau_decay_AMPA();
  updateValue<double>(__d, nest::names::tau_decay_AMPA, tmp_tau_decay_AMPA);


  double tmp_tau_rise_NMDA = get_tau_rise_NMDA();
  updateValue<double>(__d, nest::names::tau_rise_NMDA, tmp_tau_rise_NMDA);


  double tmp_tau_decay_NMDA = get_tau_decay_NMDA();
  updateValue<double>(__d, nest::names::tau_decay_NMDA, tmp_tau_decay_NMDA);


  double tmp_tau_decay_GABA_A = get_tau_decay_GABA_A();
  updateValue<double>(__d, nest::names::tau_decay_GABA_A, tmp_tau_decay_GABA_A);


  double tmp_C_m = get_C_m();
  updateValue<double>(__d, nest::names::C_m, tmp_C_m);


  double tmp_g_L = get_g_L();
  updateValue<double>(__d, nest::names::g_L, tmp_g_L);


  double tmp_g_AMPA_ext = get_g_AMPA_ext();
  updateValue<double>(__d, "g_AMPA_ext", tmp_g_AMPA_ext);


  double tmp_g_AMPA = get_g_AMPA();
  updateValue<double>(__d, nest::names::g_AMPA, tmp_g_AMPA);


  double tmp_g_NMDA = get_g_NMDA();
  updateValue<double>(__d, nest::names::g_NMDA, tmp_g_NMDA);


  double tmp_g_GABA_A = get_g_GABA_A();
  updateValue<double>(__d, nest::names::g_GABA_A, tmp_g_GABA_A);


  double tmp_w_E = get_w_E();
  updateValue<double>(__d, "w_E", tmp_w_E);


  double tmp_w_I = get_w_I();
  updateValue<double>(__d, "w_I", tmp_w_I);


  std::vector< double >  tmp_w_E_ext = get_w_E_ext();
  updateValue<std::vector< double > >(__d, "w_E_ext", tmp_w_E_ext);


  double tmp_alpha = get_alpha();
  updateValue<double>(__d, nest::names::alpha, tmp_alpha);


  double tmp_beta = get_beta();
  updateValue<double>(__d, nest::names::beta, tmp_beta);


  double tmp_lambda_NMDA = get_lambda_NMDA();
  updateValue<double>(__d, "lambda_NMDA", tmp_lambda_NMDA);


  double tmp_I_e = get_I_e();
  updateValue<double>(__d, nest::names::I_e, tmp_I_e);


  long tmp_r = get_r();
  updateValue<long>(__d, "r", tmp_r);


  long tmp_n_receptors = get_n_receptors();
  updateValue<long>(__d, nest::names::n_receptors, tmp_n_receptors);


  double tmp_s_NMDA = get_s_NMDA();
  updateValue<double>(__d, "s_NMDA", tmp_s_NMDA);


  double tmp_V_m = get_V_m();
  updateValue<double>(__d, nest::names::V_m, tmp_V_m);


  double tmp_s_AMPA = get_s_AMPA();
  updateValue<double>(__d, "s_AMPA", tmp_s_AMPA);


  double tmp_s_GABA = get_s_GABA();
  updateValue<double>(__d, "s_GABA", tmp_s_GABA);


  double tmp_x_NMDA = get_x_NMDA();
  updateValue<double>(__d, "x_NMDA", tmp_x_NMDA);


  std::vector< double >tmp_s_AMPA_ext = get_s_AMPA_ext();
  updateValue<std::vector< double >>(__d, "s_AMPA_ext", tmp_s_AMPA_ext);

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status(__d);

  // if we get here, temporaries contain consistent set of properties


  set_V_th(tmp_V_th);



  set_V_reset(tmp_V_reset);



  set_E_L(tmp_E_L);



  set_E_ex(tmp_E_ex);



  set_E_in(tmp_E_in);



  set_t_ref(tmp_t_ref);



  set_tau_decay_AMPA(tmp_tau_decay_AMPA);



  set_tau_rise_NMDA(tmp_tau_rise_NMDA);



  set_tau_decay_NMDA(tmp_tau_decay_NMDA);



  set_tau_decay_GABA_A(tmp_tau_decay_GABA_A);



  set_C_m(tmp_C_m);



  set_g_L(tmp_g_L);



  set_g_AMPA_ext(tmp_g_AMPA_ext);



  set_g_AMPA(tmp_g_AMPA);



  set_g_NMDA(tmp_g_NMDA);



  set_g_GABA_A(tmp_g_GABA_A);



  set_w_E(tmp_w_E);



  set_w_I(tmp_w_I);



  set_w_E_ext(tmp_w_E_ext);



  set_alpha(tmp_alpha);



  set_beta(tmp_beta);



  set_lambda_NMDA(tmp_lambda_NMDA);



  set_I_e(tmp_I_e);



  set_r(tmp_r);



  set_s_NMDA(tmp_s_NMDA);



  set_V_m(tmp_V_m);



  set_s_AMPA(tmp_s_AMPA);



  set_s_GABA(tmp_s_GABA);



  set_x_NMDA(tmp_x_NMDA);



  set_s_AMPA_ext(tmp_s_AMPA_ext);



  updateValue< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
  if ( P_.__gsl_error_tol <= 0. ){
    throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
  }
  
};

#endif /* #ifndef IAF_COND_DECO2014 */
#endif /* HAVE GSL */