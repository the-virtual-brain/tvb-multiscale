
/*
*  iaf_cond_nmda_deco2014.h
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
*  2019-09-27 16:55:37.720629
*/
#ifndef IAF_COND_NMDA_DECO2014
#define IAF_COND_NMDA_DECO2014

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
extern "C" inline int iaf_cond_nmda_deco2014_dynamics( double, const double y[], double f[], void* pnode );


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
  Name: iaf_cond_nmda_deco2014.

  Description:  
     -*- coding: utf-8 -*-

  Name: iaf_cond_nmda_deco2014 - Conductance based leaky integrate-and-fire neuron model
                            with separate excitatory AMPA & NMDA, and inhibitory GABA synapses

  Description:
  iaf_cond_delta_deco2014 is an implementation of a spiking NMDA neuron using IAF dynamics with
  conductance-based synapses.

    dx/dt = -(1/tau_NMDA_rise) * x + SUM_k{delta(t-t_k}, where t_k is the time of a spike emitted by neuron this neuron
    ds/dt =  -(1/tau_NMDA_decay) * s + alpha * x * (1 - s)
    where tau_syn = tau_AMPA/GABA for AMPA/GABA respectively, and s bounded in [0,1].

    The spike is emitted when the membrane voltage V_m >= V_th, in which case it is reset to V_reset,
    and kept there for refractory time t_ref.

    The V_m dynamics is given by the following equations:

    dV_m/dt = 1/C_m *( -g_m * (V_m - E_L)
                       -g_AMPA_ext * (V_m - E_ex))*s_AMPA_ext  // input from external AMPA neurons
                       -g_AMPA_rec * (V_m - E_ex))*s_AMPA_rec// input from recursive AMPA neurons
                       -g_NMDA / (1 + lambda_NMDA * exp(-beta*V_m)) * (V_m - E_ex))*s_NMDA // input from recursive NMDA neurons
                       -g_GABA * (V_m - E_in))*s_GABA // input from recursive GABA neurons

    where
    ds_(AMPA/GABA/NMDA)/dt = -(1/tau_(AMPA/GABA/NMDA)) * s_(AMPA/GABA/NMDA) + SUM_j{SUM_k{delta(t-t_j_k}},
    where t_k_j is the time of a spike received by neuron j

  Sends: SpikeEvent

  Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

  References:

  [1] Ponce-Alvarez A., Mantini D., Deco G., Corbetta M., Hagmann P., & Romani G. L. (2014).
  How Local Excitation-Inhibition Ratio Impacts the Whole Brain Dynamics. Journal of Neuroscience.
  34(23), 7886-7898. https://doi.org/10.1523/jneurosci.5068-13.2014

  [2] Meffin, H., Burkitt, A. N., & Grayden, D. B. (2004). An analytical
  model for the large, fluctuating synaptic conductance state typical of
  neocortical neurons in vivo. J.  Comput. Neurosci., 16, 159-175.

  [3] Bernander, O., Douglas, R. J., Martin, K. A. C., & Koch, C. (1991).
  Synaptic background activity influences spatiotemporal integration in
  single pyramidal cells.  Proc. Natl. Acad. Sci. USA, 88(24),
  11569-11573.

  [4] Kuhn, Aertsen, Rotter (2004). Neuronal Integration of Synaptic Input in
  the Fluctuation- Driven Regime. Jneurosci 24(10) 2345-2356

  Author: Dionysios Perdikis

  SeeAlso: iaf_cond_beta



  Parameters:
  The following parameters can be set in the status dictionary.
  V_th [mV]  Threshold
  V_reset [mV]  Reset value of the membrane potential
  E_L [mV]  Resting potential.
  E_ex [mV]  Excitatory reversal potential
  E_in [mV]  Inhibitory reversal potential
  t_ref [ms]  1ms for Inh, Refractory period.
  tau_AMPA [ms]  AMPA synapse decay time
  tau_NMDA_rise [ms]  NMDA synapse rise time
  tau_NMDA_decay [ms]  NMDA synapse decay time
  tau_GABA [ms]  GABA synapse decay time
  C_m [pF]  200 for Inh, Capacity of the membrane
  g_m [nS]  20nS for Inh, Membrane leak conductance
  g_AMPA_ext [nS]  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
  g_AMPA_rec [nS]  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
  g_NMDA [nS]  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
  g_GABA [nS]  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
  alpha [kHz] 
  beta [real] 
  lamda_NMDA [real] 
  I_e [pA]  External current.
  

  Dynamic state variables:
  r [integer]  counts number of tick during the refractory period
  

  Initial values:
  V_m [mV]  membrane potential
  

  References: Empty

  Sends: nest::SpikeEvent

  Receives: Spike, Current, DataLoggingRequest
*/
class iaf_cond_nmda_deco2014 : public nest::Archiving_Node{
public:
  /**
  * The constructor is only used to create the model prototype in the model manager.
  */
  iaf_cond_nmda_deco2014();

  /**
  * The copy constructor is used to create model copies and instances of the model.
  * @node The copy constructor needs to initialize the parameters and the state.
  *       Initialization of buffers and interal variables is deferred to
  *       @c init_buffers_() and @c calibrate().
  */
  iaf_cond_nmda_deco2014(const iaf_cond_nmda_deco2014 &);

  /**
  * Releases resources.
  */
  ~iaf_cond_nmda_deco2014();

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
  /**
     * Synapse types to connect to
     * @note Excluded upper and lower bounds are defined as INF_, SUP_.
     *       Excluding port 0 avoids accidental connections.
     */
    enum SynapseTypes
    {
      INF_SPIKE_RECEPTOR = 0,
      SPIKESEXC_AMPA_EXT ,
      SPIKESEXC_AMPA_REC ,
      SPIKESEXC_NMDA ,
      SPIKESINH_GABA ,
      SUP_SPIKE_RECEPTOR
    };
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
  friend class nest::RecordablesMap<iaf_cond_nmda_deco2014>;
  friend class nest::UniversalDataLogger<iaf_cond_nmda_deco2014>;

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
    double tau_AMPA;

    //!  NMDA synapse rise time
    double tau_NMDA_rise;

    //!  NMDA synapse decay time
    double tau_NMDA_decay;

    //!  GABA synapse decay time
    double tau_GABA;

    //!  200 for Inh, Capacity of the membrane
    double C_m;

    //!  20nS for Inh, Membrane leak conductance
    double g_m;

    //!  2.59nS for Inh, Membrane conductance for AMPA external excitatory currents
    double g_AMPA_ext;

    //!  0.051nS for Inh, Membrane conductance for AMPA recurrent excitatory currents
    double g_AMPA_rec;

    //!  0.16nS for Inh, Membrane conductance for NMDA recurrent excitatory currents
    double g_NMDA;

    //!  8.51nS for Inh, Membrane conductance for GABA recurrent inhibitory currents
    double g_GABA;

    //! 
    double alpha;

    //! 
    double beta;

    //! 
    double lamda_NMDA;

    //!  External current.
    double I_e;

    double __gsl_error_tol;
    /** Initialize parameters to their default values. */
    Parameters_();
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
    
      x_NMDA,
      
      s,
      
      s_AMPA_ext,
      
      s_AMPA_rec,
      
      s_NMDA,
      
      s_GABA,
      //  membrane potential
      V_m,
      STATE_VEC_SIZE
    };
    //! state vector, must be C-array for GSL solver
    double ode_state[STATE_VEC_SIZE];    

    //!  counts number of tick during the refractory period
    long r;

    double s_bound;

    double emmited_spike;

    double I_leak;

    double I_AMPA_ext;

    double I_AMPA_rec;

    double I_NMDA;

    double I_GABA;    
        State_();
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
    Buffers_(iaf_cond_nmda_deco2014 &);
    Buffers_(const Buffers_ &, iaf_cond_nmda_deco2014 &);

    /** Logger for all analog data */
    nest::UniversalDataLogger<iaf_cond_nmda_deco2014> logger_;
    
    std::vector<long> receptor_types_;
    /** buffers and sums up incoming spikes/currents */
    std::vector< nest::RingBuffer > spike_inputs_;

    
    inline nest::RingBuffer& get_spikesExc_AMPA_ext() {  return spike_inputs_[SPIKESEXC_AMPA_EXT - 1]; }
    double spikesExc_AMPA_ext_grid_sum_;
    
    inline nest::RingBuffer& get_spikesExc_AMPA_rec() {  return spike_inputs_[SPIKESEXC_AMPA_REC - 1]; }
    double spikesExc_AMPA_rec_grid_sum_;
    
    inline nest::RingBuffer& get_spikesExc_NMDA() {  return spike_inputs_[SPIKESEXC_NMDA - 1]; }
    double spikesExc_NMDA_grid_sum_;
    
    inline nest::RingBuffer& get_spikesInh_GABA() {  return spike_inputs_[SPIKESINH_GABA - 1]; }
    double spikesInh_GABA_grid_sum_;
    
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
    S_.r = __v;
  }

  inline double get_s_bound() const {
    return (S_.ode_state[State_::s]<0.0) ? (0.0) : (((S_.ode_state[State_::s]>1.0) ? (1.0) : (S_.ode_state[State_::s])));
  }

  inline double get_emmited_spike() const {
    return ((S_.ode_state[State_::V_m]>=P_.V_th)) ? (1.0) : (0.0);
  }

  inline double get_I_leak() const {
    return P_.g_m * (S_.ode_state[State_::V_m] - P_.E_L);
  }

  inline double get_I_AMPA_ext() const {
    return P_.g_AMPA_ext * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_AMPA_ext];
  }

  inline double get_I_AMPA_rec() const {
    return P_.g_AMPA_rec * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_AMPA_rec];
  }

  inline double get_I_NMDA() const {
    return P_.g_NMDA / (1 + P_.lamda_NMDA * std::exp((-P_.beta) * (S_.ode_state[State_::V_m]))) * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_NMDA];
  }

  inline double get_I_GABA() const {
    return P_.g_GABA * (S_.ode_state[State_::V_m] - P_.E_in) * S_.ode_state[State_::s_GABA];
  }

  inline double get_x_NMDA() const {
    return S_.ode_state[State_::x_NMDA];
  }
  inline void set_x_NMDA(const double __v) {
    S_.ode_state[State_::x_NMDA] = __v;
  }

  inline double get_s() const {
    return S_.ode_state[State_::s];
  }
  inline void set_s(const double __v) {
    S_.ode_state[State_::s] = __v;
  }

  inline double get_s_AMPA_ext() const {
    return S_.ode_state[State_::s_AMPA_ext];
  }
  inline void set_s_AMPA_ext(const double __v) {
    S_.ode_state[State_::s_AMPA_ext] = __v;
  }

  inline double get_s_AMPA_rec() const {
    return S_.ode_state[State_::s_AMPA_rec];
  }
  inline void set_s_AMPA_rec(const double __v) {
    S_.ode_state[State_::s_AMPA_rec] = __v;
  }

  inline double get_s_NMDA() const {
    return S_.ode_state[State_::s_NMDA];
  }
  inline void set_s_NMDA(const double __v) {
    S_.ode_state[State_::s_NMDA] = __v;
  }

  inline double get_s_GABA() const {
    return S_.ode_state[State_::s_GABA];
  }
  inline void set_s_GABA(const double __v) {
    S_.ode_state[State_::s_GABA] = __v;
  }

  inline double get_V_m() const {
    return S_.ode_state[State_::V_m];
  }
  inline void set_V_m(const double __v) {
    S_.ode_state[State_::V_m] = __v;
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
    P_.t_ref = __v;
  }


  inline double get_tau_AMPA() const {
    return P_.tau_AMPA;
  }
  inline void set_tau_AMPA(const double __v) {
    P_.tau_AMPA = __v;
  }

  inline double get_tau_NMDA_rise() const {
    return P_.tau_NMDA_rise;
  }
  inline void set_tau_NMDA_rise(const double __v) {
    P_.tau_NMDA_rise = __v;
  }

  inline double get_tau_NMDA_decay() const {
    return P_.tau_NMDA_decay;
  }
  inline void set_tau_NMDA_decay(const double __v) {
    P_.tau_NMDA_decay = __v;
  }

  inline double get_tau_GABA() const {
    return P_.tau_GABA;
  }
  inline void set_tau_GABA(const double __v) {
    P_.tau_GABA = __v;
  }

  inline double get_C_m() const {
    return P_.C_m;
  }
  inline void set_C_m(const double __v) {
    P_.C_m = __v;
  }

  inline double get_g_m() const {
    return P_.g_m;
  }
  inline void set_g_m(const double __v) {
    P_.g_m = __v;
  }

  inline double get_g_AMPA_ext() const {
    return P_.g_AMPA_ext;
  }
  inline void set_g_AMPA_ext(const double __v) {
    P_.g_AMPA_ext = __v;
  }

  inline double get_g_AMPA_rec() const {
    return P_.g_AMPA_rec;
  }
  inline void set_g_AMPA_rec(const double __v) {
    P_.g_AMPA_rec = __v;
  }

  inline double get_g_NMDA() const {
    return P_.g_NMDA;
  }
  inline void set_g_NMDA(const double __v) {
    P_.g_NMDA = __v;
  }

  inline double get_g_GABA() const {
    return P_.g_GABA;
  }
  inline void set_g_GABA(const double __v) {
    P_.g_GABA = __v;
  }

  inline double get_alpha() const {
    return P_.alpha;
  }
  inline void set_alpha(const double __v) {
    P_.alpha = __v;
  }

  inline double get_beta() const {
    return P_.beta;
  }
  inline void set_beta(const double __v) {
    P_.beta = __v;
  }

  inline double get_lamda_NMDA() const {
    return P_.lamda_NMDA;
  }
  inline void set_lamda_NMDA(const double __v) {
    P_.lamda_NMDA = __v;
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
    V_.RefractoryCounts = __v;
  }


  
  inline nest::RingBuffer& get_spikesExc_AMPA_ext() {return B_.get_spikesExc_AMPA_ext();};
  
  inline nest::RingBuffer& get_spikesExc_AMPA_rec() {return B_.get_spikesExc_AMPA_rec();};
  
  inline nest::RingBuffer& get_spikesExc_NMDA() {return B_.get_spikesExc_NMDA();};
  
  inline nest::RingBuffer& get_spikesInh_GABA() {return B_.get_spikesInh_GABA();};
  
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
  static nest::RecordablesMap<iaf_cond_nmda_deco2014> recordablesMap_;

  friend int iaf_cond_nmda_deco2014_dynamics( double, const double y[], double f[], void* pnode );
  
/** @} */
}; /* neuron iaf_cond_nmda_deco2014 */

inline nest::port iaf_cond_nmda_deco2014::send_test_event(
    nest::Node& target, nest::rport receptor_type, nest::synindex, bool){
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c nest::SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender(*this);
  return target.handles_test_event(e, receptor_type);
}

inline nest::port iaf_cond_nmda_deco2014::handles_test_event(nest::SpikeEvent&, nest::port receptor_type){
  assert( B_.spike_inputs_.size() == 4 );

    if ( !( INF_SPIKE_RECEPTOR < receptor_type && receptor_type < SUP_SPIKE_RECEPTOR ) )
    {
      throw nest::UnknownReceptorType( receptor_type, get_name() );
      return 0;
    }
    else {
      return receptor_type - 1;
    }
}



inline nest::port iaf_cond_nmda_deco2014::handles_test_event(
    nest::CurrentEvent&, nest::port receptor_type){
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if (receptor_type != 0)
  throw nest::UnknownReceptorType(receptor_type, get_name());
  return 0;
}

inline nest::port iaf_cond_nmda_deco2014::handles_test_event(
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
inline void iaf_cond_nmda_deco2014::get_status(DictionaryDatum &__d) const{  

  def<double>(__d, "V_th", get_V_th());

  def<double>(__d, "V_reset", get_V_reset());

  def<double>(__d, "E_L", get_E_L());

  def<double>(__d, "E_ex", get_E_ex());

  def<double>(__d, "E_in", get_E_in());

  def<double>(__d, "t_ref", get_t_ref());

  def<double>(__d, "tau_AMPA", get_tau_AMPA());

  def<double>(__d, "tau_NMDA_rise", get_tau_NMDA_rise());

  def<double>(__d, "tau_NMDA_decay", get_tau_NMDA_decay());

  def<double>(__d, "tau_GABA", get_tau_GABA());

  def<double>(__d, "C_m", get_C_m());

  def<double>(__d, "g_m", get_g_m());

  def<double>(__d, "g_AMPA_ext", get_g_AMPA_ext());
      
  def<double>(__d, "g_AMPA_rec", get_g_AMPA_rec());
      
  def<double>(__d, "g_NMDA", get_g_NMDA());
      
  def<double>(__d, "g_GABA", get_g_GABA());

  def<double>(__d, "alpha", get_alpha());
      
  def<double>(__d, "beta", get_beta());
      
  def<double>(__d, "lamda_NMDA", get_lamda_NMDA());
      
  def<double>(__d, "I_e", get_I_e());
      
  def<long>(__d, "r", get_r());
      
  def<double>(__d, "s_bound", get_s_bound());
      
  def<double>(__d, "emmited_spike", get_emmited_spike());
      
  def<double>(__d, "I_leak", get_I_leak());
      
  def<double>(__d, "I_AMPA_ext", get_I_AMPA_ext());
      
  def<double>(__d, "I_AMPA_rec", get_I_AMPA_rec());
      
  def<double>(__d, "I_NMDA", get_I_NMDA());
      
  def<double>(__d, "I_GABA", get_I_GABA());
      
  def<double>(__d, "x_NMDA", get_x_NMDA());
      
  def<double>(__d, "s", get_s());
      
  def<double>(__d, "s_AMPA_ext", get_s_AMPA_ext());
      
  def<double>(__d, "s_AMPA_rec", get_s_AMPA_rec());
      
  def<double>(__d, "s_NMDA", get_s_NMDA());
      
  def<double>(__d, "s_GABA", get_s_GABA());
      
  def<double>(__d, "V_m", get_V_m());
    DictionaryDatum __receptor_type = new Dictionary();
  ( *__receptor_type )[ "SPIKESEXC_AMPA_EXT" ] = SPIKESEXC_AMPA_EXT;
  ( *__receptor_type )[ "SPIKESEXC_AMPA_REC" ] = SPIKESEXC_AMPA_REC;
  ( *__receptor_type )[ "SPIKESEXC_NMDA" ] = SPIKESEXC_NMDA;
  ( *__receptor_type )[ "SPIKESINH_GABA" ] = SPIKESINH_GABA;
  
  ( *__d )[ "receptor_types" ] = __receptor_type;
  

  (*__d)[nest::names::recordables] = recordablesMap_.get_list();
  
  def< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
  if ( P_.__gsl_error_tol <= 0. ){
    throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
  }
  

}

inline void iaf_cond_nmda_deco2014::set_status(const DictionaryDatum &__d){

  double tmp_V_th = get_V_th();
  updateValue<double>(__d, "V_th", tmp_V_th);


  double tmp_V_reset = get_V_reset();
  updateValue<double>(__d, "V_reset", tmp_V_reset);


  double tmp_E_L = get_E_L();
  updateValue<double>(__d, "E_L", tmp_E_L);


  double tmp_E_ex = get_E_ex();
  updateValue<double>(__d, "E_ex", tmp_E_ex);


  double tmp_E_in = get_E_in();
  updateValue<double>(__d, "E_in", tmp_E_in);


  double tmp_t_ref = get_t_ref();
  updateValue<double>(__d, "t_ref", tmp_t_ref);


  double tmp_tau_AMPA = get_tau_AMPA();
  updateValue<double>(__d, "tau_AMPA", tmp_tau_AMPA);


  double tmp_tau_NMDA_rise = get_tau_NMDA_rise();
  updateValue<double>(__d, "tau_NMDA_rise", tmp_tau_NMDA_rise);


  double tmp_tau_NMDA_decay = get_tau_NMDA_decay();
  updateValue<double>(__d, "tau_NMDA_decay", tmp_tau_NMDA_decay);


  double tmp_tau_GABA = get_tau_GABA();
  updateValue<double>(__d, "tau_GABA", tmp_tau_GABA);


  double tmp_C_m = get_C_m();
  updateValue<double>(__d, "C_m", tmp_C_m);


  double tmp_g_m = get_g_m();
  updateValue<double>(__d, "g_m", tmp_g_m);


  double tmp_g_AMPA_ext = get_g_AMPA_ext();
  updateValue<double>(__d, "g_AMPA_ext", tmp_g_AMPA_ext);


  double tmp_g_AMPA_rec = get_g_AMPA_rec();
  updateValue<double>(__d, "g_AMPA_rec", tmp_g_AMPA_rec);


  double tmp_g_NMDA = get_g_NMDA();
  updateValue<double>(__d, "g_NMDA", tmp_g_NMDA);


  double tmp_g_GABA = get_g_GABA();
  updateValue<double>(__d, "g_GABA", tmp_g_GABA);


  double tmp_alpha = get_alpha();
  updateValue<double>(__d, "alpha", tmp_alpha);


  double tmp_beta = get_beta();
  updateValue<double>(__d, "beta", tmp_beta);


  double tmp_lamda_NMDA = get_lamda_NMDA();
  updateValue<double>(__d, "lamda_NMDA", tmp_lamda_NMDA);


  double tmp_I_e = get_I_e();
  updateValue<double>(__d, "I_e", tmp_I_e);


  long tmp_r = get_r();
  updateValue<long>(__d, "r", tmp_r);

  
// ignores 's_bound' double' since it is an function and setter isn't defined

  
// ignores 'emmited_spike' double' since it is an function and setter isn't defined

  
// ignores 'I_leak' double' since it is an function and setter isn't defined

  
// ignores 'I_AMPA_ext' double' since it is an function and setter isn't defined

  
// ignores 'I_AMPA_rec' double' since it is an function and setter isn't defined

  
// ignores 'I_NMDA' double' since it is an function and setter isn't defined

  
// ignores 'I_GABA' double' since it is an function and setter isn't defined

  

  double tmp_x_NMDA = get_x_NMDA();
  updateValue<double>(__d, "x_NMDA", tmp_x_NMDA);

  

  double tmp_s = get_s();
  updateValue<double>(__d, "s", tmp_s);

  

  double tmp_s_AMPA_ext = get_s_AMPA_ext();
  updateValue<double>(__d, "s_AMPA_ext", tmp_s_AMPA_ext);

  

  double tmp_s_AMPA_rec = get_s_AMPA_rec();
  updateValue<double>(__d, "s_AMPA_rec", tmp_s_AMPA_rec);

  

  double tmp_s_NMDA = get_s_NMDA();
  updateValue<double>(__d, "s_NMDA", tmp_s_NMDA);

  

  double tmp_s_GABA = get_s_GABA();
  updateValue<double>(__d, "s_GABA", tmp_s_GABA);

  

  double tmp_V_m = get_V_m();
  updateValue<double>(__d, "V_m", tmp_V_m);

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



  set_tau_AMPA(tmp_tau_AMPA);



  set_tau_NMDA_rise(tmp_tau_NMDA_rise);



  set_tau_NMDA_decay(tmp_tau_NMDA_decay);



  set_tau_GABA(tmp_tau_GABA);



  set_C_m(tmp_C_m);



  set_g_m(tmp_g_m);



  set_g_AMPA_ext(tmp_g_AMPA_ext);



  set_g_AMPA_rec(tmp_g_AMPA_rec);



  set_g_NMDA(tmp_g_NMDA);



  set_g_GABA(tmp_g_GABA);



  set_alpha(tmp_alpha);



  set_beta(tmp_beta);



  set_lamda_NMDA(tmp_lamda_NMDA);



  set_I_e(tmp_I_e);



  set_r(tmp_r);


  // ignores 's_bound' double' since it is an function and setter isn't defined

  // ignores 'emmited_spike' double' since it is an function and setter isn't defined

  // ignores 'I_leak' double' since it is an function and setter isn't defined

  // ignores 'I_AMPA_ext' double' since it is an function and setter isn't defined

  // ignores 'I_AMPA_rec' double' since it is an function and setter isn't defined

  // ignores 'I_NMDA' double' since it is an function and setter isn't defined

  // ignores 'I_GABA' double' since it is an function and setter isn't defined


  set_x_NMDA(tmp_x_NMDA);



  set_s(tmp_s);



  set_s_AMPA_ext(tmp_s_AMPA_ext);



  set_s_AMPA_rec(tmp_s_AMPA_rec);



  set_s_NMDA(tmp_s_NMDA);



  set_s_GABA(tmp_s_GABA);



  set_V_m(tmp_V_m);


  
  updateValue< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
  if ( P_.__gsl_error_tol <= 0. ){
    throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
  }
  
};

#endif /* #ifndef IAF_COND_NMDA_DECO2014 */
#endif /* HAVE GSL */