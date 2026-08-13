/**
 *  eglif_cond_alpha_multisyn.h
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
 *  Generated from NESTML at time: 2022-06-14 09:05:14.243707
**/
#ifndef EGLIF_COND_ALPHA_MULTISYN
#define EGLIF_COND_ALPHA_MULTISYN

#include "config.h"

#ifndef HAVE_GSL
#error "The GSL library is required for neurons that require a numerical solver."
#endif

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

// Includes from sli:
#include "dictdatum.h"

namespace nest
{
namespace eglif_cond_alpha_multisyn_names
{
    const Name _V_m( "V_m" );
    const Name _Iadap( "Iadap" );
    const Name _Idep( "Idep" );
    const Name _g_4__X__rec4_spikes( "g_4__X__rec4_spikes" );
    const Name _g_4__X__rec4_spikes__d( "g_4__X__rec4_spikes__d" );
    const Name _g_1__X__rec1_spikes( "g_1__X__rec1_spikes" );
    const Name _g_1__X__rec1_spikes__d( "g_1__X__rec1_spikes__d" );
    const Name _g_2__X__rec2_spikes( "g_2__X__rec2_spikes" );
    const Name _g_2__X__rec2_spikes__d( "g_2__X__rec2_spikes__d" );
    const Name _g_3__X__rec3_spikes( "g_3__X__rec3_spikes" );
    const Name _g_3__X__rec3_spikes__d( "g_3__X__rec3_spikes__d" );
    const Name _C_m( "C_m" );
    const Name _t_ref( "t_ref" );
    const Name _lambda_0( "lambda_0" );
    const Name _tau_V( "tau_V" );
    const Name _V_reset( "V_reset" );
    const Name _tau_m( "tau_m" );
    const Name _E_L( "E_L" );
    const Name _kadap( "kadap" );
    const Name _k2( "k2" );
    const Name _A2( "A2" );
    const Name _k1( "k1" );
    const Name _A1( "A1" );
    const Name _V_th( "V_th" );
    const Name _V_min( "V_min" );
    const Name _Vinit( "Vinit" );
    const Name _E_rev1( "E_rev1" );
    const Name _tau_syn1( "tau_syn1" );
    const Name _E_rev2( "E_rev2" );
    const Name _tau_syn2( "tau_syn2" );
    const Name _E_rev3( "E_rev3" );
    const Name _tau_syn3( "tau_syn3" );
    const Name _E_rev4( "E_rev4" );
    const Name _tau_syn4( "tau_syn4" );
    const Name _I_e( "I_e" );
}
}



/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL. Internally, it is
 *       a first-class C++ function, but cannot be a member function
 *       because of the C-linkage.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 * @param void* Pointer to model neuron instance.
**/
extern "C" inline int eglif_cond_alpha_multisyn_dynamics( double, const double y[], double f[], void* pnode );


#include "nest_time.h"




/* BeginDocumentation
  Name: eglif_cond_alpha_multisyn.

  Description:

    """
  eglif_cond_alpha_multisyn - Conductance based extended-generalized leaky integrate-and-fire neuron model##############################################################################

  Description
  +++++++++++

  aeif_cond_alpha is the adaptive exponential integrate and fire neuron according to Brette and Gerstner (2005), with post-synaptic conductances in the form of a bi-exponential ("alpha") function.

  The membrane potential is given by the following differential equation:

  .. math::

     C_m \frac{dV_m}{dt} =
     -g_L(V_m-E_L)+g_L\Delta_T\exp\left(\frac{V_m-V_{th}}{\Delta_T}\right) -
   g_e(t)(V_m-E_e) \\
                                                       -g_i(t)(V_m-E_i)-w + I_e

  and

  .. math::

   \tau_w \frac{dw}{dt} = a(V_m-E_L) - w

  Note that the membrane potential can diverge to positive infinity due to the exponential term. To avoid numerical instabilities, instead of :math:`V_m`, the value :math:`\min(V_m,V_{peak})` is used in the dynamical equations.


  References
  ++++++++++

  .. [1] Brette R and Gerstner W (2005). Adaptive exponential
         integrate-and-fire model as an effective description of neuronal
         activity. Journal of Neurophysiology. 943637-3642
         DOI: https://doi.org/10.1152/jn.00686.2005


  See also
  ++++++++

  iaf_cond_alpha, aeif_cond_exp
  """


  Parameters:
  The following parameters can be set in the status dictionary.
C_m [pF]  membrane parameters
 Membrane Capacitance
t_ref [ms]  Refractory period
V_reset [mV]  Reset Potential
tau_m [ms]  Membrane time constant
E_L [mV]  Leak reversal Potential (aka resting potential)
 spike adaptation parameters
kadap [pA / (ms mV)]  spike adaptation parameters
 Subthreshold adaptation
k2 [1 / ms]  Spike-triggered adaptation
k1 [1 / ms]  Adaptation time constant
V_th [mV]  Threshold Potential
V_min [mV]  Minimum Membrane Potential
 synaptic parameters
E_rev1 [mV]  synaptic parameters
 Receptor 1 reversal Potential
tau_syn1 [ms]  Synaptic Time Constant for Synapse on Receptor 1
E_rev2 [mV]  Receptor 2 reversal Potential
tau_syn2 [ms]  Synaptic Time Constant for Synapse on Receptor 2
E_rev3 [mV]  Receptor 3 reversal Potential
tau_syn3 [ms]  Synaptic Time Constant for Synapse on Receptor 3    
E_rev4 [mV]  Receptor 4 reversal Potential
tau_syn4 [ms]  Synaptic Time Constant for Synapse on Receptor 4
 constant external input current
I_e [pA]  constant external input current


  Dynamic state variables:
V_m [mV]  Membrane potential
Iadap [pA]  Spike-adaptation current
Idep [pA]  Spike-triggered current


  Sends: nest::SpikeEvent

  Receives: Spike, Current, DataLoggingRequest
*/
class eglif_cond_alpha_multisyn : public nest::ArchivingNode
{
public:
  /**
   * The constructor is only used to create the model prototype in the model manager.
  **/
  eglif_cond_alpha_multisyn();

  /**
   * The copy constructor is used to create model copies and instances of the model.
   * @node The copy constructor needs to initialize the parameters and the state.
   *       Initialization of buffers and interal variables is deferred to
   *       @c init_buffers_() and @c calibrate().
  **/
  eglif_cond_alpha_multisyn(const eglif_cond_alpha_multisyn &);

  /**
   * Destructor.
  **/
  ~eglif_cond_alpha_multisyn() override;

  // -------------------------------------------------------------------------
  //   Import sets of overloaded virtual functions.
  //   See: Technical Issues / Virtual Functions: Overriding, Overloading,
  //        and Hiding
  // -------------------------------------------------------------------------

  using nest::Node::handles_test_event;
  using nest::Node::handle;

  /**
   * Used to validate that we can send nest::SpikeEvent to desired target:port.
  **/
  size_t send_test_event(nest::Node& target, size_t receptor_type, nest::synindex, bool) override;

  // -------------------------------------------------------------------------
  //   Functions handling incoming events.
  //   We tell nest that we can handle incoming events of various types by
  //   defining handle() for the given event.
  // -------------------------------------------------------------------------


  void handle(nest::SpikeEvent &);        //! accept spikes
  void handle(nest::CurrentEvent &);      //! accept input current
  void handle(nest::DataLoggingRequest &);//! allow recording with multimeter
  size_t handles_test_event(nest::SpikeEvent&, size_t) override;
  size_t handles_test_event(nest::CurrentEvent&, size_t) override;
  size_t handles_test_event(nest::DataLoggingRequest&, size_t) override;

  // -------------------------------------------------------------------------
  //   Functions for getting/setting parameters and state values.
  // -------------------------------------------------------------------------

  void get_status(DictionaryDatum &) const override;
  void set_status(const DictionaryDatum &) override;

  // -------------------------------------------------------------------------
  //   Getters/setters for state block
  // -------------------------------------------------------------------------

  inline double get_V_m() const
  {
    return S_.ode_state[State_::V_m];
  }

  inline void set_V_m(const double __v)
  {
    S_.ode_state[State_::V_m] = __v;
  }

  inline double get_Iadap() const
  {
    return S_.ode_state[State_::Iadap];
  }

  inline void set_Iadap(const double __v)
  {
    S_.ode_state[State_::Iadap] = __v;
  }

  inline double get_Idep() const
  {
    return S_.ode_state[State_::Idep];
  }

  inline void set_Idep(const double __v)
  {
    S_.ode_state[State_::Idep] = __v;
  }

  inline double get_g_4__X__rec4_spikes() const
  {
    return S_.ode_state[State_::g_4__X__rec4_spikes];
  }

  inline void set_g_4__X__rec4_spikes(const double __v)
  {
    S_.ode_state[State_::g_4__X__rec4_spikes] = __v;
  }

  inline double get_g_4__X__rec4_spikes__d() const
  {
    return S_.ode_state[State_::g_4__X__rec4_spikes__d];
  }

  inline void set_g_4__X__rec4_spikes__d(const double __v)
  {
    S_.ode_state[State_::g_4__X__rec4_spikes__d] = __v;
  }

  inline double get_g_1__X__rec1_spikes() const
  {
    return S_.ode_state[State_::g_1__X__rec1_spikes];
  }

  inline void set_g_1__X__rec1_spikes(const double __v)
  {
    S_.ode_state[State_::g_1__X__rec1_spikes] = __v;
  }

  inline double get_g_1__X__rec1_spikes__d() const
  {
    return S_.ode_state[State_::g_1__X__rec1_spikes__d];
  }

  inline void set_g_1__X__rec1_spikes__d(const double __v)
  {
    S_.ode_state[State_::g_1__X__rec1_spikes__d] = __v;
  }

  inline double get_g_2__X__rec2_spikes() const
  {
    return S_.ode_state[State_::g_2__X__rec2_spikes];
  }

  inline void set_g_2__X__rec2_spikes(const double __v)
  {
    S_.ode_state[State_::g_2__X__rec2_spikes] = __v;
  }

  inline double get_g_2__X__rec2_spikes__d() const
  {
    return S_.ode_state[State_::g_2__X__rec2_spikes__d];
  }

  inline void set_g_2__X__rec2_spikes__d(const double __v)
  {
    S_.ode_state[State_::g_2__X__rec2_spikes__d] = __v;
  }

  inline double get_g_3__X__rec3_spikes() const
  {
    return S_.ode_state[State_::g_3__X__rec3_spikes];
  }

  inline void set_g_3__X__rec3_spikes(const double __v)
  {
    S_.ode_state[State_::g_3__X__rec3_spikes] = __v;
  }

  inline double get_g_3__X__rec3_spikes__d() const
  {
    return S_.ode_state[State_::g_3__X__rec3_spikes__d];
  }

  inline void set_g_3__X__rec3_spikes__d(const double __v)
  {
    S_.ode_state[State_::g_3__X__rec3_spikes__d] = __v;
  }


  // -------------------------------------------------------------------------
  //   Getters/setters for parameters
  // -------------------------------------------------------------------------

  inline double get_C_m() const
  {
    return P_.C_m;
  }

  inline void set_C_m(const double __v)
  {
    P_.C_m = __v;
  }

  inline double get_t_ref() const
  {
    return P_.t_ref;
  }

  inline void set_t_ref(const double __v)
  {
    P_.t_ref = __v;
  }
  
  inline double get_lambda_0() const
  {
    return P_.lambda_0;
  }

  inline void set_lambda_0(const double __v)
  {
    P_.lambda_0 = __v;
  }
  
  
  inline double get_tau_V() const
  {
    return P_.tau_V;
  }

  inline void set_tau_V(const double __v)
  {
    P_.tau_V = __v;
  }
  

  inline double get_V_reset() const
  {
    return P_.V_reset;
  }

  inline void set_V_reset(const double __v)
  {
    P_.V_reset = __v;
  }

  inline double get_tau_m() const
  {
    return P_.tau_m;
  }

  inline void set_tau_m(const double __v)
  {
    P_.tau_m = __v;
  }

  inline double get_E_L() const
  {
    return P_.E_L;
  }

  inline void set_E_L(const double __v)
  {
    P_.E_L = __v;
  }

  inline double get_kadap() const
  {
    return P_.kadap;
  }

  inline void set_kadap(const double __v)
  {
    P_.kadap = __v;
  }

  inline double get_k2() const
  {
    return P_.k2;
  }

  inline void set_k2(const double __v)
  {
    P_.k2 = __v;
  }

  inline double get_A2() const
  {
    return P_.A2;
  }

  inline void set_A2(const double __v)
  {
    P_.A2 = __v;
  }

  inline double get_k1() const
  {
    return P_.k1;
  }

  inline void set_k1(const double __v)
  {
    P_.k1 = __v;
  }

  inline double get_A1() const
  {
    return P_.A1;
  }

  inline void set_A1(const double __v)
  {
    P_.A1 = __v;
  }

  inline double get_V_th() const
  {
    return P_.V_th;
  }

  inline void set_V_th(const double __v)
  {
    P_.V_th = __v;
  }

  inline double get_V_min() const
  {
    return P_.V_min;
  }

  inline void set_V_min(const double __v)
  {
    P_.V_min = __v;
  }
  
  inline double get_Vinit() const
  {
    return P_.Vinit;
  }

  inline void set_Vinit(const double __v)
  {
    P_.Vinit = __v;
  }

  inline double get_E_rev1() const
  {
    return P_.E_rev1;
  }

  inline void set_E_rev1(const double __v)
  {
    P_.E_rev1 = __v;
  }

  inline double get_tau_syn1() const
  {
    return P_.tau_syn1;
  }

  inline void set_tau_syn1(const double __v)
  {
    P_.tau_syn1 = __v;
  }

  inline double get_E_rev2() const
  {
    return P_.E_rev2;
  }

  inline void set_E_rev2(const double __v)
  {
    P_.E_rev2 = __v;
  }

  inline double get_tau_syn2() const
  {
    return P_.tau_syn2;
  }

  inline void set_tau_syn2(const double __v)
  {
    P_.tau_syn2 = __v;
  }

  inline double get_E_rev3() const
  {
    return P_.E_rev3;
  }

  inline void set_E_rev3(const double __v)
  {
    P_.E_rev3 = __v;
  }

  inline double get_tau_syn3() const
  {
    return P_.tau_syn3;
  }

  inline void set_tau_syn3(const double __v)
  {
    P_.tau_syn3 = __v;
  }

  inline double get_E_rev4() const
  {
    return P_.E_rev4;
  }

  inline void set_E_rev4(const double __v)
  {
    P_.E_rev4 = __v;
  }

  inline double get_tau_syn4() const
  {
    return P_.tau_syn4;
  }

  inline void set_tau_syn4(const double __v)
  {
    P_.tau_syn4 = __v;
  }

  inline double get_I_e() const
  {
    return P_.I_e;
  }

  inline void set_I_e(const double __v)
  {
    P_.I_e = __v;
  }


  // -------------------------------------------------------------------------
  //   Getters/setters for internals
  // -------------------------------------------------------------------------

  inline double get_PSConInit_rec1() const
  {
    return V_.PSConInit_rec1;
  }

  inline void set_PSConInit_rec1(const double __v)
  {
    V_.PSConInit_rec1 = __v;
  }

  inline double get___h() const
  {
    return V_.__h;
  }

  inline void set___h(const double __v)
  {
    V_.__h = __v;
  }

  inline double get_PSConInit_rec2() const
  {
    return V_.PSConInit_rec2;
  }

  inline void set_PSConInit_rec2(const double __v)
  {
    V_.PSConInit_rec2 = __v;
  }

  inline double get_PSConInit_rec3() const
  {
    return V_.PSConInit_rec3;
  }

  inline void set_PSConInit_rec3(const double __v)
  {
    V_.PSConInit_rec3 = __v;
  }

  inline double get_PSConInit_rec4() const
  {
    return V_.PSConInit_rec4;
  }

  inline void set_PSConInit_rec4(const double __v)
  {
    V_.PSConInit_rec4 = __v;
  }

  inline long get_RefractoryCounts() const
  {
    return V_.RefractoryCounts;
  }

  inline void set_RefractoryCounts(const long __v)
  {
    V_.RefractoryCounts = __v;
  }

  inline long get_r() const
  {
    return V_.r;
  }

  inline void set_r(const long __v)
  {
    V_.r = __v;
  }

  inline double get___P__Idep__Idep() const
  {
    return V_.__P__Idep__Idep;
  }

  inline void set___P__Idep__Idep(const double __v)
  {
    V_.__P__Idep__Idep = __v;
  }

  inline double get___P__g_4__X__rec4_spikes__g_4__X__rec4_spikes() const
  {
    return V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes;
  }

  inline void set___P__g_4__X__rec4_spikes__g_4__X__rec4_spikes(const double __v)
  {
    V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes = __v;
  }

  inline double get___P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d() const
  {
    return V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d;
  }

  inline void set___P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d(const double __v)
  {
    V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d = __v;
  }

  inline double get___P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes() const
  {
    return V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes;
  }

  inline void set___P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes(const double __v)
  {
    V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes = __v;
  }

  inline double get___P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d() const
  {
    return V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d;
  }

  inline void set___P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d(const double __v)
  {
    V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d = __v;
  }

  inline double get___P__g_1__X__rec1_spikes__g_1__X__rec1_spikes() const
  {
    return V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes;
  }

  inline void set___P__g_1__X__rec1_spikes__g_1__X__rec1_spikes(const double __v)
  {
    V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes = __v;
  }

  inline double get___P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d() const
  {
    return V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d;
  }

  inline void set___P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d(const double __v)
  {
    V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d = __v;
  }

  inline double get___P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes() const
  {
    return V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes;
  }

  inline void set___P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes(const double __v)
  {
    V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes = __v;
  }

  inline double get___P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d() const
  {
    return V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d;
  }

  inline void set___P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d(const double __v)
  {
    V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d = __v;
  }

  inline double get___P__g_2__X__rec2_spikes__g_2__X__rec2_spikes() const
  {
    return V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes;
  }

  inline void set___P__g_2__X__rec2_spikes__g_2__X__rec2_spikes(const double __v)
  {
    V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes = __v;
  }

  inline double get___P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d() const
  {
    return V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d;
  }

  inline void set___P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d(const double __v)
  {
    V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d = __v;
  }

  inline double get___P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes() const
  {
    return V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes;
  }

  inline void set___P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes(const double __v)
  {
    V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes = __v;
  }

  inline double get___P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d() const
  {
    return V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d;
  }

  inline void set___P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d(const double __v)
  {
    V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d = __v;
  }

  inline double get___P__g_3__X__rec3_spikes__g_3__X__rec3_spikes() const
  {
    return V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes;
  }

  inline void set___P__g_3__X__rec3_spikes__g_3__X__rec3_spikes(const double __v)
  {
    V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes = __v;
  }

  inline double get___P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d() const
  {
    return V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d;
  }

  inline void set___P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d(const double __v)
  {
    V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d = __v;
  }

  inline double get___P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes() const
  {
    return V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes;
  }

  inline void set___P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes(const double __v)
  {
    V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes = __v;
  }

  inline double get___P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d() const
  {
    return V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d;
  }

  inline void set___P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d(const double __v)
  {
    V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d = __v;
  }



protected:

private:
  void calibrate_variables(bool exclude_timestep=false);

private:
  /**
   * Synapse types to connect to
   * @note Excluded upper and lower bounds are defined as INF_, SUP_.
   *       Excluding port 0 avoids accidental connections.
  **/
  enum SynapseTypes
  {
    INF_SPIKE_RECEPTOR = 0,
      REC1_SPIKES ,
      REC2_SPIKES ,
      REC3_SPIKES ,
      REC4_SPIKES ,
    SUP_SPIKE_RECEPTOR
  };

#if NEST2_COMPAT
  /**
   * Reset state of neuron.
  **/
  void init_state_(const Node& proto);
#endif

  /**
   * Reset internal buffers of neuron.
  **/
  void init_buffers_() override;

  /**
   * Initialize auxiliary quantities, leave parameters and state untouched.
  **/
  void pre_run_hook() override;

  /**
   * Take neuron through given time interval
  **/
  void update(nest::Time const &, const long, const long) override;

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap<eglif_cond_alpha_multisyn>;
  friend class nest::UniversalDataLogger<eglif_cond_alpha_multisyn>;

  /**
   * Free parameters of the neuron.
   *
   *
   *
   * These are the parameters that can be set by the user through @c `node.set()`.
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
  **/
  struct Parameters_
  {    
    //!  membrane parameters
    //!  Membrane Capacitance
    double C_m;
    //!  Refractory period
    double t_ref;
    //!  Reset Potential
    double lambda_0;
    //!  Random spike generation
    double tau_V;
    //!  Random spike generation
    double V_reset;
    //!  Membrane time constant
    double tau_m;
    //!  Leak reversal Potential (aka resting potential)
    //!  spike adaptation parameters
    double E_L;
    //!  spike adaptation parameters
    //!  Subthreshold adaptation
    double kadap;
    //!  Spike-triggered adaptation
    double k2;
    double A2;
    //!  Adaptation time constant
    double k1;
    double A1;
    //!  Threshold Potential
    double V_th;
    //!  Minimum Membrane Potential
    //!  synaptic parameters
    double V_min;
    double Vinit;
    //!  synaptic parameters
    //!  Receptor 1 reversal Potential
    double E_rev1;
    //!  Synaptic Time Constant for Synapse on Receptor 1
    double tau_syn1;
    //!  Receptor 2 reversal Potential
    double E_rev2;
    //!  Synaptic Time Constant for Synapse on Receptor 2
    double tau_syn2;
    //!  Receptor 3 reversal Potential
    double E_rev3;
    //!  Synaptic Time Constant for Synapse on Receptor 3    
    double tau_syn3;
    //!  Receptor 4 reversal Potential
    double E_rev4;
    //!  Synaptic Time Constant for Synapse on Receptor 4
    //!  constant external input current
    double tau_syn4;
    //!  constant external input current
    double I_e;

    double __gsl_error_tol;

    /**
     * Initialize parameters to their default values.
    **/
    Parameters_();
  };

  /**
   * Dynamic state of the neuron.
   *
   *
   *
   * These are the state variables that are advanced in time by calls to
   * @c update(). In many models, some or all of them can be set by the user
   * through @c `node.set()`. The state variables are initialized from the model
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
  **/
  struct State_
  {
    //! Symbolic indices to the elements of the state vector y
    enum StateVecElems
    {
      V_m,
      Iadap,
      Idep,
      g_4__X__rec4_spikes,
      g_4__X__rec4_spikes__d,
      g_1__X__rec1_spikes,
      g_1__X__rec1_spikes__d,
      g_2__X__rec2_spikes,
      g_2__X__rec2_spikes__d,
      g_3__X__rec3_spikes,
      g_3__X__rec3_spikes__d,
      // moved state variables from synapse
      STATE_VEC_SIZE
    };

    //! state vector, must be C-array for GSL solver
    double ode_state[STATE_VEC_SIZE];
    
    double time;

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
  **/
  struct Variables_
  {
    //!  Impulse to add to DG_1 on spike arrival to evoke unit-amplitude conductance excursion
    double PSConInit_rec1;
    double __h;
    //!  Impulse to add to DG_2 on spike arrival to evoke unit-amplitude conductance excursion
    double PSConInit_rec2;
    //!  Impulse to add to DG_3 on spike arrival to evoke unit-amplitude conductance excursion
    double PSConInit_rec3;
    //!  Impulse to add to DG_4 on spike arrival to evoke unit-amplitude conductance excursion
    double PSConInit_rec4;
    //!  refractory time in steps
    //!  counts number of tick during the refractory period
    long RefractoryCounts;
    //!  counts number of tick during the refractory period
    long r;
    double __P__Idep__Idep;
    double __P__g_4__X__rec4_spikes__g_4__X__rec4_spikes;
    double __P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d;
    double __P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes;
    double __P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d;
    double __P__g_1__X__rec1_spikes__g_1__X__rec1_spikes;
    double __P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d;
    double __P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes;
    double __P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d;
    double __P__g_2__X__rec2_spikes__g_2__X__rec2_spikes;
    double __P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d;
    double __P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes;
    double __P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d;
    double __P__g_3__X__rec3_spikes__g_3__X__rec3_spikes;
    double __P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d;
    double __P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes;
    double __P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d;
  };

  /**
   * Buffers of the neuron.
   * Usually buffers for incoming spikes and data logged for analog recorders.
   * Buffers must be initialized by @c init_buffers_(), which is called before
   * @c calibrate() on the first call to @c Simulate after the start of NEST,
   * ResetKernel or ResetNetwork.
   * @node Buffers_ needs neither constructor, copy constructor or assignment operator,
   *       since it is initialized by @c init_nodes_(). If Buffers_ has members that
   *       cannot destroy themselves, Buffers_ will need a destructor.
  **/
  struct Buffers_
  {
    Buffers_(eglif_cond_alpha_multisyn &);
    Buffers_(const Buffers_ &, eglif_cond_alpha_multisyn &);

    /**
     * Logger for all analog data
    **/
    nest::UniversalDataLogger<eglif_cond_alpha_multisyn> logger_;
    std::vector<long> receptor_types_;
    // -----------------------------------------------------------------------
    //   Buffers and sums of incoming spikes/currents per timestep
    // -----------------------------------------------------------------------
    std::vector< nest::RingBuffer > spike_inputs_;
    inline nest::RingBuffer& get_rec1_spikes() {  return spike_inputs_[REC1_SPIKES - 1]; }
    double rec1_spikes_grid_sum_;
    inline nest::RingBuffer& get_rec2_spikes() {  return spike_inputs_[REC2_SPIKES - 1]; }
    double rec2_spikes_grid_sum_;
    inline nest::RingBuffer& get_rec3_spikes() {  return spike_inputs_[REC3_SPIKES - 1]; }
    double rec3_spikes_grid_sum_;
    inline nest::RingBuffer& get_rec4_spikes() {  return spike_inputs_[REC4_SPIKES - 1]; }
    double rec4_spikes_grid_sum_;
    //!< Buffer for input (type: pA)
    nest::RingBuffer I_stim;
    inline nest::RingBuffer& get_I_stim() {return I_stim;}
    double I_stim_grid_sum_;

    // -----------------------------------------------------------------------
    //   GSL ODE solver data structures
    // -----------------------------------------------------------------------

    gsl_odeiv_step* __s;    //!< stepping function
    gsl_odeiv_control* __c; //!< adaptive stepsize control function
    gsl_odeiv_evolve* __e;  //!< evolution function
    gsl_odeiv_system __sys; //!< struct describing system

    // __integration_step should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double __step;             //!< step size in ms
    double __integration_step; //!< current integration time step, updated by GSL
  };

  // -------------------------------------------------------------------------
  //   Getters/setters for inline expressions
  // -------------------------------------------------------------------------
  
  inline double get_V_bounded() const
  {
    return std::max(get_V_m(), P_.V_min);
  }

  inline double get_I_syn1() const
  {
    return get_g_1__X__rec1_spikes() * ((std::max(get_V_m(), P_.V_min)) - P_.E_rev1);
  }

  inline double get_I_syn2() const
  {
    return get_g_2__X__rec2_spikes() * ((std::max(get_V_m(), P_.V_min)) - P_.E_rev2);
  }

  inline double get_I_syn3() const
  {
    return get_g_3__X__rec3_spikes() * ((std::max(get_V_m(), P_.V_min)) - P_.E_rev3);
  }

  inline double get_I_syn4() const
  {
    return get_g_4__X__rec4_spikes() * ((std::max(get_V_m(), P_.V_min)) - P_.E_rev4);
  }


  // -------------------------------------------------------------------------
  //   Getters/setters for input buffers
  // -------------------------------------------------------------------------
  
  inline nest::RingBuffer& get_rec1_spikes() {return B_.get_rec1_spikes();};
  inline nest::RingBuffer& get_rec2_spikes() {return B_.get_rec2_spikes();};
  inline nest::RingBuffer& get_rec3_spikes() {return B_.get_rec3_spikes();};
  inline nest::RingBuffer& get_rec4_spikes() {return B_.get_rec4_spikes();};
  inline nest::RingBuffer& get_I_stim() {return B_.get_I_stim();};

  // -------------------------------------------------------------------------
  //   Member variables of neuron model.
  //   Each model neuron should have precisely the following four data members,
  //   which are one instance each of the parameters, state, buffers and variables
  //   structures. Experience indicates that the state and variables member should
  //   be next to each other to achieve good efficiency (caching).
  //   Note: Devices require one additional data member, an instance of the
  //   ``Device`` child class they belong to.
  // -------------------------------------------------------------------------


  Parameters_ P_;  //!< Free parameters.
  State_      S_;  //!< Dynamic state.
  Variables_  V_;  //!< Internal Variables
  Buffers_    B_;  //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap<eglif_cond_alpha_multisyn> recordablesMap_;
  friend int eglif_cond_alpha_multisyn_dynamics( double, const double y[], double f[], void* pnode );







}; /* neuron eglif_cond_alpha_multisyn */

inline size_t eglif_cond_alpha_multisyn::send_test_event(nest::Node& target, size_t receptor_type, nest::synindex, bool)
{
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c nest::SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender(*this);
  return target.handles_test_event(e, receptor_type);
}

inline size_t eglif_cond_alpha_multisyn::handles_test_event(nest::SpikeEvent&, size_t receptor_type)
{
    assert( B_.spike_inputs_.size() == 4 );

    if ( !( INF_SPIKE_RECEPTOR < receptor_type && receptor_type < SUP_SPIKE_RECEPTOR ) )
    {
      throw nest::UnknownReceptorType( receptor_type, get_name() );
      return 0;
    }
    else
    {
      return receptor_type - 1;
    }
}

inline size_t eglif_cond_alpha_multisyn::handles_test_event(nest::CurrentEvent&, size_t receptor_type)
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if (receptor_type != 0)
  {
    throw nest::UnknownReceptorType(receptor_type, get_name());
  }
  return 0;
}

inline size_t eglif_cond_alpha_multisyn::handles_test_event(nest::DataLoggingRequest& dlr, size_t receptor_type)
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if (receptor_type != 0)
  {
    throw nest::UnknownReceptorType(receptor_type, get_name());
  }

  return B_.logger_.connect_logging_device(dlr, recordablesMap_);
}

inline void eglif_cond_alpha_multisyn::get_status(DictionaryDatum &__d) const
{
  // parameters
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_C_m, get_C_m());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_t_ref, get_t_ref());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_lambda_0, get_lambda_0());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_V, get_tau_V());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_reset, get_V_reset());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_m, get_tau_m());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_L, get_E_L());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_kadap, get_kadap());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_k2, get_k2());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_A2, get_A2());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_k1, get_k1());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_A1, get_A1());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_th, get_V_th());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_min, get_V_min());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_Vinit, get_Vinit());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev1, get_E_rev1());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn1, get_tau_syn1());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev2, get_E_rev2());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn2, get_tau_syn2());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev3, get_E_rev3());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn3, get_tau_syn3());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev4, get_E_rev4());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn4, get_tau_syn4());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_I_e, get_I_e());

  // initial values for state variables in ODE or kernel
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_m, get_V_m());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_Iadap, get_Iadap());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_Idep, get_Idep());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_4__X__rec4_spikes, get_g_4__X__rec4_spikes());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_4__X__rec4_spikes__d, get_g_4__X__rec4_spikes__d());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_1__X__rec1_spikes, get_g_1__X__rec1_spikes());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_1__X__rec1_spikes__d, get_g_1__X__rec1_spikes__d());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_2__X__rec2_spikes, get_g_2__X__rec2_spikes());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_2__X__rec2_spikes__d, get_g_2__X__rec2_spikes__d());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_3__X__rec3_spikes, get_g_3__X__rec3_spikes());
  def<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_3__X__rec3_spikes__d, get_g_3__X__rec3_spikes__d());

  ArchivingNode::get_status( __d );
  DictionaryDatum __receptor_type = new Dictionary();
  ( *__receptor_type )[ "REC1_SPIKES" ] = REC1_SPIKES;
  ( *__receptor_type )[ "REC2_SPIKES" ] = REC2_SPIKES;
  ( *__receptor_type )[ "REC3_SPIKES" ] = REC3_SPIKES;
  ( *__receptor_type )[ "REC4_SPIKES" ] = REC4_SPIKES;
  ( *__d )[ "receptor_types" ] = __receptor_type;

  (*__d)[nest::names::recordables] = recordablesMap_.get_list();
  def< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
  if ( P_.__gsl_error_tol <= 0. ){
    throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
  }
}

inline void eglif_cond_alpha_multisyn::set_status(const DictionaryDatum &__d)
{
  // parameters
  double tmp_C_m = get_C_m();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_C_m, tmp_C_m);
  double tmp_t_ref = get_t_ref();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_t_ref, tmp_t_ref);
  double tmp_lambda_0 = get_lambda_0();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_lambda_0, tmp_lambda_0);
  double tmp_tau_V = get_tau_V();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_V, tmp_tau_V);
  double tmp_V_reset = get_V_reset();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_reset, tmp_V_reset);
  double tmp_tau_m = get_tau_m();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_m, tmp_tau_m);
  double tmp_E_L = get_E_L();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_L, tmp_E_L);
  double tmp_kadap = get_kadap();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_kadap, tmp_kadap);
  double tmp_k2 = get_k2();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_k2, tmp_k2);
  double tmp_A2 = get_A2();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_A2, tmp_A2);
  double tmp_k1 = get_k1();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_k1, tmp_k1);
  double tmp_A1 = get_A1();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_A1, tmp_A1);
  double tmp_V_th = get_V_th();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_th, tmp_V_th);
  double tmp_V_min = get_V_min();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_min, tmp_V_min);
  double tmp_Vinit = get_Vinit();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_Vinit, tmp_Vinit);
  double tmp_E_rev1 = get_E_rev1();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev1, tmp_E_rev1);
  double tmp_tau_syn1 = get_tau_syn1();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn1, tmp_tau_syn1);
  double tmp_E_rev2 = get_E_rev2();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev2, tmp_E_rev2);
  double tmp_tau_syn2 = get_tau_syn2();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn2, tmp_tau_syn2);
  double tmp_E_rev3 = get_E_rev3();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev3, tmp_E_rev3);
  double tmp_tau_syn3 = get_tau_syn3();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn3, tmp_tau_syn3);
  double tmp_E_rev4 = get_E_rev4();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_E_rev4, tmp_E_rev4);
  double tmp_tau_syn4 = get_tau_syn4();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_tau_syn4, tmp_tau_syn4);
  double tmp_I_e = get_I_e();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_I_e, tmp_I_e);

  // initial values for state variables in ODE or kernel
  double tmp_V_m = get_V_m();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_V_m, tmp_V_m);
  double tmp_Iadap = get_Iadap();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_Iadap, tmp_Iadap);
  double tmp_Idep = get_Idep();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_Idep, tmp_Idep);
  double tmp_g_4__X__rec4_spikes = get_g_4__X__rec4_spikes();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_4__X__rec4_spikes, tmp_g_4__X__rec4_spikes);
  double tmp_g_4__X__rec4_spikes__d = get_g_4__X__rec4_spikes__d();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_4__X__rec4_spikes__d, tmp_g_4__X__rec4_spikes__d);
  double tmp_g_1__X__rec1_spikes = get_g_1__X__rec1_spikes();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_1__X__rec1_spikes, tmp_g_1__X__rec1_spikes);
  double tmp_g_1__X__rec1_spikes__d = get_g_1__X__rec1_spikes__d();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_1__X__rec1_spikes__d, tmp_g_1__X__rec1_spikes__d);
  double tmp_g_2__X__rec2_spikes = get_g_2__X__rec2_spikes();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_2__X__rec2_spikes, tmp_g_2__X__rec2_spikes);
  double tmp_g_2__X__rec2_spikes__d = get_g_2__X__rec2_spikes__d();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_2__X__rec2_spikes__d, tmp_g_2__X__rec2_spikes__d);
  double tmp_g_3__X__rec3_spikes = get_g_3__X__rec3_spikes();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_3__X__rec3_spikes, tmp_g_3__X__rec3_spikes);
  double tmp_g_3__X__rec3_spikes__d = get_g_3__X__rec3_spikes__d();
  updateValue<double>(__d, nest::eglif_cond_alpha_multisyn_names::_g_3__X__rec3_spikes__d, tmp_g_3__X__rec3_spikes__d);

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  ArchivingNode::set_status(__d);

  // if we get here, temporaries contain consistent set of properties
  set_C_m(tmp_C_m);
  set_t_ref(tmp_t_ref);
  set_V_reset(tmp_V_reset);
  set_tau_m(tmp_tau_m);
  set_E_L(tmp_E_L);
  set_kadap(tmp_kadap);
  set_k2(tmp_k2);
  set_A2(tmp_A2);
  set_k1(tmp_k1);
  set_A1(tmp_A1);
  set_V_th(tmp_V_th);
  set_V_min(tmp_V_min);
  set_E_rev1(tmp_E_rev1);
  set_tau_syn1(tmp_tau_syn1);
  set_E_rev2(tmp_E_rev2);
  set_tau_syn2(tmp_tau_syn2);
  set_E_rev3(tmp_E_rev3);
  set_tau_syn3(tmp_tau_syn3);
  set_E_rev4(tmp_E_rev4);
  set_tau_syn4(tmp_tau_syn4);
  set_I_e(tmp_I_e);
  set_V_m(tmp_V_m);
  set_Iadap(tmp_Iadap);
  set_Idep(tmp_Idep);
  set_g_4__X__rec4_spikes(tmp_g_4__X__rec4_spikes);
  set_g_4__X__rec4_spikes__d(tmp_g_4__X__rec4_spikes__d);
  set_g_1__X__rec1_spikes(tmp_g_1__X__rec1_spikes);
  set_g_1__X__rec1_spikes__d(tmp_g_1__X__rec1_spikes__d);
  set_g_2__X__rec2_spikes(tmp_g_2__X__rec2_spikes);
  set_g_2__X__rec2_spikes__d(tmp_g_2__X__rec2_spikes__d);
  set_g_3__X__rec3_spikes(tmp_g_3__X__rec3_spikes);
  set_g_3__X__rec3_spikes__d(tmp_g_3__X__rec3_spikes__d);


  updateValue< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
  if ( P_.__gsl_error_tol <= 0. )
  {
    throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
  }
};

#endif /* #ifndef EGLIF_COND_ALPHA_MULTISYN */
