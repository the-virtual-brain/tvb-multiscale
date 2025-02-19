#define DEBUG 1
/*
 *  eglif_cond_alpha_multisyn.cpp
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
 *  Generated from NESTML at time: 2022-06-14 09:05:13.763236
**/

// C++ includes:
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

#include "eglif_cond_alpha_multisyn.h"

// ---------------------------------------------------------------------------
//   Recordables map
// ---------------------------------------------------------------------------
nest::RecordablesMap<eglif_cond_alpha_multisyn> eglif_cond_alpha_multisyn::recordablesMap_;
namespace nest
{

  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
template <> void RecordablesMap<eglif_cond_alpha_multisyn>::create()
  {
    // add state variables to recordables map
   insert_(eglif_cond_alpha_multisyn_names::_V_m, &eglif_cond_alpha_multisyn::get_V_m);
   insert_(eglif_cond_alpha_multisyn_names::_Iadap, &eglif_cond_alpha_multisyn::get_Iadap);
   insert_(eglif_cond_alpha_multisyn_names::_Idep, &eglif_cond_alpha_multisyn::get_Idep);
   insert_(eglif_cond_alpha_multisyn_names::_g_4__X__rec4_spikes, &eglif_cond_alpha_multisyn::get_g_4__X__rec4_spikes);
   insert_(eglif_cond_alpha_multisyn_names::_g_4__X__rec4_spikes__d, &eglif_cond_alpha_multisyn::get_g_4__X__rec4_spikes__d);
   insert_(eglif_cond_alpha_multisyn_names::_g_1__X__rec1_spikes, &eglif_cond_alpha_multisyn::get_g_1__X__rec1_spikes);
   insert_(eglif_cond_alpha_multisyn_names::_g_1__X__rec1_spikes__d, &eglif_cond_alpha_multisyn::get_g_1__X__rec1_spikes__d);
   insert_(eglif_cond_alpha_multisyn_names::_g_2__X__rec2_spikes, &eglif_cond_alpha_multisyn::get_g_2__X__rec2_spikes);
   insert_(eglif_cond_alpha_multisyn_names::_g_2__X__rec2_spikes__d, &eglif_cond_alpha_multisyn::get_g_2__X__rec2_spikes__d);
   insert_(eglif_cond_alpha_multisyn_names::_g_3__X__rec3_spikes, &eglif_cond_alpha_multisyn::get_g_3__X__rec3_spikes);
   insert_(eglif_cond_alpha_multisyn_names::_g_3__X__rec3_spikes__d, &eglif_cond_alpha_multisyn::get_g_3__X__rec3_spikes__d);

    // Add vector variables  
  }
}

// ---------------------------------------------------------------------------
//   Default constructors defining default parameters and state
//   Note: the implementation is empty. The initialization is of variables
//   is a part of eglif_cond_alpha_multisyn's constructor.
// ---------------------------------------------------------------------------

eglif_cond_alpha_multisyn::Parameters_::Parameters_()
{
}

eglif_cond_alpha_multisyn::State_::State_()
{
}

// ---------------------------------------------------------------------------
//   Parameter and state extractions and manipulation functions
// ---------------------------------------------------------------------------

eglif_cond_alpha_multisyn::Buffers_::Buffers_(eglif_cond_alpha_multisyn &n):
  logger_(n)
  , spike_inputs_( std::vector< nest::RingBuffer >( SUP_SPIKE_RECEPTOR - 1 ) )
  , __s( 0 ), __c( 0 ), __e( 0 )
{
  // Initialization of the remaining members is deferred to init_buffers_().
}

eglif_cond_alpha_multisyn::Buffers_::Buffers_(const Buffers_ &, eglif_cond_alpha_multisyn &n):
  logger_(n)
  , spike_inputs_( std::vector< nest::RingBuffer >( SUP_SPIKE_RECEPTOR - 1 ) )
  , __s( 0 ), __c( 0 ), __e( 0 )
{
  // Initialization of the remaining members is deferred to init_buffers_().
}

// ---------------------------------------------------------------------------
//   Default constructor for node
// ---------------------------------------------------------------------------

eglif_cond_alpha_multisyn::eglif_cond_alpha_multisyn():ArchivingNode(), P_(), S_(), B_(*this)
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

  pre_run_hook();

  // use a default "good enough" value for the absolute error. It can be adjusted via `node.set()`
  P_.__gsl_error_tol = 1e-3;
  // initial values for parameters
    
    P_.C_m = 281.0; // as pF
    
    P_.t_ref = 0.0; // as ms
 
    /* Added for stochasticity*/
    P_.lambda_0 = ((0.001));	// [1/ms]
    P_.tau_V = ((0.5));		// [mV]
    
    P_.V_reset = (-(60.0)); // as mV
    
    P_.tau_m = 30.0; // as ms
    
    P_.E_L = (-(70.6)); // as mV
    
    P_.kadap = 4 / (1.0 * 1.0); // as pA / (ms mV)
    
    P_.k2 = pow(80.5, (-(1))); // as 1 / ms
    
    P_.A2 = 100.0; // as pA
    
    P_.k1 = pow(144.0, (-(1))); // as 1 / ms
    
    P_.A1 = 100.0; // as pA
    
    P_.V_th = (-(50.4)); // as mV
    
    P_.V_min = (-(150.0)); // as mV
    
    P_.Vinit = ((-60.0));
    
    P_.E_rev1 = 0; // as mV
    
    P_.tau_syn1 = 0.2; // as ms
    
    P_.E_rev2 = (-(80.0)); // as mV
    
    P_.tau_syn2 = 2.0; // as ms
    
    P_.E_rev3 = 0.0; // as mV
    
    P_.tau_syn3 = 2.0; // as ms
    
    P_.E_rev4 = (-(80.0)); // as mV
    
    P_.tau_syn4 = 2.0; // as ms
    
    P_.I_e = 0; // as pA
  // initial values for state variables
    
    S_.ode_state[State_::V_m] = P_.E_L; // as mV
    
    S_.ode_state[State_::Iadap] = 0; // as pA
    
    S_.ode_state[State_::Idep] = 0; // as pA
    
    S_.time = 0;
    
    S_.ode_state[State_::g_4__X__rec4_spikes] = 0; // as real
    
    S_.ode_state[State_::g_4__X__rec4_spikes__d] = 0; // as real
    
    S_.ode_state[State_::g_1__X__rec1_spikes] = 0; // as real
    
    S_.ode_state[State_::g_1__X__rec1_spikes__d] = 0; // as real
    
    S_.ode_state[State_::g_2__X__rec2_spikes] = 0; // as real
    
    S_.ode_state[State_::g_2__X__rec2_spikes__d] = 0; // as real
    
    S_.ode_state[State_::g_3__X__rec3_spikes] = 0; // as real
    
    S_.ode_state[State_::g_3__X__rec3_spikes__d] = 0; // as real
  recordablesMap_.create();
}

// ---------------------------------------------------------------------------
//   Copy constructor for node
// ---------------------------------------------------------------------------

eglif_cond_alpha_multisyn::eglif_cond_alpha_multisyn(const eglif_cond_alpha_multisyn& __n):
  ArchivingNode(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this) {

  // copy parameter struct P_
  P_.C_m = __n.P_.C_m;
  P_.t_ref = __n.P_.t_ref;
  P_.V_reset = __n.P_.V_reset;
  P_.tau_m = __n.P_.tau_m;
  P_.E_L = __n.P_.E_L;
  P_.kadap = __n.P_.kadap;
  P_.k2 = __n.P_.k2;
  P_.A2 = __n.P_.A2;
  P_.k1 = __n.P_.k1;
  P_.A1 = __n.P_.A1;
  P_.V_th = __n.P_.V_th;
  P_.V_min = __n.P_.V_min;
  P_.E_rev1 = __n.P_.E_rev1;
  P_.tau_syn1 = __n.P_.tau_syn1;
  P_.E_rev2 = __n.P_.E_rev2;
  P_.tau_syn2 = __n.P_.tau_syn2;
  P_.E_rev3 = __n.P_.E_rev3;
  P_.tau_syn3 = __n.P_.tau_syn3;
  P_.E_rev4 = __n.P_.E_rev4;
  P_.tau_syn4 = __n.P_.tau_syn4;
  P_.I_e = __n.P_.I_e;

  // copy state struct S_
  S_.ode_state[State_::V_m] = __n.S_.ode_state[State_::V_m];
  S_.ode_state[State_::Iadap] = __n.S_.ode_state[State_::Iadap];
  S_.ode_state[State_::Idep] = __n.S_.ode_state[State_::Idep];
  S_.ode_state[State_::g_4__X__rec4_spikes] = __n.S_.ode_state[State_::g_4__X__rec4_spikes];
  S_.ode_state[State_::g_4__X__rec4_spikes__d] = __n.S_.ode_state[State_::g_4__X__rec4_spikes__d];
  S_.ode_state[State_::g_1__X__rec1_spikes] = __n.S_.ode_state[State_::g_1__X__rec1_spikes];
  S_.ode_state[State_::g_1__X__rec1_spikes__d] = __n.S_.ode_state[State_::g_1__X__rec1_spikes__d];
  S_.ode_state[State_::g_2__X__rec2_spikes] = __n.S_.ode_state[State_::g_2__X__rec2_spikes];
  S_.ode_state[State_::g_2__X__rec2_spikes__d] = __n.S_.ode_state[State_::g_2__X__rec2_spikes__d];
  S_.ode_state[State_::g_3__X__rec3_spikes] = __n.S_.ode_state[State_::g_3__X__rec3_spikes];
  S_.ode_state[State_::g_3__X__rec3_spikes__d] = __n.S_.ode_state[State_::g_3__X__rec3_spikes__d];


  // copy internals V_
  V_.PSConInit_rec1 = __n.V_.PSConInit_rec1;
  V_.__h = __n.V_.__h;
  V_.PSConInit_rec2 = __n.V_.PSConInit_rec2;
  V_.PSConInit_rec3 = __n.V_.PSConInit_rec3;
  V_.PSConInit_rec4 = __n.V_.PSConInit_rec4;
  V_.RefractoryCounts = __n.V_.RefractoryCounts;
  V_.r = __n.V_.r;
  V_.__P__Idep__Idep = __n.V_.__P__Idep__Idep;
  V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes = __n.V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes;
  V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d = __n.V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d;
  V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes = __n.V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes;
  V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d = __n.V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d;
  V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes = __n.V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes;
  V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d = __n.V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d;
  V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes = __n.V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes;
  V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d = __n.V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d;
  V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes = __n.V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes;
  V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d = __n.V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d;
  V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes = __n.V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes;
  V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d = __n.V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d;
  V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes = __n.V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes;
  V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d = __n.V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d;
  V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes = __n.V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes;
  V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d = __n.V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d;
}

// ---------------------------------------------------------------------------
//   Destructor for node
// ---------------------------------------------------------------------------

eglif_cond_alpha_multisyn::~eglif_cond_alpha_multisyn()
{
  // GSL structs may not have been allocated, so we need to protect destruction

  if (B_.__s)
  {
    gsl_odeiv_step_free( B_.__s );
  }

  if (B_.__c)
  {
    gsl_odeiv_control_free( B_.__c );
  }

  if (B_.__e)
  {
    gsl_odeiv_evolve_free( B_.__e );
  }
}

// ---------------------------------------------------------------------------
//   Node initialization functions
// ---------------------------------------------------------------------------

#if NEST2_COMPAT
void eglif_cond_alpha_multisyn::init_state_(const Node& proto)
{
  const eglif_cond_alpha_multisyn& pr = downcast<eglif_cond_alpha_multisyn>(proto);
  S_ = pr.S_;
}
#endif

void eglif_cond_alpha_multisyn::init_buffers_()
{
  get_rec1_spikes().clear(); //includes resize
  get_rec2_spikes().clear(); //includes resize
  get_rec3_spikes().clear(); //includes resize
  get_rec4_spikes().clear(); //includes resize
  get_I_stim().clear(); //includes resize
  B_.logger_.reset(); // includes resize

  if ( B_.__s == 0 )
  {
    B_.__s = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, 11 );
  }
  else
  {
    gsl_odeiv_step_reset( B_.__s );
  }

  if ( B_.__c == 0 )
  {
    B_.__c = gsl_odeiv_control_y_new( P_.__gsl_error_tol, 0.0 );
  }
  else
  {
    gsl_odeiv_control_init( B_.__c, P_.__gsl_error_tol, 0.0, 1.0, 0.0 );
  }

  if ( B_.__e == 0 )
  {
    B_.__e = gsl_odeiv_evolve_alloc( 11 );
  }
  else
  {
    gsl_odeiv_evolve_reset( B_.__e );
  }

  B_.__sys.function = eglif_cond_alpha_multisyn_dynamics;
  B_.__sys.jacobian = NULL;
  B_.__sys.dimension = 11;
  B_.__sys.params = reinterpret_cast< void* >( this );
  B_.__step = nest::Time::get_resolution().get_ms();
  B_.__integration_step = nest::Time::get_resolution().get_ms();
}

void eglif_cond_alpha_multisyn::calibrate_variables(bool exclude_timestep) {
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

  if (exclude_timestep) {    
    V_.PSConInit_rec1 =1.0 * numerics::e / P_.tau_syn1;
    V_.PSConInit_rec2 =1.0 * numerics::e / P_.tau_syn2;
    V_.PSConInit_rec3 =1.0 * numerics::e / P_.tau_syn3;
    V_.PSConInit_rec4 =1.0 * numerics::e / P_.tau_syn4;
    V_.RefractoryCounts =nest::Time(nest::Time::ms((double) (P_.t_ref))).get_steps();
    V_.r =0;
    V_.__P__Idep__Idep =1.0 * std::exp((-(V_.__h)) * P_.k1);
    V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes =1.0 * (V_.__h + P_.tau_syn4) * std::exp((-(V_.__h)) / P_.tau_syn4) / P_.tau_syn4;
    V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn4);
    V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn4) / pow(P_.tau_syn4, 2);
    V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn4) * std::exp((-(V_.__h)) / P_.tau_syn4) / P_.tau_syn4;
    V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes =1.0 * (V_.__h + P_.tau_syn1) * std::exp((-(V_.__h)) / P_.tau_syn1) / P_.tau_syn1;
    V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn1);
    V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn1) / pow(P_.tau_syn1, 2);
    V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn1) * std::exp((-(V_.__h)) / P_.tau_syn1) / P_.tau_syn1;
    V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes =1.0 * (V_.__h + P_.tau_syn2) * std::exp((-(V_.__h)) / P_.tau_syn2) / P_.tau_syn2;
    V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn2);
    V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn2) / pow(P_.tau_syn2, 2);
    V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn2) * std::exp((-(V_.__h)) / P_.tau_syn2) / P_.tau_syn2;
    V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes =1.0 * (V_.__h + P_.tau_syn3) * std::exp((-(V_.__h)) / P_.tau_syn3) / P_.tau_syn3;
    V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn3);
    V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn3) / pow(P_.tau_syn3, 2);
    V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn3) * std::exp((-(V_.__h)) / P_.tau_syn3) / P_.tau_syn3;
  }
  else {
    // internals V_
    V_.PSConInit_rec1 =1.0 * numerics::e / P_.tau_syn1;
    V_.__h =__resolution;
    V_.PSConInit_rec2 =1.0 * numerics::e / P_.tau_syn2;
    V_.PSConInit_rec3 =1.0 * numerics::e / P_.tau_syn3;
    V_.PSConInit_rec4 =1.0 * numerics::e / P_.tau_syn4;
    V_.RefractoryCounts =nest::Time(nest::Time::ms((double) (P_.t_ref))).get_steps();
    V_.r =0;
    V_.__P__Idep__Idep =1.0 * std::exp((-(V_.__h)) * P_.k1);
    V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes =1.0 * (V_.__h + P_.tau_syn4) * std::exp((-(V_.__h)) / P_.tau_syn4) / P_.tau_syn4;
    V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn4);
    V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn4) / pow(P_.tau_syn4, 2);
    V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn4) * std::exp((-(V_.__h)) / P_.tau_syn4) / P_.tau_syn4;
    V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes =1.0 * (V_.__h + P_.tau_syn1) * std::exp((-(V_.__h)) / P_.tau_syn1) / P_.tau_syn1;
    V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn1);
    V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn1) / pow(P_.tau_syn1, 2);
    V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn1) * std::exp((-(V_.__h)) / P_.tau_syn1) / P_.tau_syn1;
    V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes =1.0 * (V_.__h + P_.tau_syn2) * std::exp((-(V_.__h)) / P_.tau_syn2) / P_.tau_syn2;
    V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn2);
    V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn2) / pow(P_.tau_syn2, 2);
    V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn2) * std::exp((-(V_.__h)) / P_.tau_syn2) / P_.tau_syn2;
    V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes =1.0 * (V_.__h + P_.tau_syn3) * std::exp((-(V_.__h)) / P_.tau_syn3) / P_.tau_syn3;
    V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d =1.0 * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn3);
    V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes =(-(1.0)) * V_.__h * std::exp((-(V_.__h)) / P_.tau_syn3) / pow(P_.tau_syn3, 2);
    V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d =1.0 * ((-(V_.__h)) + P_.tau_syn3) * std::exp((-(V_.__h)) / P_.tau_syn3) / P_.tau_syn3;
  }
}

void eglif_cond_alpha_multisyn::pre_run_hook() {
  B_.logger_.init();

  calibrate_variables();

  // buffers B_
}

// ---------------------------------------------------------------------------
//   Update and spike handling functions
// ---------------------------------------------------------------------------

extern "C" inline int eglif_cond_alpha_multisyn_dynamics(double, const double ode_state[], double f[], void* pnode)
{
  typedef eglif_cond_alpha_multisyn::State_ State_;
  // get access to node so we can almost work as in a member function
  assert( pnode );
  const eglif_cond_alpha_multisyn& node = *( reinterpret_cast< eglif_cond_alpha_multisyn* >( pnode ) );

  // ode_state[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.ode_state[].
  f[State_::V_m] = (-(node.get_E_L())) / node.get_tau_m() + std::max(ode_state[State_::V_m], node.get_V_min()) / node.get_tau_m() + node.get_E_rev1() * ode_state[State_::g_1__X__rec1_spikes] / node.get_C_m() + node.get_E_rev4() * ode_state[State_::g_4__X__rec4_spikes] / node.get_C_m() + node.get_I_e() / node.get_C_m() + node.B_.I_stim_grid_sum_ / node.get_C_m() + (node.get_E_rev2() * ode_state[State_::g_2__X__rec2_spikes] + node.get_E_rev3() * ode_state[State_::g_3__X__rec3_spikes] - ode_state[State_::Iadap] + ode_state[State_::Idep] - ode_state[State_::g_1__X__rec1_spikes] * std::max(ode_state[State_::V_m], node.get_V_min()) - ode_state[State_::g_2__X__rec2_spikes] * std::max(ode_state[State_::V_m], node.get_V_min()) - ode_state[State_::g_3__X__rec3_spikes] * std::max(ode_state[State_::V_m], node.get_V_min()) - ode_state[State_::g_4__X__rec4_spikes] * std::max(ode_state[State_::V_m], node.get_V_min())) / node.get_C_m();
  f[State_::Iadap] = (-(node.get_E_L())) * node.get_kadap() - ode_state[State_::Iadap] * node.get_k2() + node.get_kadap() * std::max(ode_state[State_::V_m], node.get_V_min());
  f[State_::Idep] = (-(ode_state[State_::Idep])) * node.get_k1();
  f[State_::g_4__X__rec4_spikes] = 1.0 * ode_state[State_::g_4__X__rec4_spikes__d];
  f[State_::g_4__X__rec4_spikes__d] = (-(ode_state[State_::g_4__X__rec4_spikes])) / pow(node.get_tau_syn4(), 2) - 2 * ode_state[State_::g_4__X__rec4_spikes__d] / node.get_tau_syn4();
  f[State_::g_1__X__rec1_spikes] = 1.0 * ode_state[State_::g_1__X__rec1_spikes__d];
  f[State_::g_1__X__rec1_spikes__d] = (-(ode_state[State_::g_1__X__rec1_spikes])) / pow(node.get_tau_syn1(), 2) - 2 * ode_state[State_::g_1__X__rec1_spikes__d] / node.get_tau_syn1();
  f[State_::g_2__X__rec2_spikes] = 1.0 * ode_state[State_::g_2__X__rec2_spikes__d];
  f[State_::g_2__X__rec2_spikes__d] = (-(ode_state[State_::g_2__X__rec2_spikes])) / pow(node.get_tau_syn2(), 2) - 2 * ode_state[State_::g_2__X__rec2_spikes__d] / node.get_tau_syn2();
  f[State_::g_3__X__rec3_spikes] = 1.0 * ode_state[State_::g_3__X__rec3_spikes__d];
  f[State_::g_3__X__rec3_spikes__d] = (-(ode_state[State_::g_3__X__rec3_spikes])) / pow(node.get_tau_syn3(), 2) - 2 * ode_state[State_::g_3__X__rec3_spikes__d] / node.get_tau_syn3();

  return GSL_SUCCESS;
}

void eglif_cond_alpha_multisyn::update(nest::Time const & origin,const long from, const long to)
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function
  if (S_.time < 2){
	  S_.ode_state[State_::V_m] = get_Vinit();
  }
  
  if ( S_.ode_state[State_::V_m] < P_.V_min){
	  S_.ode_state[State_::V_m] = get_V_min();
    }

  for ( long lag = from ; lag < to ; ++lag )
  {
    B_.rec1_spikes_grid_sum_ = get_rec1_spikes().get_value(lag);
    B_.rec2_spikes_grid_sum_ = get_rec2_spikes().get_value(lag);
    B_.rec3_spikes_grid_sum_ = get_rec3_spikes().get_value(lag);
    B_.rec4_spikes_grid_sum_ = get_rec4_spikes().get_value(lag);
    B_.I_stim_grid_sum_ = get_I_stim().get_value(lag);
    
    
    if (lag == from)
    {
    	//S_.y_[State_::time] += 1;
    	S_.time += 1;
    	}

    // NESTML generated code for the update block:
  double Idep__tmp = get_Idep() * V_.__P__Idep__Idep;
  double g_4__X__rec4_spikes__tmp = V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes * get_g_4__X__rec4_spikes() + V_.__P__g_4__X__rec4_spikes__g_4__X__rec4_spikes__d * get_g_4__X__rec4_spikes__d();
  double g_4__X__rec4_spikes__d__tmp = V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes * get_g_4__X__rec4_spikes() + V_.__P__g_4__X__rec4_spikes__d__g_4__X__rec4_spikes__d * get_g_4__X__rec4_spikes__d();
  double g_1__X__rec1_spikes__tmp = V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes * get_g_1__X__rec1_spikes() + V_.__P__g_1__X__rec1_spikes__g_1__X__rec1_spikes__d * get_g_1__X__rec1_spikes__d();
  double g_1__X__rec1_spikes__d__tmp = V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes * get_g_1__X__rec1_spikes() + V_.__P__g_1__X__rec1_spikes__d__g_1__X__rec1_spikes__d * get_g_1__X__rec1_spikes__d();
  double g_2__X__rec2_spikes__tmp = V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes * get_g_2__X__rec2_spikes() + V_.__P__g_2__X__rec2_spikes__g_2__X__rec2_spikes__d * get_g_2__X__rec2_spikes__d();
  double g_2__X__rec2_spikes__d__tmp = V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes * get_g_2__X__rec2_spikes() + V_.__P__g_2__X__rec2_spikes__d__g_2__X__rec2_spikes__d * get_g_2__X__rec2_spikes__d();
  double g_3__X__rec3_spikes__tmp = V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes * get_g_3__X__rec3_spikes() + V_.__P__g_3__X__rec3_spikes__g_3__X__rec3_spikes__d * get_g_3__X__rec3_spikes__d();
  double g_3__X__rec3_spikes__d__tmp = V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes * get_g_3__X__rec3_spikes() + V_.__P__g_3__X__rec3_spikes__d__g_3__X__rec3_spikes__d * get_g_3__X__rec3_spikes__d();
  double __t = 0;
  // numerical integration with adaptive step size control:
  // ------------------------------------------------------
  // gsl_odeiv_evolve_apply performs only a single numerical
  // integration step, starting from t and bounded by step;
  // the while-loop ensures integration over the whole simulation
  // step (0, step] if more than one integration step is needed due
  // to a small integration step size;
  // note that (t+IntegrationStep > step) leads to integration over
  // (t, step] and afterwards setting t to step, but it does not
  // enforce setting IntegrationStep to step-t; this is of advantage
  // for a consistent and efficient integration across subsequent
  // simulation intervals
  while ( __t < B_.__step )
  {
    const int status = gsl_odeiv_evolve_apply(B_.__e,
                                              B_.__c,
                                              B_.__s,
                                              &B_.__sys,              // system of ODE
                                              &__t,                   // from t
                                              B_.__step,              // to t <= step
                                              &B_.__integration_step, // integration step size
                                              S_.ode_state);          // neuronal state

    if ( status != GSL_SUCCESS )
    {
      throw nest::GSLSolverFailure( get_name(), status );
    }
  }
  /* replace analytically solvable variables with precisely integrated values  */
  S_.ode_state[State_::Idep] = Idep__tmp;
  S_.ode_state[State_::g_4__X__rec4_spikes] = g_4__X__rec4_spikes__tmp;
  S_.ode_state[State_::g_4__X__rec4_spikes__d] = g_4__X__rec4_spikes__d__tmp;
  S_.ode_state[State_::g_1__X__rec1_spikes] = g_1__X__rec1_spikes__tmp;
  S_.ode_state[State_::g_1__X__rec1_spikes__d] = g_1__X__rec1_spikes__d__tmp;
  S_.ode_state[State_::g_2__X__rec2_spikes] = g_2__X__rec2_spikes__tmp;
  S_.ode_state[State_::g_2__X__rec2_spikes__d] = g_2__X__rec2_spikes__d__tmp;
  S_.ode_state[State_::g_3__X__rec3_spikes] = g_3__X__rec3_spikes__tmp;
  S_.ode_state[State_::g_3__X__rec3_spikes__d] = g_3__X__rec3_spikes__d__tmp;
      S_.ode_state[State_::g_4__X__rec4_spikes__d] += (B_.rec4_spikes_grid_sum_) * (numerics::e / P_.tau_syn4) / (1.0);
      S_.ode_state[State_::g_1__X__rec1_spikes__d] += (B_.rec1_spikes_grid_sum_) * (numerics::e / P_.tau_syn1) / (1.0);
      S_.ode_state[State_::g_2__X__rec2_spikes__d] += (B_.rec2_spikes_grid_sum_) * (numerics::e / P_.tau_syn2) / (1.0);
      S_.ode_state[State_::g_3__X__rec3_spikes__d] += (B_.rec3_spikes_grid_sum_) * (numerics::e / P_.tau_syn3) / (1.0);
  /* Added for stochasticity*/
  const double lambda = P_.lambda_0 * std::exp( ( S_.ode_state[State_::V_m] - P_.V_th ) / P_.tau_V );
  if (V_.r>0)
  {
      V_.r -= 1;
      S_.ode_state[State_::V_m] = P_.V_reset;
  }
  else if (lambda > 0.0)
  {
      if ( ((double) rand() / (RAND_MAX))			
          < -numerics::expm1( -lambda * nest::Time::get_resolution().get_ms() ) )
        {
            {
	      V_.r = V_.RefractoryCounts;
	      S_.ode_state[State_::V_m] = P_.V_reset;
	      S_.ode_state[State_::Iadap] += P_.A2;
	      S_.ode_state[State_::Idep] = P_.A1;
	      set_spiketime(nest::Time::step(origin.get_steps()+lag+1));
	      nest::SpikeEvent se;
	      nest::kernel().event_delivery_manager.send(*this, se, lag);
  	    }

    // voltage logging
    B_.logger_.record_data(origin.get_steps() + lag);
  	}
}
}
}
// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void eglif_cond_alpha_multisyn::handle(nest::DataLoggingRequest& e)
{
  B_.logger_.handle(e);
}

void eglif_cond_alpha_multisyn::handle(nest::SpikeEvent &e)
{
  assert(e.get_delay_steps() > 0);
  assert( e.get_rport() < static_cast< int >( B_.spike_inputs_.size() ) );

  B_.spike_inputs_[ e.get_rport() ].add_value(
    e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_multiplicity() );
}

void eglif_cond_alpha_multisyn::handle(nest::CurrentEvent& e)
{
  assert(e.get_delay_steps() > 0);

  const double current = e.get_current();     // we assume that in NEST, this returns a current in pA
  const double weight = e.get_weight();
  get_I_stim().add_value(
               e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
               weight * current );
}
