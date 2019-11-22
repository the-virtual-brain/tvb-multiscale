/*
*  iaf_cond_ampa_gaba_nmda_deco2014.cpp
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
*  2019-09-25 19:43:44.478471
*/

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

#include "iaf_cond_ampa_gaba_nmda_deco2014.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<iaf_cond_ampa_gaba_nmda_deco2014> iaf_cond_ampa_gaba_nmda_deco2014::recordablesMap_;

namespace nest
{
  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
  template <> void RecordablesMap<iaf_cond_ampa_gaba_nmda_deco2014>::create(){
  // use standard names whereever you can for consistency!  

  insert_("I_leak", &iaf_cond_ampa_gaba_nmda_deco2014::get_I_leak);

  insert_("I_AMPA_ext", &iaf_cond_ampa_gaba_nmda_deco2014::get_I_AMPA_ext);

  insert_("I_AMPA_rec", &iaf_cond_ampa_gaba_nmda_deco2014::get_I_AMPA_rec);

  insert_("I_NMDA", &iaf_cond_ampa_gaba_nmda_deco2014::get_I_NMDA);

  insert_("I_GABA", &iaf_cond_ampa_gaba_nmda_deco2014::get_I_GABA);

  insert_("s_AMPA_ext", &iaf_cond_ampa_gaba_nmda_deco2014::get_s_AMPA_ext);

  insert_("s_AMPA_rec", &iaf_cond_ampa_gaba_nmda_deco2014::get_s_AMPA_rec);

  insert_("x_NMDA", &iaf_cond_ampa_gaba_nmda_deco2014::get_x_NMDA);

  insert_("s_NMDA", &iaf_cond_ampa_gaba_nmda_deco2014::get_s_NMDA);

  insert_("s_GABA", &iaf_cond_ampa_gaba_nmda_deco2014::get_s_GABA);

  insert_("V_m", &iaf_cond_ampa_gaba_nmda_deco2014::get_V_m);
  }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization is of variables
 * is a part of the iaf_cond_ampa_gaba_nmda_deco2014's constructor.
 * ---------------------------------------------------------------- */
iaf_cond_ampa_gaba_nmda_deco2014::Parameters_::Parameters_(){}

iaf_cond_ampa_gaba_nmda_deco2014::State_::State_(){}

/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

iaf_cond_ampa_gaba_nmda_deco2014::Buffers_::Buffers_(iaf_cond_ampa_gaba_nmda_deco2014 &n):
  logger_(n), spike_inputs_( std::vector< nest::RingBuffer >( SUP_SPIKE_RECEPTOR - 1 ) ), __s( 0 ), __c( 0 ), __e( 0 ){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

iaf_cond_ampa_gaba_nmda_deco2014::Buffers_::Buffers_(const Buffers_ &, iaf_cond_ampa_gaba_nmda_deco2014 &n):
  logger_(n), spike_inputs_( std::vector< nest::RingBuffer >( SUP_SPIKE_RECEPTOR - 1 ) ), __s( 0 ), __c( 0 ), __e( 0 ){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
iaf_cond_ampa_gaba_nmda_deco2014::iaf_cond_ampa_gaba_nmda_deco2014():Archiving_Node(), P_(), S_(), B_(*this)
{
  recordablesMap_.create();
  // use a default `good` enough value for the absolute error.
  // it cab be adjusted via `SetStatus`

  P_.V_th = (-50.0); // as mV

  P_.V_reset = (-55.0); // as mV

  P_.E_L = (-70.0); // as mV
  
  P_.E_ex = 0.0; // as mV

  P_.E_in = (-70.0); // as mV
  
  P_.t_ref = 2.0; // as ms

  P_.tau_AMPA = 2.0; // as ms

  P_.tau_NMDA_rise = 2.0; // as ms

  P_.tau_NMDA_decay = 100.0; // as ms

  P_.tau_GABA = 10.0; // as ms

  P_.C_m = 500.0; // as pF

  P_.g_m = 25.0; // as nS
  
  P_.g_AMPA_ext = 3.37; // as nS
  
  P_.g_AMPA_rec = 0.065; // as nS
  
  P_.g_NMDA = 0.2; // as nS
  
  P_.g_GABA = 10.94; // as nS

  P_.alpha = 0.5; // as kHz
  
  P_.beta = 0.062; // as real
  
  P_.lamda_NMDA = 0.28; // as real
  
  P_.I_e = 0.0; // as pA
  
  S_.r = 0; // as integer
  
  S_.I_leak = P_.g_m * (S_.ode_state[State_::V_m] - P_.E_L); // as pA
  
  S_.I_AMPA_ext = P_.g_AMPA_ext * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_AMPA_ext]; // as pA
  
  S_.I_AMPA_rec = P_.g_AMPA_rec * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_AMPA_rec]; // as pA
  
  S_.I_NMDA = P_.g_NMDA * (S_.ode_state[State_::V_m] - P_.E_ex) * S_.ode_state[State_::s_NMDA]; // as pA
  
  S_.I_GABA = P_.g_GABA * (S_.ode_state[State_::V_m] - P_.E_in) * S_.ode_state[State_::s_GABA]; // as pA
  
  S_.ode_state[State_::s_AMPA_ext] = 0.0; // as nS
  
  S_.ode_state[State_::s_AMPA_rec] = 0.0; // as nS
  
  S_.ode_state[State_::x_NMDA] = 0.0; // as nS / ms
  
  S_.ode_state[State_::s_NMDA] = 0.0; // as nS
  
  S_.ode_state[State_::s_GABA] = 0.0; // as nS
  
  S_.ode_state[State_::V_m] = P_.E_L; // as mV

  P_.__gsl_error_tol = 1e-3;
}

iaf_cond_ampa_gaba_nmda_deco2014::iaf_cond_ampa_gaba_nmda_deco2014(const iaf_cond_ampa_gaba_nmda_deco2014& __n):
  Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this){

  P_.V_th = __n.P_.V_th;
  P_.V_reset = __n.P_.V_reset;
  P_.E_L = __n.P_.E_L;
  P_.E_ex = __n.P_.E_ex;
  P_.E_in = __n.P_.E_in;
  P_.t_ref = __n.P_.t_ref;
  P_.tau_AMPA = __n.P_.tau_AMPA;
  P_.tau_NMDA_rise = __n.P_.tau_NMDA_rise;
  P_.tau_NMDA_decay = __n.P_.tau_NMDA_decay;
  P_.tau_GABA = __n.P_.tau_GABA;
  P_.C_m = __n.P_.C_m;
  P_.g_m = __n.P_.g_m;
  P_.g_AMPA_ext = __n.P_.g_AMPA_ext;
  P_.g_AMPA_rec = __n.P_.g_AMPA_rec;
  P_.g_NMDA = __n.P_.g_NMDA;
  P_.g_GABA = __n.P_.g_GABA;
  P_.alpha = __n.P_.alpha;
  P_.beta = __n.P_.beta;
  P_.lamda_NMDA = __n.P_.lamda_NMDA;
  P_.I_e = __n.P_.I_e;
  
  S_.r = __n.S_.r;
  
  S_.ode_state[State_::s_AMPA_ext] = __n.S_.ode_state[State_::s_AMPA_ext];
  S_.ode_state[State_::s_AMPA_rec] = __n.S_.ode_state[State_::s_AMPA_rec];
  S_.ode_state[State_::x_NMDA] = __n.S_.ode_state[State_::x_NMDA];
  S_.ode_state[State_::s_NMDA] = __n.S_.ode_state[State_::s_NMDA];
  S_.ode_state[State_::s_GABA] = __n.S_.ode_state[State_::s_GABA];
  S_.ode_state[State_::V_m] = __n.S_.ode_state[State_::V_m];
  
  V_.RefractoryCounts = __n.V_.RefractoryCounts;
  
}

iaf_cond_ampa_gaba_nmda_deco2014::~iaf_cond_ampa_gaba_nmda_deco2014(){
  // GSL structs may not have been allocated, so we need to protect destruction
  if (B_.__s)
    gsl_odeiv_step_free( B_.__s );
  if (B_.__c)
    gsl_odeiv_control_free( B_.__c );
  if (B_.__e)
    gsl_odeiv_evolve_free( B_.__e );
}

/* ----------------------------------------------------------------
* Node initialization functions
* ---------------------------------------------------------------- */

void iaf_cond_ampa_gaba_nmda_deco2014::init_state_(const Node& proto){
  const iaf_cond_ampa_gaba_nmda_deco2014& pr = downcast<iaf_cond_ampa_gaba_nmda_deco2014>(proto);
  S_ = pr.S_;
}



extern "C" inline int iaf_cond_ampa_gaba_nmda_deco2014_dynamics(double, const double ode_state[], double f[], void* pnode){
  typedef iaf_cond_ampa_gaba_nmda_deco2014::State_ State_;
  // get access to node so we can almost work as in a member function
  assert( pnode );
  const iaf_cond_ampa_gaba_nmda_deco2014& node = *( reinterpret_cast< iaf_cond_ampa_gaba_nmda_deco2014* >( pnode ) );

  // ode_state[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.ode_state[].
  double I_leak = node.get_g_m() * (ode_state[State_::V_m] - node.get_E_L());
  double I_AMPA_ext = node.get_g_AMPA_ext() * (ode_state[State_::V_m] - node.get_E_ex()) * ode_state[State_::s_AMPA_ext];
  double I_AMPA_rec = node.get_g_AMPA_rec() * (ode_state[State_::V_m] - node.get_E_ex()) * ode_state[State_::s_AMPA_rec];
  double I_NMDA = node.get_g_NMDA() * (ode_state[State_::V_m] - node.get_E_ex()) * ode_state[State_::s_NMDA];
  double I_GABA = node.get_g_GABA() * (ode_state[State_::V_m] - node.get_E_in()) * ode_state[State_::s_GABA];
      
  
  f[State_::s_AMPA_ext] = -(ode_state[State_::s_AMPA_ext]) / node.get_tau_AMPA();
  f[State_::s_AMPA_rec] = -(ode_state[State_::s_AMPA_rec]) / node.get_tau_AMPA();
  f[State_::x_NMDA] = -(ode_state[State_::x_NMDA]) / node.get_tau_NMDA_rise();
  f[State_::s_NMDA] = -(ode_state[State_::s_NMDA]) / node.get_tau_NMDA_decay() + node.get_alpha() * ode_state[State_::x_NMDA] * (1 - ode_state[State_::s_NMDA]);
  f[State_::s_GABA] = -(ode_state[State_::s_GABA]) / node.get_tau_GABA();
  f[State_::V_m] = (-(I_leak) - I_AMPA_ext - I_AMPA_rec - I_NMDA / (1 + node.get_lamda_NMDA() * std::exp(-(node.get_beta()) * (ode_state[State_::V_m]))) - I_GABA + node.get_I_e() + node.B_.currents_grid_sum_) / node.get_C_m();
  return GSL_SUCCESS;
}



void iaf_cond_ampa_gaba_nmda_deco2014::init_buffers_(){
  get_spikesExc_AMPA_ext().clear(); //includes resize
  get_spikesExc_AMPA_rec().clear(); //includes resize
  get_spikesExc_NMDA().clear(); //includes resize
  get_spikesInh_GABA().clear(); //includes resize
  get_currents().clear(); //includes resize
  
  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();
  
  if ( B_.__s == 0 ){
    B_.__s = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, 6 );
  } else {
    gsl_odeiv_step_reset( B_.__s );
  }

  if ( B_.__c == 0 ){
    B_.__c = gsl_odeiv_control_y_new( P_.__gsl_error_tol, 0.0 );
  } else {
    gsl_odeiv_control_init( B_.__c, P_.__gsl_error_tol, 0.0, 1.0, 0.0 );
  }

  if ( B_.__e == 0 ){
    B_.__e = gsl_odeiv_evolve_alloc( 6 );
  } else {
    gsl_odeiv_evolve_reset( B_.__e );
  }

  B_.__sys.function = iaf_cond_ampa_gaba_nmda_deco2014_dynamics;
  B_.__sys.jacobian = NULL;
  B_.__sys.dimension = 6;
  B_.__sys.params = reinterpret_cast< void* >( this );
  B_.__step = nest::Time::get_resolution().get_ms();
  B_.__integration_step = nest::Time::get_resolution().get_ms();
}

void iaf_cond_ampa_gaba_nmda_deco2014::calibrate(){
  B_.logger_.init();
  
  
  V_.RefractoryCounts =nest::Time(nest::Time::ms((double) (P_.t_ref))).get_steps();
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

/*
 *
 */
void iaf_cond_ampa_gaba_nmda_deco2014::update(nest::Time const & origin,const long from, const long to){
  double __t = 0;

  for ( long lag = from ; lag < to ; ++lag ) {
    B_.spikesExc_AMPA_ext_grid_sum_ = get_spikesExc_AMPA_ext().get_value(lag);
    B_.spikesExc_AMPA_rec_grid_sum_ = get_spikesExc_AMPA_rec().get_value(lag);
    B_.spikesExc_NMDA_grid_sum_ = get_spikesExc_NMDA().get_value(lag);
    B_.spikesInh_GABA_grid_sum_ = get_spikesInh_GABA().get_value(lag);
    B_.currents_grid_sum_ = get_currents().get_value(lag);
      
    

    __t = 0;
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

      if ( status != GSL_SUCCESS ) {
        throw nest::GSLSolverFailure( get_name(), status );
      }
    }
    



    if (S_.r!=0) {
      

      S_.r = S_.r - 1;
      

      S_.ode_state[State_::V_m] = P_.V_reset;
    }else if(S_.ode_state[State_::V_m]>=P_.V_th) {
      

      S_.r = V_.RefractoryCounts;
      

      S_.ode_state[State_::V_m] = P_.V_reset;
      
      set_spiketime(nest::Time::step(origin.get_steps()+lag+1));
      nest::SpikeEvent se;
      nest::kernel().event_delivery_manager.send(*this, se, lag);
    } /* if end */

    

    S_.ode_state[State_::s_AMPA_ext] += B_.spikesExc_AMPA_ext_grid_sum_;
    

    S_.ode_state[State_::s_AMPA_rec] += B_.spikesExc_AMPA_rec_grid_sum_;
    

    S_.ode_state[State_::x_NMDA] += B_.spikesExc_NMDA_grid_sum_;
    

    S_.ode_state[State_::s_GABA] += B_.spikesInh_GABA_grid_sum_;

    // voltage logging
    B_.logger_.record_data(origin.get_steps()+lag);
  }

}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void iaf_cond_ampa_gaba_nmda_deco2014::handle(nest::DataLoggingRequest& e){
  B_.logger_.handle(e);
}


void iaf_cond_ampa_gaba_nmda_deco2014::handle(nest::SpikeEvent &e){
  assert(e.get_delay_steps() > 0);
  assert( e.get_rport() < static_cast< int >( B_.spike_inputs_.size() ) );

  double weight = e.get_weight();
  if (weight < 0) {
    weight = -weight;  // ensure conductance is positive
  }

  B_.spike_inputs_[ e.get_rport() ].add_value(
    e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ),
    weight * e.get_multiplicity() );
  
}

void iaf_cond_ampa_gaba_nmda_deco2014::handle(nest::CurrentEvent& e){
  assert(e.get_delay_steps() > 0);

  const double current=e.get_current();
  const double weight=e.get_weight();

  // add weighted current; HEP 2002-10-04
  get_currents().add_value(
               e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
               weight * current );
  
}
