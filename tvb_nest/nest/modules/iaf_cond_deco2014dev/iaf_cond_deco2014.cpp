/*
*  iaf_cond_deco2014.cpp
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
*  2019-11-14 14:51:07.265855
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

#include "iaf_cond_deco2014.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<iaf_cond_deco2014> iaf_cond_deco2014::recordablesMap_;

namespace nest
{
  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
  template <> void RecordablesMap<iaf_cond_deco2014>::create(){
  // use standard names whereever you can for consistency!  

  insert_(names::V_m, &iaf_cond_deco2014::get_V_m);

  insert_("s_AMPA", &iaf_cond_deco2014::get_s_AMPA);

  insert_("s_GABA", &iaf_cond_deco2014::get_s_GABA);

  insert_("s_NMDA", &iaf_cond_deco2014::get_s_NMDA);

  insert_("x_NMDA", &iaf_cond_deco2014::get_x_NMDA);

  insert_("s_AMPA_ext", &iaf_cond_deco2014::get_s_AMPA_ext_sum);

  insert_("I_L", &iaf_cond_deco2014::get_I_L);

  insert_("I_AMPA", &iaf_cond_deco2014::get_I_AMPA);

  insert_("I_GABA", &iaf_cond_deco2014::get_I_GABA);

  insert_("I_NMDA", &iaf_cond_deco2014::get_I_NMDA);

  insert_("I_AMPA_ext", &iaf_cond_deco2014::get_I_AMPA_ext_sum);

  }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization of variables
 * is a part of the iaf_cond_deco2014's constructor.
 * ---------------------------------------------------------------- */
iaf_cond_deco2014::Parameters_::Parameters_()
    : V_th( -50.0 )                 // in mV
    , V_reset( -55.0 )              // in mV
    , E_L( -70.0 )                  // in mV
    , E_ex( 0.0 )                   // in mV
    , E_in( -70.0 )                 // in mV
    , t_ref( 2.0 )                  // in ms
    , tau_decay_AMPA( 2.0 )         // in ms
    , tau_rise_NMDA( 2.0 )          // in ms
    , tau_decay_NMDA( 100.0 )       // in ms
    , tau_decay_GABA_A( 10.0 )      // in ms
    , C_m( 500.0 )                  // in pF
    , g_L( 25.0 )                   // in nS
    , g_AMPA_ext( 3.37 )            // in nS
    , g_AMPA( 0.065 )               // in nS
    , g_NMDA( 0.2 )                 // in nS
    , g_GABA_A( 10.94 )             // in nS
    , w_E( 1.4 )                    // real number
    , w_I( 1.0 )                    // real number
    , w_E_ext(1, 0.1)               // vector of real numbers
    , alpha( 0.5 )                  // in kHz
    , beta( 0.062 )                 // real number
    , lambda_NMDA( 0.28 )           // real number
    , I_e( 0.0 )                    // in pA
    , __gsl_error_tol( 1e-3 )
    , has_connections_( false )
{
}

iaf_cond_deco2014::State_::State_( const Parameters_& p )
  : ode_state( STATE_VEC_SIZE + NUM_STATE_ELEMENTS_PER_RECEPTOR * p.n_receptors(), 0.0 )
  , r( 0 )
{
  ode_state[ V_m ] = p.E_L;
  ode_state[ s_AMPA ] = 0.0;
  ode_state[ s_GABA ] = 0.0;
  ode_state[ x_NMDA ] =0.0;
  ode_state[ s_NMDA ] = 0.0;
  for ( long i = 0 ; i < p.n_receptors() ; ++i ) {
    ode_state[ s_AMPA_ext + i ] = 0.0;
  }
}

iaf_cond_deco2014::State_::State_( const State_& s  )
    : ode_state( s.ode_state )
    , r ( s.r ){
}

iaf_cond_deco2014::State_&
    iaf_cond_deco2014::State_::
  operator=( const State_& s )
{
    assert( this != &s ); // would be bad logical error in program
    ode_state = s.ode_state;
    r = s.r;
}


/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

iaf_cond_deco2014::Buffers_::Buffers_(iaf_cond_deco2014 &n):
  logger_(n), __s( 0 ), __c( 0 ), __e( 0 ){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

iaf_cond_deco2014::Buffers_::Buffers_(const Buffers_ &, iaf_cond_deco2014 &n):
  logger_(n), __s( 0 ), __c( 0 ), __e( 0 ){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
iaf_cond_deco2014::iaf_cond_deco2014():Archiving_Node(), P_(), S_(P_), B_(*this)
{
  recordablesMap_.create();
}

iaf_cond_deco2014::iaf_cond_deco2014(const iaf_cond_deco2014& __n):
  Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this){
  P_.V_th = __n.P_.V_th;
  P_.V_reset = __n.P_.V_reset;
  P_.E_L = __n.P_.E_L;
  P_.E_ex = __n.P_.E_ex;
  P_.E_in = __n.P_.E_in;
  P_.t_ref = __n.P_.t_ref;
  P_.tau_decay_AMPA = __n.P_.tau_decay_AMPA;
  P_.tau_rise_NMDA = __n.P_.tau_rise_NMDA;
  P_.tau_decay_NMDA = __n.P_.tau_decay_NMDA;
  P_.tau_decay_GABA_A = __n.P_.tau_decay_GABA_A;
  P_.C_m = __n.P_.C_m;
  P_.g_L = __n.P_.g_L;
  P_.g_AMPA_ext = __n.P_.g_AMPA_ext;
  P_.g_AMPA = __n.P_.g_AMPA;
  P_.g_NMDA = __n.P_.g_NMDA;
  P_.g_GABA_A = __n.P_.g_GABA_A;
  P_.w_E = __n.P_.w_E;
  P_.w_I = __n.P_.w_I;
  const long old_n_receptors = P_.n_receptors();
  if ( (long) __n.P_.w_E_ext.size() < old_n_receptors && P_.has_connections_ )  {
        throw nest::BadProperty(
        "The neuron has connections, therefore the number of ports cannot be "
        "reduced." );
  }
  P_.w_E_ext = __n.P_.w_E_ext;

  P_.alpha = __n.P_.alpha;
  P_.beta = __n.P_.beta;
  P_.lambda_NMDA = __n.P_.lambda_NMDA;
  P_.I_e = __n.P_.I_e;
  
  S_.r = __n.S_.r;

  S_.ode_state = __n.S_.ode_state;

}

iaf_cond_deco2014::~iaf_cond_deco2014(){ 
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

void iaf_cond_deco2014::init_state_(const Node& proto){
  const iaf_cond_deco2014& pr = downcast<iaf_cond_deco2014>(proto);
  S_ = pr.S_;
}



extern "C" inline int iaf_cond_deco2014_dynamics(double, const double ode_state[], double f[], void* pnode){
  typedef iaf_cond_deco2014::State_ State_;
  // get access to node so we can almost work as in a member function
  assert( pnode );
  const iaf_cond_deco2014& node = *( reinterpret_cast< iaf_cond_deco2014* >( pnode ) );

  // The following code is verbose for the sake of clarity. We assume that a
  // good compiler will optimize the verbosity away ...
  const double V_m_E_ex = ode_state[State_::V_m] - node.get_E_ex();
  const double I_L = node.get_g_L() * ode_state[State_::V_m] - node.V_.g_L_E_L;
  const double I_AMPA = node.get_g_AMPA() * V_m_E_ex * ode_state[State_::s_AMPA];
  const double I_GABA = node.get_g_GABA_A() * (ode_state[State_::V_m] - node.get_E_in()) * ode_state[State_::s_GABA];
  const double I_NMDA = node.get_g_NMDA() /
                            ( 1 + node.get_lambda_NMDA() * std::exp(node.V_.minus_beta * ode_state[State_::V_m]) ) *
                            V_m_E_ex * ode_state[State_::s_NMDA];

  double I_AMPA_ext = 0.0;
  for ( long i = 0 ; i < node.get_n_receptors() ; ++i ) {
        I_AMPA_ext +=
            node.get_g_AMPA() * V_m_E_ex * ode_state[State_::s_AMPA_ext + i];
        f[State_::s_AMPA_ext + i] =
            node.V_.minus_w_E_ext_tau_decay_AMPA[i] * ode_state[State_::s_AMPA_ext + i] +
            node.B_.spikesExc_ext_grid_sum_[i] ;

  }

  // ode_state[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.ode_state[].
  f[State_::V_m] = ( - ( I_L + I_AMPA_ext + I_AMPA + I_GABA + I_NMDA )+
                     node.get_I_e() + node.B_.currents_grid_sum_ )/ node.get_C_m();

  f[State_::s_AMPA] = node.V_.minus_w_E_tau_decay_AMPA * ode_state[State_::s_AMPA] + node.B_.spikesExc_grid_sum_ ;
  f[State_::s_GABA] = node.V_.minus_w_I_tau_decay_GABA_A * ode_state[State_::s_GABA] + node.B_.spikesInh_grid_sum_ ;
  f[State_::x_NMDA] = node.V_.minus_w_E_tau_rise_NMDA * ode_state[State_::x_NMDA] + node.B_.spikesExc_grid_sum_ ;
  f[State_::s_NMDA] = -ode_state[State_::s_NMDA] / node.get_tau_decay_NMDA() +
                       node.get_alpha() * ode_state[State_::x_NMDA] * (1 - ode_state[State_::s_NMDA]);

  return GSL_SUCCESS;
}



void iaf_cond_deco2014::init_buffers_(){
  get_spikesExc().clear(); //includes resize
  get_spikesInh().clear(); //includes resize
  get_currents().clear(); //includes resize
  get_spikesExc_ext().resize( P_.n_receptors() );
  for (long i=0; i < P_.n_receptors(); i++){
    get_spikesExc_ext()[i].clear(); //includes resize
  }

  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();

  const int state_size =
        State_::NUMBER_OF_FIXED_STATES_ELEMENTS + State_::NUM_STATE_ELEMENTS_PER_RECEPTOR * P_.n_receptors();

  B_.__step = nest::Time::get_resolution().get_ms();
  B_.__integration_step = nest::Time::get_resolution().get_ms();

  if ( B_.__s == 0 ){
    B_.__s = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, state_size );
  } else {
    gsl_odeiv_step_reset( B_.__s );
  }

  if ( B_.__c == 0 ){
    B_.__c = gsl_odeiv_control_y_new( P_.__gsl_error_tol, 0.0 );
  } else {
    gsl_odeiv_control_init( B_.__c, P_.__gsl_error_tol, 0.0, 1.0, 0.0 );
  }

  if ( B_.__e == 0 ){
    B_.__e = gsl_odeiv_evolve_alloc( state_size );
  } else {
    gsl_odeiv_evolve_reset( B_.__e );
  }

  B_.__sys.function = iaf_cond_deco2014_dynamics;
  B_.__sys.jacobian = NULL;
  B_.__sys.dimension = state_size;
  B_.__sys.params = reinterpret_cast< void* >( this );
}

void iaf_cond_deco2014::calibrate(){
  // std::cout << "Calibrating model...";

  B_.logger_.init();

  V_.RefractoryCounts =nest::Time(nest::Time::ms((double) (P_.t_ref))).get_steps();

  V_.g_L_E_L = P_.g_L * P_.E_L;
  V_.minus_beta = - P_.beta;
  V_.minus_w_E_tau_decay_AMPA = -P_.w_E / P_.tau_decay_AMPA;
  V_.minus_w_I_tau_decay_GABA_A = -P_.w_I / P_.tau_decay_GABA_A;
  V_.minus_w_E_tau_rise_NMDA = -P_.w_E / P_.tau_rise_NMDA;

  B_.spikesExc_ext_grid_sum_.resize(P_.n_receptors(), 0.0);
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

/*
 *
 */
void iaf_cond_deco2014::update(nest::Time const & origin,const long from, const long to){
  double __t = 0;

  for ( long lag = from ; lag < to ; ++lag ) {
    
    for (long i=0; i < P_.n_receptors(); i++){
      B_.spikesExc_ext_grid_sum_[i] = get_spikesExc_ext()[i].get_value(lag);
    }
    B_.spikesExc_grid_sum_ = get_spikesExc().get_value(lag);
    B_.spikesInh_grid_sum_ = get_spikesInh().get_value(lag);
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
                                                &S_.ode_state[0]);          // neuronal state converted to double[]

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

    S_.ode_state[State_::s_AMPA] += B_.spikesExc_grid_sum_;

    S_.ode_state[State_::x_NMDA] += B_.spikesExc_grid_sum_;

    S_.ode_state[State_::s_GABA] += B_.spikesInh_grid_sum_;

    for (long i=0; i < P_.n_receptors(); i++) {
        S_.ode_state[State_::s_AMPA_ext + i] += B_.spikesExc_ext_grid_sum_[i];
    }

    // voltage logging
    B_.logger_.record_data(origin.get_steps()+lag);
  }
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void iaf_cond_deco2014::handle(nest::DataLoggingRequest& e){
  B_.logger_.handle(e);
}


void iaf_cond_deco2014::handle(nest::SpikeEvent &e){
  assert(e.get_delay_steps() > 0);

  const long rport = e.get_rport();
  assert( ( rport >= 0 ) && ( rport <= P_.n_receptors() ) );

  const double weight = e.get_weight() * e.get_multiplicity();

  if (rport == 0) {
    if (weight >= 0) {
        B_.spikesExc.add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ), weight );
    } else {  // make sure that weight is always positive for conductance synapses
        B_.spikesInh.add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ), -weight );
    }
  } else {
    if (weight < 0) {
        throw nest::BadProperty("Synaptic weights for AMPA_ext synapses must be positive");
    } else {
        B_.spikesExc_ext[rport - 1].add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ), weight );
    }
  }

}

void iaf_cond_deco2014::handle(nest::CurrentEvent& e){
  assert(e.get_delay_steps() > 0);

  const double current=e.get_current();
  const double weight=e.get_weight();

  // add weighted current; HEP 2002-10-04
  get_currents().add_value(
               e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
               weight * current );
  
}
