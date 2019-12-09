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
 */

#include "iaf_cond_deco2014.h"

#ifdef HAVE_GSL

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


namespace nest // template specialization must be placed in namespace
{

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
DynamicRecordablesMap< iaf_cond_deco2014 >::create(
  iaf_cond_deco2014& host )
{
  // use standard names wherever you can for consistency!
  insert( names::V_m,
    host.get_data_access_functor( iaf_cond_deco2014::State_::V_M ) );

  insert( "s_AMPA",
    host.get_data_access_functor( iaf_cond_deco2014::State_::S_AMPA ) );

  insert( "s_GABA",
    host.get_data_access_functor( iaf_cond_deco2014::State_::S_GABA ) );

  insert( "x_NMDA",
    host.get_data_access_functor( iaf_cond_deco2014::State_::X_NMDA ) );

  insert( "s_NMDA",
    host.get_data_access_functor( iaf_cond_deco2014::State_::S_NMDA ) );

  host.insert_conductance_recordables();
}

Name
iaf_cond_deco2014::get_g_receptor_name( size_t receptor )
{
  std::stringstream receptor_name;
  receptor_name << "s_AMPA_ext_" << receptor;
  return Name( receptor_name.str() );
}

void
iaf_cond_deco2014::insert_conductance_recordables( size_t first )
{
  for ( size_t receptor = first; receptor < P_.w_E_ext.size(); ++receptor )
  {
    size_t elem = iaf_cond_deco2014::State_::S_AMPA_EXT + receptor;
    recordablesMap_.insert(
      get_g_receptor_name( receptor ), this->get_data_access_functor( elem ) );
  }
}

DataAccessFunctor< iaf_cond_deco2014 >
iaf_cond_deco2014::get_data_access_functor( size_t elem )
{
  return DataAccessFunctor< iaf_cond_deco2014 >( *this, elem );
}

/* ----------------------------------------------------------------
 * Right-hand side function
 * ---------------------------------------------------------------- */

extern "C" int
iaf_cond_deco2014_dynamics( double,
  const double y[],
  double f[],
  void* pnode )
{
  // y[] is the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.y[].

  typedef nest::iaf_cond_deco2014::State_ S;

  // get access to node so we can almost work as in a member function
  assert( pnode );
  const nest::iaf_cond_deco2014& node =
    *( reinterpret_cast< nest::iaf_cond_deco2014* >( pnode ) );

  const bool is_refractory = node.S_.r_ > 0;

  // Clamp membrane potential to V_reset while refractory
  const double& V_m = is_refractory ? node.P_.V_reset : y[ S::V_M ];

  // The following code is verbose for the sake of clarity. We assume that a
  // good compiler will optimize the verbosity away ...
  const double V_m_minus_E_ex = V_m - node.P_.E_ex;
  const double I_L = node.P_.g_L * V_m - node.V_.g_L_E_L;
  const double I_AMPA = node.V_.w_E_g_AMPA * V_m_minus_E_ex * y[S::S_AMPA];
  const double I_GABA = node.V_.w_I_g_GABA_A * (V_m - node.P_.E_in) * y[S::S_GABA];
  const double I_NMDA = node.V_.w_E_g_NMDA /
                            ( 1 + node.P_.lambda_NMDA * std::exp(node.V_.minus_beta * V_m ) )*
                            V_m_minus_E_ex * y[S::S_NMDA];

  double I_AMPA_ext = 0.0;
  for ( size_t i = 0; i < node.P_.n_receptors(); ++i )
  {
    I_AMPA_ext += node.V_.w_E_ext_g_AMPA_ext[i] * V_m_minus_E_ex * y[S::S_AMPA_EXT + i];
    f[S::S_AMPA_EXT + i] = y[S::S_AMPA_EXT + i] / node.V_.minus_tau_decay_AMPA;
  }

  // dv/dt
  f[ S::V_M ] =
    is_refractory ? 0 : ( - (I_L + I_AMPA + I_GABA + I_NMDA + I_AMPA_ext)
                          + node.P_.I_e + node.B_.I_stim ) / node.P_.C_m;

  f[S::S_AMPA] = y[S::S_AMPA] / node.V_.minus_tau_decay_AMPA ;
  f[S::S_GABA] = y[S::S_GABA] / node.V_.minus_tau_decay_GABA_A ;
  f[S::X_NMDA] = y[S::X_NMDA] / node.V_.minus_tau_rise_NMDA ;
  f[S::S_NMDA] = y[S::S_NMDA] / node.V_.minus_tau_decay_NMDA +
                       node.P_.alpha * y[S::X_NMDA] * (1 - y[S::S_NMDA] / node.P_.N_E);


  return GSL_SUCCESS;
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
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
    , N_E( 1 )                      // positive integer
    , alpha( 0.5 )                  // in kHz
    , beta( 0.062 )                 // real number
    , lambda_NMDA( 0.28 )           // real number
    , I_e( 0.0 )                    // in pA
    , gsl_error_tol( 1e-6 )
    , has_connections_( false )
{
}

iaf_cond_deco2014::State_::State_( const Parameters_& p )
  : y_( STATE_VECTOR_MIN_SIZE, 0.0 )
  , r_( 0 )
{
  y_[ 0 ] = p.E_L;
}

iaf_cond_deco2014::State_::State_( const State_& s )
  : r_( s.r_ )
{
  y_ = s.y_;
}

iaf_cond_deco2014::State_& iaf_cond_deco2014::State_::
operator=( const State_& s )
{
  assert( this != &s ); // would be bad logical error in program

  y_ = s.y_;
  r_ = s.r_;
  return *this;
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
iaf_cond_deco2014::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::V_th, V_th );
  def< double >( d, names::V_reset, V_reset );
  def< double >( d, names::E_L, E_L );
  def< double >( d, names::E_ex, E_ex );
  def< double >( d, names::E_in, E_in );

  def< double >( d, names::t_ref, t_ref );
  def< double >( d, names::tau_decay_AMPA, tau_decay_AMPA );
  def< double >( d, names::tau_rise_NMDA, tau_rise_NMDA );
  def< double >( d, names::tau_decay_NMDA, tau_decay_NMDA );
  def< double >( d, names::tau_decay_GABA_A, tau_decay_GABA_A );

  def< double >( d, names::C_m, C_m );

  def< double >( d, names::g_L, g_L );
  def< double >( d, "g_AMPA_ext", g_AMPA_ext );
  def< double >( d, names::g_AMPA, g_AMPA );
  def< double >( d, names::g_NMDA, g_NMDA );
  def< double >( d, names::g_GABA_A, g_GABA_A );

  def< double >( d, "w_E", w_E );
  def< double >( d, "w_I", w_I );
  ArrayDatum w_E_ext_ad( w_E_ext );
  def< ArrayDatum >( d, "w_E_ext", w_E_ext_ad );

  def< long >( d, "N_E", N_E );

  def< double >( d, names::alpha, alpha );
  def< double >( d, names::beta, beta );
  def< double >( d, "lambda_NMDA", lambda_NMDA );

  def< double >( d, names::I_e, I_e );

  def< size_t >( d, names::n_receptors, n_receptors() );

  def< double >( d, names::gsl_error_tol, gsl_error_tol );
  def< bool >( d, names::has_connections, has_connections_ );
}

void
iaf_cond_deco2014::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::V_th, V_th );
  updateValue< double >( d, names::V_reset, V_reset );
  updateValue< double >( d, names::E_L, E_L );
  updateValue< double >( d, names::E_ex, E_ex );
  updateValue< double >( d, names::E_in, E_in );

  if ( t_ref < 0 )
  {
    throw BadProperty("Refractory time cannot be negative" );
  }
  updateValue< double >( d, names::t_ref, t_ref );
  if ( tau_decay_AMPA <= 0 )
  {
    throw BadProperty("AMPA decay time constant must be strictly positive" );
  }
  updateValue< double >( d, names::tau_decay_AMPA, tau_decay_AMPA );
  if ( tau_rise_NMDA <= 0 )
  {
    throw BadProperty("NMDA rise time constant must be strictly positive" );
  }
  updateValue< double >( d, names::tau_rise_NMDA, tau_rise_NMDA );
  if ( tau_decay_NMDA <= 0 )
  {
    throw BadProperty("NMDA decay time constant must be strictly positive" );
  }
  updateValue< double >( d, names::tau_decay_NMDA, tau_decay_NMDA );
  if ( tau_decay_GABA_A <= 0 )
  {
    throw BadProperty("GABA_A decay time constant must be strictly positive" );
  }
  updateValue< double >( d, names::tau_decay_GABA_A, tau_decay_GABA_A );

  if ( C_m <= 0 )
  {
    throw BadProperty("Membrane capacitance must be strictly positive" );
  }
  updateValue< double >( d, names::C_m, C_m );

  if ( g_L < 0 )
  {
    throw BadProperty("Leak conductance cannot be negative" );
  }
  updateValue< double >( d, names::g_L, g_L );
  if ( g_AMPA_ext < 0 )
  {
    throw BadProperty("External synbapses' AMPA conductance cannot be negative" );
  }
  updateValue< double >( d, "g_AMPA_ext", g_AMPA_ext );
  if ( g_AMPA < 0 )
  {
    throw BadProperty("AMPA conductance cannot be negative" );
  }
  updateValue< double >( d, names::g_AMPA, g_AMPA );
  if ( g_NMDA < 0 )
  {
    throw BadProperty("NMDA conductance cannot be negative" );
  }
  updateValue< double >( d, names::g_NMDA, g_NMDA );
  if ( g_GABA_A < 0 )
  {
    throw BadProperty("GABA_A conductance cannot be negative" );
  }
  updateValue< double >( d, names::g_GABA_A, g_GABA_A );

  if ( w_E < 0 )
  {
    throw BadProperty("Population recursive excitatory connection weight cannot be negative" );
  }
  updateValue< double >( d, "w_E", w_E );
  if ( w_I < 0 )
  {
    throw BadProperty("Population recursive inhibitory connection weight cannot be negative" );
  }
  updateValue< double >( d, "w_I", w_I );

  const size_t old_n_receptors = n_receptors();
  bool w_E_ext_flag =
    updateValue< std::vector< double > >( d, "w_E_ext", w_E_ext );

  if ( w_E_ext_flag )
  { // receptor arrays have been modified

    if ( w_E_ext.size() < old_n_receptors && has_connections_ )
    {
      throw BadProperty(
        "The neuron has connections, therefore the number of ports cannot be "
        "reduced." );
    }
    for ( size_t i = 0; i < w_E_ext.size(); ++i )
    {
      if ( w_E_ext[ i ] < 0 )
      {
        throw BadProperty("Population external excitatory connection weights cannot be negative" );
      }
    }
  }

  if ( N_E <= 0 )
  {
    throw BadProperty("The number of excitatory neurons in the population must be a positive integer" );
  }
  updateValue< long >( d, "N_E", N_E );

  if ( alpha < 0 )
  {
    throw BadProperty("alpha cannot be negative" );
  }
  updateValue< double >( d, names::alpha, alpha );
  if ( beta < 0 )
  {
    throw BadProperty( "beta cannot be negative" );
  }
  updateValue< double >( d, names::beta, beta );
  if ( lambda_NMDA < 0 )
  {
    throw BadProperty("lambda_NMDA cannot be negative" );
  }
  updateValue< double >( d, "lambda_NMDA", lambda_NMDA );

  updateValue< double >( d, names::I_e, I_e );

  updateValue< double >( d, names::gsl_error_tol, gsl_error_tol );

  if ( V_reset >= V_th )
  {
    throw BadProperty( "V_reset < V_th required." );
  }

  if ( gsl_error_tol <= 0. )
  {
    throw BadProperty( "The gsl_error_tol must be strictly positive." );
  }
}

void
iaf_cond_deco2014::State_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::V_m, y_[ V_M ] );

  std::vector< double >* s_AMPA_ext = new std::vector< double >();

  for ( size_t i = 0;
        i < ( y_.size() - State_::NUMBER_OF_FIXED_STATES_ELEMENTS );
        ++i )
  {
    s_AMPA_ext->push_back( y_[ State_::S_AMPA_EXT + i ] );
  }

  ( *d )[ "s_AMPA_ext" ] = DoubleVectorDatum( s_AMPA_ext );

  def< double >( d, "s_AMPA", y_[ S_AMPA ] );
  def< double >( d, "s_GABA", y_[ S_GABA ] );
  def< double >( d, "s_NMDA", y_[ S_NMDA ] );
  def< double >( d, "x_NMDA", y_[ X_NMDA ] );

}

void
iaf_cond_deco2014::State_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::V_m, y_[ V_M ] );
  updateValue< double >( d, "s_AMPA", y_[ S_AMPA ] );
  updateValue< double >( d, "s_GABA", y_[ S_GABA ] );
  updateValue< double >( d, "s_NMDA", y_[ S_NMDA ] );
  updateValue< double >( d, "x_NMDA", y_[ X_NMDA ] );

// TODO: find out how to do this setting...
//  std::vector< double >* s_AMPA_ext = new std::vector< double >();
//
//  for ( size_t i = 0;
//        i < ( y_.size() - State_::NUMBER_OF_FIXED_STATES_ELEMENTS );
//        ++i )
//  {
//    s_AMPA_ext->push_back( y_[ State_::S_AMPA_EXT + i  ] );
//  }
//  updateValue< double >( d, "s_AMPA_ext", DoubleVectorDatum( s_AMPA_ext ) );
}

iaf_cond_deco2014::Buffers_::Buffers_(
  iaf_cond_deco2014& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
  , step_( Time::get_resolution().get_ms() )
  , IntegrationStep_( std::min( 0.01, step_ ) )
  , I_stim( 0.0 )
{
}

iaf_cond_deco2014::Buffers_::Buffers_( const Buffers_& b,
  iaf_cond_deco2014& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
  , step_( b.step_ )
  , IntegrationStep_( b.IntegrationStep_ )
  , I_stim( b.I_stim )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

iaf_cond_deco2014::iaf_cond_deco2014()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create( *this );
}

iaf_cond_deco2014::iaf_cond_deco2014(
  const iaf_cond_deco2014& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
  recordablesMap_.create( *this );
}

iaf_cond_deco2014::~iaf_cond_deco2014()
{
  // GSL structs may not have been allocated, so we need to protect destruction
  if ( B_.s_ )
  {
    gsl_odeiv_step_free( B_.s_ );
  }
  if ( B_.c_ )
  {
    gsl_odeiv_control_free( B_.c_ );
  }
  if ( B_.e_ )
  {
    gsl_odeiv_evolve_free( B_.e_ );
  }
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
iaf_cond_deco2014::init_state_( const Node& proto )
{
  const iaf_cond_deco2014& pr =
    downcast< iaf_cond_deco2014 >( proto );
  S_ = pr.S_;
}

void
iaf_cond_deco2014::init_buffers_()
{
  B_.spikesExc.clear();   // includes resize
  B_.spikesInh.clear(); //includes resize
  B_.currents.clear(); // includes resize
  B_.spikesExc_ext.resize( P_.n_receptors() );

  Archiving_Node::clear_history();

  B_.logger_.reset();

  B_.step_ = Time::get_resolution().get_ms();

  // We must integrate this model with high-precision to obtain decent results
  B_.IntegrationStep_ = B_.step_;

  if ( B_.c_ == 0 )
  {
    B_.c_ = gsl_odeiv_control_yp_new( P_.gsl_error_tol, P_.gsl_error_tol );
  }
  else
  {
    gsl_odeiv_control_init(
      B_.c_, P_.gsl_error_tol, P_.gsl_error_tol, 0.0, 1.0 );
  }

  // Stepping function and evolution function are allocated in calibrate()

  B_.sys_.function = iaf_cond_deco2014_dynamics;
  B_.sys_.jacobian = NULL;
  B_.sys_.params = reinterpret_cast< void* >( this );
  // B_.sys_.dimension is assigned in calibrate()
  B_.I_stim = 0.0;

}

void
iaf_cond_deco2014::calibrate()
{
  // ensures initialization in case mm connected after Simulate
  B_.logger_.init();

  V_.refractory_counts_ = Time( Time::ms( P_.t_ref ) ).get_steps();
  assert( V_.refractory_counts_
    >= 0 ); // since t_ref >= 0, this can only fail in error

  V_.g_L_E_L = P_.g_L * P_.E_L;

  V_.minus_beta = - P_.beta;

  V_.minus_tau_decay_AMPA = - P_.tau_decay_AMPA;
  V_.minus_tau_decay_GABA_A = - P_.tau_decay_GABA_A;
  V_.minus_tau_rise_NMDA = - P_.tau_rise_NMDA;
  V_.minus_tau_decay_NMDA = - P_.tau_decay_NMDA;

  V_.w_E_g_AMPA = P_.w_E * P_.g_AMPA;
  V_.w_E_g_NMDA = P_.w_E * P_.g_NMDA;
  V_.w_I_g_GABA_A = P_.w_I * P_.g_GABA_A;
  // w_E_ext_g_AMPA_ext will be initialized in the loop below:
  V_.w_E_ext_g_AMPA_ext.resize(P_.n_receptors(), 0.0);
  for ( size_t i = 0; i < P_.n_receptors(); ++i )
  {
    V_.w_E_ext_g_AMPA_ext[i] = P_.g_AMPA_ext * P_.w_E_ext[i] ;
  }

  B_.spikesExc_ext.resize( P_.n_receptors() );
  for (size_t i=0; i < P_.n_receptors(); i++){
    B_.spikesExc_ext[i].clear(); //includes resize
  }

  S_.y_.resize( State_::NUMBER_OF_FIXED_STATES_ELEMENTS + P_.n_receptors() , 0.0 );

  // reallocate instance of stepping function for ODE GSL solver
  if ( B_.s_ != 0 )
  {
    gsl_odeiv_step_free( B_.s_ );
  }
  B_.s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, S_.y_.size() );

  // reallocate instance of evolution function for ODE GSL solver
  if ( B_.e_ != 0 )
  {
    gsl_odeiv_evolve_free( B_.e_ );
  }
  B_.e_ = gsl_odeiv_evolve_alloc( S_.y_.size() );

  B_.sys_.dimension = S_.y_.size();
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */
void
iaf_cond_deco2014::update( Time const& origin,
  const long from,
  const long to )
{
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );
  assert( State_::V_M == 0 );

  double spikesExc_grid_sum_ = 0.0;

  for ( long lag = from; lag < to; ++lag ) // proceed by stepsize B_.step_
  {
    double t = 0.0; // internal time of the integration period

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

    while ( t < B_.step_ )
    {
      const int status = gsl_odeiv_evolve_apply( B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,             // system of ODE
        &t,                   // from t
        B_.step_,             // to t <= step
        &B_.IntegrationStep_, // integration step size
        &S_.y_[ 0 ] );        // neuronal state converted to double[]

      if ( status != GSL_SUCCESS )
      {
        throw GSLSolverFailure( get_name(), status );
      }

      // check for unreasonable values; we allow V_M to explode
      if ( S_.y_[ State_::V_M ] < -1e3 )
      {
        throw NumericalInstability( get_name() );
      }

      // TODO: check whether/how S_ variables should be bounded from above as well

      if ( S_.y_[ State_::S_AMPA ] < 0.0 )
      {
        S_.y_[ State_::S_AMPA ] = 0.0;
      }

      if ( S_.y_[ State_::X_NMDA ] < 0.0 )
      {
        S_.y_[ State_::X_NMDA ] = 0.0;
      }

      if ( S_.y_[ State_::S_NMDA ] < 0.0 )
      {
        S_.y_[ State_::S_NMDA ] = 0.0;
      }

      if ( S_.y_[ State_::S_GABA ] < 0.0 )
      {
        S_.y_[ State_::S_GABA ] = 0.0;
      }

      for ( size_t i = 0; i < P_.n_receptors(); ++i )
      {
        if ( S_.y_[ State_::S_AMPA_EXT + i ] < 0.0 )
        {
            S_.y_[ State_::S_AMPA + i] = 0.0;
        }
      }

      if ( S_.r_ > 0 ) // if neuron is still in refractory period
      {
        S_.y_[ State_::V_M ] = P_.V_reset; // clamp it to V_reset
      }
      else if ( S_.y_[ State_::V_M ] >= P_.V_th ) // V_m >= V_th: spike
      {
        S_.y_[ State_::V_M ] = P_.V_reset;

        /* Initialize refractory step counter.
         * - We need to add 1 to compensate for count-down immediately after
         *   while loop.
         * - If neuron has no refractory time, set to 0 to avoid refractory
         *   artifact inside while loop.
         */
        S_.r_ = V_.refractory_counts_ > 0 ? V_.refractory_counts_ + 1 : 0;

        set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
        SpikeEvent se;
        kernel().event_delivery_manager.send( *this, se, lag );
      } /* if end */
    } /* while end */

    if ( S_.r_ > 0 ) // if neuron is still in refractory period
    {
        --S_.r_;
    }

    spikesExc_grid_sum_ = B_.spikesExc.get_value(lag);

    S_.y_[State_::S_AMPA] += spikesExc_grid_sum_;

    S_.y_[State_::X_NMDA] += spikesExc_grid_sum_;

    S_.y_[State_::S_GABA] += B_.spikesInh.get_value(lag);

    for ( size_t i = 0; i < P_.n_receptors(); ++i )
    {
       // add incoming spike:
      S_.y_[ State_::S_AMPA_EXT + i ] += B_.spikesExc_ext[ i ].get_value( lag ) ;
    }
    // set new input current
    B_.I_stim = B_.currents.get_value( lag );

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );

  } // for-loop
}

void
iaf_cond_deco2014::handle( SpikeEvent& e )
{
  const long rport = e.get_rport();
  assert( ( rport >= 0 ) && ( rport <= (long) P_.n_receptors() ) );

  const double weight = e.get_weight() * e.get_multiplicity();

  // Ignore spikes if weight = 0
  if (rport == 0) {  // these are excitatory and inhibitory spikes internal to the population
    if (weight > 0) {
        B_.spikesExc.add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ), weight );
    } else if (weight < 0) {  // make sure that weight is always positive for conductance synapses
        B_.spikesInh.add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ), -weight );
    }
  } else {  // these are excitatory AMPA spikes coming from other populations
    if (weight < 0) {
        throw nest::BadProperty("Synaptic weights for AMPA_ext synapses must be positive");
    } else if (weight > 0) {
        B_.spikesExc_ext[rport - 1].add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ), weight );
    }
  }

}

void
iaf_cond_deco2014::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  const double I = e.get_current();
  const double w = e.get_weight();

  // add weighted current; HEP 2002-10-04
  B_.currents.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    w * I );
}

void
iaf_cond_deco2014::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

void
iaf_cond_deco2014::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d );         // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status( d );

  /*
   * Here is where we must update the recordablesMap_ if new receptors
   * are added!
   */
  if ( ptmp.w_E_ext.size() > P_.w_E_ext.size() ) // Number of receptors increased
  {
    for ( size_t receptor = P_.w_E_ext.size(); receptor < ptmp.w_E_ext.size();
          ++receptor )
    {
      size_t elem = iaf_cond_deco2014::State_::S_AMPA_EXT + receptor;
      recordablesMap_.insert(
        get_g_receptor_name( receptor ), get_data_access_functor( elem ) );
    }
  }
  else if ( ptmp.w_E_ext.size() < P_.w_E_ext.size() )
  { // Number of receptors decreased
    for ( size_t receptor = ptmp.w_E_ext.size(); receptor < P_.w_E_ext.size();
          ++receptor )
    {
      recordablesMap_.erase( get_g_receptor_name( receptor ) );
    }
  }

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace nest

#endif // HAVE_GSL
