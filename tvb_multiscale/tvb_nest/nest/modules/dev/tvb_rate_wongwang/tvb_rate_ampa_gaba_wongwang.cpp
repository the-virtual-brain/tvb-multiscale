/*
 *  tvb_rate_ampa_gaba_wongwang.cpp
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

#include "tvb_rate_ampa_gaba_wongwang.h"

// C++ includes:
#include <cmath> // in case we need isnan() // fabs
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

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

using namespace nest;

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< tvbnest::tvb_rate_ampa_gaba_wongwang > tvbnest::tvb_rate_ampa_gaba_wongwang::recordablesMap_;

namespace nest
{
    
// // Override the create() method with one call to RecordablesMap::insert_()
// // for each quantity to be recorded.
template <>
void
RecordablesMap< tvbnest::tvb_rate_ampa_gaba_wongwang >::create()
{
  insert_( names::S, &tvbnest::tvb_rate_ampa_gaba_wongwang::get_S_ );
  insert_( "noise_S", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_noise_S_ );
  insert_( names::V_m, &tvbnest::tvb_rate_ampa_gaba_wongwang::get_V_m_ );
  insert_( "noise_V_m", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_noise_V_m_ );
  insert_( "s_AMPA_ext", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_s_AMPA_ext );
  insert_( "s_AMPA_rec", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_s_AMPA_rec );
  insert_( "s_NMDA", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_s_NMDA );
  insert_( "s_GABA", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_s_GABA );
  insert_( "I_leak", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_I_leak);
  insert_( "I_AMPA_ext", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_I_AMPA_ext );
  insert_( "I_AMPA_rec", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_I_AMPA_rec );
  insert_( "I_NMDA", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_I_NMDA );
  insert_( "I_GABA", &tvbnest::tvb_rate_ampa_gaba_wongwang::get_I_GABA );
  insert_( names::spike, &tvbnest::tvb_rate_ampa_gaba_wongwang::get_spike_ );
  insert_( names::I_syn, &tvbnest::tvb_rate_ampa_gaba_wongwang::get_I_syn_ );
}
} // namespace

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

tvbnest::tvb_rate_ampa_gaba_wongwang::Parameters_::Parameters_()
  : V_th_( -50.0 ) // mV
  , V_reset_( -55.0 ) // mV
  , E_L_( -70.0 ) // mV
  , E_ex_( 0.0 ) // mV
  , E_in_( -70.0 ) // mV
  , t_ref_( 2.0 )   // ms
  , tau_syn_( 2.0 )   // ms, 10.0 ms for GABA
  , C_m_( 500.0 ) // pF
  , g_L_( 25.0 ) // nS
  , g_AMPA_ext_( 3.37 ) // nS
  , g_AMPA_rec_( 0.065 ) // nS
  , g_NMDA_( 0.2 ) // nS
  , g_GABA_( 10.94 ) // nS
  , beta_( 0.062 ) // real unitless
  , lamda_NMDA_( 0.28 ) // real unitless
  , I_e_( 0.0 ) // pA
  , spike_amplitude_( 2.0 ) // real unitless default set equal to tau_syn
  , sigma_( 0.01 )
  , sigma_S_( 1.0 )
  , sigma_V_m_( 1.0 )
  , rectify_output_( true )
  , consistent_integration_( true )
{
  recordablesMap_.create();
}

tvbnest::tvb_rate_ampa_gaba_wongwang::State_::State_()
  : S_( 0.0 )
  , noise_S_( 0.0 )
  , V_m_( -70.0 )
  , noise_V_m_( 0.0 )
  , s_AMPA_ext_( 0.0 )
  , s_AMPA_rec_( 0.0 )
  , s_NMDA_( 0.0 )
  , s_GABA_( 0.0 )
  , I_leak_( 0.0 )
  , I_AMPA_ext_( 0.0 )
  , I_AMPA_rec_( 0.0 )
  , I_NMDA_( 0.0 )
  , I_GABA_( 0.0 )
  , spike_( 0.0 )
  , r( 0 )
{
}
 

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
tvbnest::tvb_rate_ampa_gaba_wongwang::Parameters_::get(
  DictionaryDatum& d ) const
{
  def< double >( d, names::V_th, V_th_ );
  def< double >( d, names::V_reset, V_reset_ );
  def< double >( d, names::E_L, E_L_ );
  def< double >( d, names::E_ex, E_ex_ );
  def< double >( d, names::E_in, E_in_ );
  def< double >( d, names::t_ref, t_ref_ );
  def< double >( d, names::tau_syn, tau_syn_ );
  def< double >( d, names::C_m, C_m_ );
  def< double >( d, names::g_L, g_L_ );
  def< double >( d, "g_AMPA_ext", g_AMPA_ext_ );
  def< double >( d, "g_AMPA_rec", g_AMPA_rec_ );
  def< double >( d, names::g_NMDA, g_NMDA_ );
  def< double >( d, "g_GABA", g_GABA_ );
  def< double >( d, names::beta, beta_ );
  def< double >( d, "lamda_NMDA", lamda_NMDA_ );
  def< double >( d, names::sigma, sigma_ );
  def< double >( d, "sigma_S", sigma_S_ );
  def< double >( d, "sigma_V_m", sigma_V_m_ );
  def< double >( d, names::I_e, I_e_ );
  def< double >( d, "spike_amplitude", spike_amplitude_ );
  def< bool >( d, names::rectify_output, rectify_output_ );
  def< bool >( d, names::consistent_integration, consistent_integration_ );

  // Also allow old names (to not break old scripts)
  def< double >( d, names::std, sigma_ );

  DictionaryDatum __receptor_type = new Dictionary();
  ( *__receptor_type )[ "AMPA_EXT" ] = AMPA_EXT;
  ( *__receptor_type )[ "AMPA_REC" ] = AMPA_REC;
  ( *__receptor_type )[ "NMDA" ] = NMDA;
  ( *__receptor_type )[ "GABA" ] = GABA;

  ( *d )[ "receptor_types" ] = __receptor_type;
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::Parameters_::set(
  const DictionaryDatum& d )
{
  updateValue< double >( d, names::V_th, V_th_ );
  updateValue< double >( d, names::V_reset, V_reset_ );
  updateValue< double >( d, names::E_L, E_L_ );
  updateValue< double >( d, names::E_ex, E_ex_ );
  updateValue< double >( d, names::E_in, E_in_ );
  updateValue< double >( d, names::t_ref, t_ref_ );
  updateValue< double >( d, names::tau_syn, tau_syn_ );
  updateValue< double >( d, names::C_m, C_m_ );
  updateValue< double >( d, names::g_L, g_L_ );
  updateValue< double >( d, "g_AMPA_ext", g_AMPA_ext_ );
  updateValue< double >( d, "g_AMPA_rec", g_AMPA_rec_ );
  updateValue< double >( d, names::g_NMDA, g_NMDA_ );
  updateValue< double >( d, "g_GABA", g_GABA_ );
  updateValue< double >( d, names::beta, beta_ );
  updateValue< double >( d, "lamda_NMDA", lamda_NMDA_ );
  updateValue< double >( d, names::I_e, I_e_ );
  updateValue< double >( d, "spike_amplitude", spike_amplitude_ );
  updateValue< double >( d, names::sigma, sigma_ );
  updateValue< double >( d, "sigma_S", sigma_S_ );
  updateValue< double >( d, "sigma_V_m", sigma_V_m_ );
  updateValue< bool >( d, names::rectify_output, rectify_output_ );
  updateValue< bool >( d, names::consistent_integration, consistent_integration_ );

  // Check for old names
  if ( updateValue< double >( d, names::std, sigma_ ) )
  {
    LOG( M_WARNING,
      "tvb_rate_ampa_gaba_wongwang::Parameters_::set",
      "The parameter std has been renamed to sigma. Please use the new "
      "name from now on." );
  }

  // Check for invalid parameters
  if ( t_ref_ <= 0 )
  {
    throw nest::BadProperty( "Time constant t_ref must be > 0." );
  }
  if ( tau_syn_ <= 0 )
  {
    throw nest::BadProperty( "Time constant tau_syn must be > 0." );
  }
  if ( C_m_ < 0 )
  {
    throw nest::BadProperty( "Membrane capacitance C_m must be >= 0." );
  }
  if ( g_L_ <= 0 )
  {
    throw nest::BadProperty( "Conductance parameter g_L must be > 0." );
  }
  if ( g_AMPA_ext_ <= 0 )
  {
    throw nest::BadProperty( "Conductance parameter g_AMPA_ext must be > 0." );
  }
  if ( g_AMPA_rec_ <= 0 )
  {
    throw nest::BadProperty( "Conductance parameter g_AMPA_rec must be > 0." );
  }
  if ( g_NMDA_ <= 0 )
  {
    throw nest::BadProperty( "Conductance parameter g_NMDA must be > 0." );
  }
  if ( g_GABA_ <= 0 )
  {
    throw nest::BadProperty( "Conductance parameter g_GABA must be > 0." );
  }
  if ( beta_ < 0 )
  {
    throw nest::BadProperty( "beta must be >= 0." );
  }
  if ( lamda_NMDA_ <= 0 )
  {
    throw nest::BadProperty( "lamda_NMDA must be > 0." );
  }
  if ( I_e_ < 0 )
  {
    throw nest::BadProperty( "Overall effective external input current I_e must be >= 0." );
  }
  if ( spike_amplitude_ < 0 )
  {
    throw nest::BadProperty( "Spike amplitude spike_amplitude_ must be >= 0." );
  }
  if ( sigma_ < 0 )
  {
    throw nest::BadProperty( "Noise parameter sigma must be >= 0." );
  }

  if ( sigma_S_ < 0 )
  {
    throw nest::BadProperty( "Noise parameter sigma_S must be >= 0." );
  }

  if ( sigma_V_m_ < 0 )
  {
    throw nest::BadProperty( "Noise parameter sigma_V_m must be >= 0." );
  }
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::State_::get(
  DictionaryDatum& d ) const
{
  def< double >( d, names::S, S_ );
  def< double >( d, "noise_S", noise_S_ );
  def< double >( d, names::V_m, V_m_ );
  def< double >( d, "noise_V_m", noise_V_m_ );
  def< double >( d, "s_AMPA_ext", s_AMPA_ext_ );
  def< double >( d, "s_AMPA_rec", s_AMPA_rec_ );
  def< double >( d, "s_NMDA", s_NMDA_ );
  def< double >( d, "s_GABA", s_GABA_ );
  def< double >( d, "I_leak", I_leak_ );
  def< double >( d, "I_AMPA_ext", I_AMPA_ext_ );
  def< double >( d, "I_AMPA_rec", I_AMPA_rec_ );
  def< double >( d, "I_NMDA", I_NMDA_ );
  def< double >( d, "I_GABA", I_GABA_ );
  def< double >( d, names::spike, spike_ );
  def< long >( d, "r", r );
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::State_::set(
  const DictionaryDatum& d )
{
  updateValue< double >( d, names::S, S_ );
  updateValue< double >( d, "noise_S", noise_S_ );
  updateValue< double >( d, names::V_m, V_m_ );
  updateValue< double >( d, "noise_V_m", noise_V_m_ );
  updateValue< double >( d, "s_AMPA_ext", s_AMPA_ext_ );
  updateValue< double >( d, "s_AMPA_rec", s_AMPA_rec_ );
  updateValue< double >( d, "s_NMDA", s_NMDA_ );
  updateValue< double >( d, "s_GABA", s_GABA_ );
  updateValue< double >( d, "I_leak", I_leak_ );
  updateValue< double >( d, "I_AMPA_ext", I_AMPA_ext_ );
  updateValue< double >( d, "I_AMPA_rec", I_AMPA_rec_ );
  updateValue< double >( d, "I_NMDA", I_NMDA_ );
  updateValue< double >( d, "I_GABA", I_GABA_ );
  updateValue< double >( d, names::spike, spike_ );
  updateValue< long >( d, "r", r );
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

tvbnest::tvb_rate_ampa_gaba_wongwang::tvb_rate_ampa_gaba_wongwang()
  : Archiving_Node()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
  Node::set_node_uses_wfr( kernel().simulation_manager.use_wfr() );
}

tvbnest::tvb_rate_ampa_gaba_wongwang::tvb_rate_ampa_gaba_wongwang(
    const tvbnest::tvb_rate_ampa_gaba_wongwang& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
  Node::set_node_uses_wfr( kernel().simulation_manager.use_wfr() );
}

tvbnest::tvb_rate_ampa_gaba_wongwang::Buffers_::Buffers_(
  tvbnest::tvb_rate_ampa_gaba_wongwang& n )
  : logger_( n ),
    delayed_S_inputs_( std::vector< nest::RingBuffer >( SUP_RECEPTOR - 1 ) ),
    instant_S_inputs_( std::vector<std::vector< double >> ( SUP_RECEPTOR - 1 ) ),
    last_y_values( std::vector<std::vector< double >> ( 2 ) ),
    random_numbers( std::vector<std::vector< double >> ( 2 ) ){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

tvbnest::tvb_rate_ampa_gaba_wongwang::Buffers_::Buffers_( const Buffers_&,
  tvbnest::tvb_rate_ampa_gaba_wongwang& n )
  : logger_( n ),
    delayed_S_inputs_( std::vector< nest::RingBuffer >( SUP_RECEPTOR - 1 ) ),
    instant_S_inputs_( std::vector<std::vector< double >> ( SUP_RECEPTOR - 1 ) ),
    last_y_values( std::vector<std::vector< double >> ( 2 ) ),
    random_numbers( std::vector<std::vector< double >> ( 2 ) ){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}


/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
tvbnest::tvb_rate_ampa_gaba_wongwang::init_state_( const Node& proto )
{
  const tvbnest::tvb_rate_ampa_gaba_wongwang& pr = downcast< tvbnest::tvb_rate_ampa_gaba_wongwang >( proto );
  S_ = pr.S_;
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::init_buffers_()
{
  get_delayed_S_AMPA_ext().clear(); //includes resize
  get_delayed_S_AMPA_rec().clear(); //includes resize
  get_delayed_S_NMDA().clear(); //includes resize
  get_delayed_S_GABA().clear(); //includes resize

  // resize buffers
  const size_t buffer_size = kernel().connection_manager.get_min_delay();
  get_instant_S_AMPA_ext().resize( buffer_size, 0.0 ); //includes resize
  get_instant_S_AMPA_rec().resize( buffer_size, 0.0 ); //includes resize
  get_instant_S_NMDA().resize( buffer_size, 0.0 ); //includes resize
  get_instant_S_GABA().resize( buffer_size, 0.0 ); //includes resize

  for ( unsigned int j = 0; j < 2; j++ )
  {
      B_.last_y_values[ j ].resize( buffer_size, 0.0 );
      B_.random_numbers[ j ].resize( buffer_size, numerics::nan );
      // initialize random numbers
      for ( unsigned int i = 0; i < buffer_size; i++ )
      {
        B_.random_numbers[ j ][ i ] =
          V_.normal_dev_( kernel().rng_manager.get_rng( get_thread() ) );
      }
  }
  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::calibrate()
{
  B_.logger_
    .init(); // ensures initialization in case mm connected after Simulate

  const double h = Time::get_resolution().get_ms();
  // tau_V_m_ = P_.C_m_/ P_.g_L_;
  const std::vector<double> tau = {P_.tau_syn_, P_.C_m_/ P_.g_L_};
  // tau_V_m_ = P_.C_m_/ P_.g_L_;

  V_.g_L_E_L_ = P_.g_L_ * P_.E_L_ ;

  if ( P_.consistent_integration_ )
  {
    // use stochastic exponential Euler method
    for ( unsigned int j = 0; j < 2; j++ )
    {
        double h_tau = h / tau[ j ];
        V_.P1_[ j ] = std::exp( - h_tau );
        V_.P2_[ j ] = -1.0 * numerics::expm1( - h_tau );
        V_.input_noise_factor_[ j ] = std::sqrt(
          -0.5 * numerics::expm1( -2. * h_tau ) ); //??
    }
  }
  else
  {
    // use Euler-Maruyama method
    for ( unsigned int j = 0; j < 2; j++ )
    {
        double h_tau = h / tau[ j ];
        V_.P1_[ j ] = 1;
        V_.P2_[ j ] = h_tau;
        V_.input_noise_factor_[ j ] = std::sqrt( h_tau );
    }
  }
  V_.sigma_ = {P_.sigma_ * P_.sigma_S_, P_.sigma_ * P_.sigma_V_m_};

}

/* ----------------------------------------------------------------
 * Update and event handling functions
 */

bool
tvbnest::tvb_rate_ampa_gaba_wongwang::update_( Time const& origin,
  const long from,
  const long to,
  const bool called_from_wfr_update )
{
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  const size_t buffer_size = kernel().connection_manager.get_min_delay();
  const double wfr_tol = kernel().simulation_manager.get_wfr_tol();
  bool wfr_tol_exceeded = false;

  // allocate memory to store currents to be sent by rate events
  std::vector< double > new_S( buffer_size, 0.0 );

  for ( long lag = from; lag < to; ++lag )
  {
    // store current state
    new_S[ lag ] = S_.S_ ;

    // get noise
    S_.noise_S_ = V_.sigma_[0] * B_.random_numbers[0][ lag ];
    S_.noise_V_m_ = V_.sigma_[1] * B_.random_numbers[1][ lag ];
    // propagate to new time step
    S_.S_ = V_.P1_[0] * new_S[ lag ]
          - V_.P2_[0] * new_S[ lag ]
          + V_.input_noise_factor_[0] * S_.noise_S_ ;

    // Compute coupling
    double delayed_S_AMPA_ext = 0;
    double delayed_S_AMPA_rec = 0;
    double delayed_S_NMDA = 0;
    double delayed_S_GABA = 0;

    if ( called_from_wfr_update )
    {
      // use get_value_wfr_update to keep values in buffer
      delayed_S_AMPA_ext = get_delayed_S_AMPA_ext().get_value_wfr_update( lag );
      delayed_S_AMPA_rec = get_delayed_S_AMPA_rec().get_value_wfr_update( lag );
      delayed_S_NMDA = get_delayed_S_NMDA().get_value_wfr_update( lag );
      delayed_S_GABA = get_delayed_S_GABA().get_value_wfr_update( lag );
    }
    else
    {
      // use get_value to clear values in buffer after reading
      delayed_S_AMPA_ext = get_delayed_S_AMPA_ext().get_value( lag );
      delayed_S_AMPA_rec = get_delayed_S_AMPA_rec().get_value( lag );
      delayed_S_NMDA = get_delayed_S_NMDA().get_value( lag );
      delayed_S_GABA = get_delayed_S_GABA().get_value( lag );
    }
    double instant_S_AMPA_ext = get_instant_S_AMPA_ext()[ lag ];
    double instant_S_AMPA_rec = get_instant_S_AMPA_rec()[ lag ];
    double instant_S_NMDA = get_instant_S_NMDA()[ lag ];
    double instant_S_GABA = get_instant_S_GABA()[ lag ];

    // total synaptic current
    S_.s_AMPA_ext_ = delayed_S_AMPA_ext + instant_S_AMPA_ext;
    S_.s_AMPA_rec_ = delayed_S_AMPA_rec + instant_S_AMPA_rec;
    S_.s_NMDA_ = delayed_S_NMDA + instant_S_NMDA;
    S_.s_GABA_ = delayed_S_GABA + instant_S_GABA;

    double Vex = S_.V_m_ - P_.E_ex_;
    S_.I_leak_ = - P_.g_L_ * S_.V_m_ + V_.g_L_E_L_;
    S_.I_AMPA_ext_ = - P_.g_AMPA_ext_ * Vex * S_.s_AMPA_ext_;
    S_.I_AMPA_rec_ = - P_.g_AMPA_rec_ * Vex * S_.s_AMPA_rec_;
    S_.I_NMDA_ = - P_.g_NMDA_ * Vex * S_.s_NMDA_ / (1 + std::exp( - P_.beta_ *  S_.V_m_));
    S_.I_GABA_ = - P_.g_GABA_ * (S_.V_m_ - P_.E_in_) * S_.s_GABA_;

    // Check for refractoriness or spike emission

    // TODO: Decide whether we should add noise to V_reset
    S_.spike_ = 0.0;
    if (S_.r != 0) {
        S_.r = S_.r - 1;
        S_.V_m_ = P_.V_reset_;
    } else if (S_.V_m_ >= P_.V_th_) {
        S_.r = V_.RefractoryCounts;
        S_.V_m_ = P_.V_reset_;
        S_.spike_ = 1.0;
        S_.S_ += V_.P2_[1] * P_.spike_amplitude_ * S_.spike_;
    } else {
        // Update with coupling
        S_.V_m_ = V_.P1_[1] * S_.V_m_
                + V_.P2_[1] * ( - S_.V_m_ + P_.E_L_ +
                                ( S_.I_AMPA_ext_ + S_.I_AMPA_rec_ + S_.I_NMDA_ + S_.I_GABA_ + P_.I_e_ ) / P_.g_L_ )
                + V_.input_noise_factor_[1] * S_.noise_V_m_;
    }

    if ( P_.rectify_output_ )
    {
        // TODO: Discuss about this...
        if ( S_.S_ < 0 )
        {
          S_.S_ = 0.0 ; // or 0.0 - S_.S_ ?
        }
        else if ( S_.S_ > 1 )
        {
          S_.S_ = 1.0 ; // or = 1.0 - S_.S_?
        }
    }

    if ( called_from_wfr_update )
    {
      // check if deviation from last iteration exceeds wfr_tol
      wfr_tol_exceeded = wfr_tol_exceeded
        or fabs( S_.S_ - B_.last_y_values[ 0 ][ lag ] ) > wfr_tol
        or fabs( S_.V_m_ - B_.last_y_values[ 1 ][ lag ] ) > wfr_tol;
      // update last_y_values for next wfr iteration
      B_.last_y_values[ 0 ][ lag ] = S_.S_;
      B_.last_y_values[ 1 ][ lag ] = S_.V_m_;
    }
    else
    {
      // rate logging
      B_.logger_.record_data( origin.get_steps() + lag );
    }
  }

  // Prepare for secondary data event transmission:
  if ( not called_from_wfr_update )
  {
    // Send delay-rate-neuron-event. This only happens in the final iteration
    // to avoid accumulation in the buffers of the receiving neurons.
    DelayedRateConnectionEvent drve;
    drve.set_coeffarray(new_S);
    kernel().event_delivery_manager.send_secondary( *this, drve );

    for ( unsigned int j = 0; j < 2; j++ )
    {
        // clear last_y_values
        std::vector< double >( buffer_size, 0.0 ).swap( B_.last_y_values[ j ] );

        // create new random numbers
        B_.random_numbers[ j ].resize( buffer_size, numerics::nan );
        for ( unsigned int i = 0; i < buffer_size; i++ )
        {
          B_.random_numbers[ j ][ i ] =
            V_.normal_dev_( kernel().rng_manager.get_rng( get_thread() ) );
        }
    }
  }

  // Send rate-neuron-event
  InstantaneousRateConnectionEvent rve;
  rve.set_coeffarray(new_S);
  kernel().event_delivery_manager.send_secondary( *this, rve );

  // Reset variables
  std::vector< double >( buffer_size, 0.0 ).swap( get_instant_S_AMPA_ext() );
  std::vector< double >( buffer_size, 0.0 ).swap( get_instant_S_AMPA_rec() );
  std::vector< double >( buffer_size, 0.0 ).swap( get_instant_S_NMDA() );
  std::vector< double >( buffer_size, 0.0 ).swap( get_instant_S_GABA() );

  return wfr_tol_exceeded;
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::handle(
  InstantaneousRateConnectionEvent& e )
{
  double weight = e.get_weight();
  if (weight < 0) {
    weight = -weight;  // ensure conductance is positive
  }

  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();
  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( it != e.end() )
  {
    B_.instant_S_inputs_[ e.get_rport() ][ i ] += weight * e.get_coeffvalue( it );
    i++;
  }
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::handle(
  DelayedRateConnectionEvent& e )
{
  double weight = e.get_weight();
  if (weight < 0) {
    weight = -weight;  // ensure conductance is positive
  }
  const long delay = e.get_delay_steps();

  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();
  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( it != e.end() )
  {
    B_.delayed_S_inputs_[ e.get_rport() ].add_value(
        delay + i, weight * e.get_coeffvalue( it ) );
    ++i;
  }
}

void
tvbnest::tvb_rate_ampa_gaba_wongwang::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}
