/*
 *  tvb_rate_redwongwang_inh.cpp
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

#include "tvb_rate_redwongwang_inh.h"

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

nest::RecordablesMap< tvbnest::tvb_rate_redwongwang_inh > tvbnest::tvb_rate_redwongwang_inh::recordablesMap_;

namespace nest
{
    
// // Override the create() method with one call to RecordablesMap::insert_()
// // for each quantity to be recorded.
template <>
void
RecordablesMap< tvbnest::tvb_rate_redwongwang_inh >::create()
{
  insert_( names::S, &tvbnest::tvb_rate_redwongwang_inh::get_S_ );
  insert_( names::noise, &tvbnest::tvb_rate_redwongwang_inh::get_noise_ );
  insert_( names::currents, &tvbnest::tvb_rate_redwongwang_inh::get_currents_ );
  insert_( names::I_syn, &tvbnest::tvb_rate_redwongwang_inh::get_I_syn_ );
  insert_( names::rate, &tvbnest::tvb_rate_redwongwang_inh::get_r_ );
}
} // namespace

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

tvbnest::tvb_rate_redwongwang_inh::Parameters_::Parameters_()
  : g_( 0.001 ) // s
  , tau_( 10.0 )   // ms, GABA
  , w_rec_( 1.0 ) // unitless
  , W_I_( 0.7 ) // unitless
  , a_( 615.0 ) // nC^-1??
  , b_( 177.0 ) // Hz??
  , d_( 0.087 ) // s
  , J_i_( 1.0 ) // nA
  , Io_( 0.382 ) // nA
  , I_e_( 0.0 ) // nA
  , sigma_( 0.01 )
  , rectify_output_( true )
  , consistent_integration_( true )
{
  recordablesMap_.create();
}

tvbnest::tvb_rate_redwongwang_inh::State_::State_()
  : S_( 0.0 )
  , noise_( 0.0 )
  , I_syn_( 0.0 )
  , r_( 0.0 )
{
}
 

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
tvbnest::tvb_rate_redwongwang_inh::Parameters_::get(
  DictionaryDatum& d ) const
{
  def< double >( d, names::g, g_ );
  def< double >( d, names::tau, tau_ );
  def< double >( d, "w_rec", w_rec_ );
  def< double >( d, "W_I", W_I_ );
  def< double >( d, names::a, a_ );
  def< double >( d, names::b, b_ );
  def< double >( d, names::d, d_ );
  def< double >( d, "J_i", J_i_ );
  def< double >( d, "Io", Io_ );
  def< double >( d, names::I_e, I_e_ );
  def< double >( d, names::sigma, sigma_ );
  def< bool >( d, names::rectify_output, rectify_output_ );
  def< bool >( d, names::consistent_integration, consistent_integration_ );

  // Also allow old names (to not break old scripts)
  def< double >( d, names::std, sigma_ );
}

void
tvbnest::tvb_rate_redwongwang_inh::Parameters_::set(
  const DictionaryDatum& d )
{
  updateValue< double >( d, names::g, g_ );
  updateValue< double >( d, names::tau, tau_ );
  updateValue< double >( d, names::w, w_rec_ );
  updateValue< double >( d, "W_I", W_I_ );
  updateValue< double >( d, names::a, a_ );
  updateValue< double >( d, names::b, b_ );
  updateValue< double >( d, names::d, d_ );
  updateValue< double >( d, "J_i", J_i_ );
  updateValue< double >( d, "Io", Io_ );
  updateValue< double >( d, names::I_e, I_e_ );
  updateValue< double >( d, names::sigma, sigma_ );
  updateValue< bool >( d, names::rectify_output, rectify_output_ );
  updateValue< bool >( d, names::consistent_integration, consistent_integration_ );

  // Check for old names
  if ( updateValue< double >( d, names::std, sigma_ ) )
  {
    LOG( M_WARNING,
      "tvb_rate_redwongwang_inh::Parameters_::set",
      "The parameter std has been renamed to sigma. Please use the new "
      "name from now on." );
  }

  // Check for invalid parameters
  if ( g_ <= 0 )
  {
    throw nest::BadProperty( "Kinetic parameter g must be > 0." );
  }
  if ( tau_ <= 0 )
  {
    throw nest::BadProperty( "Time constant tau must be > 0." );
  }
  if ( w_rec_ < 0 )
  {
    throw nest::BadProperty( "Local synaptic recurrence weight w_rec must be >= 0." );
  }
  if ( W_I_ < 0 )
  {
    throw nest::BadProperty( "External synaptic weight w_rec must be >= 0." );
  }
  if ( a_ <= 0 )
  {
    throw nest::BadProperty( "Sigmoidal function parameter a must be > 0." );
  }
  if ( b_ < 0 )
  {
    throw nest::BadProperty( "Sigmoidal function parameter b must be >= 0." );
  }
  if ( d_ <= 0 )
  {
    throw nest::BadProperty( "Sigmoidal function parameter d must be > 0." );
  }
  if ( J_i_ < 0 )
  {
    throw nest::BadProperty( "Synaptic coupling current J_i must be >= 0." );
  }
  if ( Io_ < 0 )
  {
    throw nest::BadProperty( "Overall effective external input current Io must be >= 0." );
  }
  if ( I_e_ < 0 )
  {
    throw nest::BadProperty( "External (stimulus) input current I_e must be >= 0." );
  }
  if ( sigma_ < 0 )
  {
    throw nest::BadProperty( "Noise parameter sigma must be >= 0." );
  }
}

void
tvbnest::tvb_rate_redwongwang_inh::State_::get(
  DictionaryDatum& d ) const
{
  def< double >( d, names::S, S_ );   // Synaptic gating
  def< double >( d, names::noise, noise_ ); // Noise
  def< double >( d, names::I_syn, I_syn_ ); // I_syn
  def< double >( d, names::rate, r_ ); // Rate
}

void
tvbnest::tvb_rate_redwongwang_inh::State_::set(
  const DictionaryDatum& d )
{
  updateValue< double >( d, names::S, S_ ); // Synaptic gating
  updateValue< double >( d, names::noise, noise_ ); // Noise
  updateValue< double >( d, names::I_syn, I_syn_ ); // I_syn
  updateValue< double >( d, names::rate, r_ ); // Rate
}

tvbnest::tvb_rate_redwongwang_inh::Buffers_::Buffers_(
  tvbnest::tvb_rate_redwongwang_inh& n )
  : logger_( n )
{
}

tvbnest::tvb_rate_redwongwang_inh::Buffers_::Buffers_( const Buffers_&,
  tvbnest::tvb_rate_redwongwang_inh& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

tvbnest::tvb_rate_redwongwang_inh::tvb_rate_redwongwang_inh()
  : Archiving_Node()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
  Node::set_node_uses_wfr( kernel().simulation_manager.use_wfr() );
}

tvbnest::tvb_rate_redwongwang_inh::tvb_rate_redwongwang_inh(
    const tvbnest::tvb_rate_redwongwang_inh& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
  Node::set_node_uses_wfr( kernel().simulation_manager.use_wfr() );
}


/* ----------------------------------------------------------------
 * Sigmoid function
 * ---------------------------------------------------------------- */

double
tvbnest::tvb_rate_redwongwang_inh::sigmoid( double I_syn )
{
  double temp = P_.a_ * I_syn - P_.b_;
  return temp / (1 + exp(-P_.d_ * temp));
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
tvbnest::tvb_rate_redwongwang_inh::init_state_( const Node& proto )
{
  const tvbnest::tvb_rate_redwongwang_inh& pr = downcast< tvbnest::tvb_rate_redwongwang_inh >( proto );
  S_ = pr.S_;
}

void
tvbnest::tvb_rate_redwongwang_inh::init_buffers_()
{
  B_.delayed_currents_ex_.clear(); // includes resize
  B_.delayed_currents_in_.clear(); // includes resize

  // resize buffers
  const size_t buffer_size = kernel().connection_manager.get_min_delay();
  B_.instant_currents_ex_.resize( buffer_size, 0.0 );
  B_.instant_currents_in_.resize( buffer_size, 0.0 );
  B_.last_y_values.resize( buffer_size, 0.0 );
  B_.random_numbers.resize( buffer_size, numerics::nan );

  // initialize random numbers
  for ( unsigned int i = 0; i < buffer_size; i++ )
  {
    B_.random_numbers[ i ] =
      V_.normal_dev_( kernel().rng_manager.get_rng( get_thread() ) );
  }

  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();
}

void
tvbnest::tvb_rate_redwongwang_inh::calibrate()
{
  B_.logger_
    .init(); // ensures initialization in case mm connected after Simulate

  const double h = Time::get_resolution().get_ms();
  const double h_tau = h / P_.tau_;
  V_.W_I_Io_I_e_ = P_.W_I_ * P_.Io_ + P_.I_e_;
  V_.w_rec_J_i_ = P_.w_rec_ * P_.J_i_ ;

  if ( P_.consistent_integration_ )
  {
    // use stochastic exponential Euler method
    V_.P1_ = std::exp( - h_tau );
    V_.P2_ = -1.0 * numerics::expm1( - h_tau );
    V_.input_noise_factor_ = std::sqrt(
      -0.5 * numerics::expm1( -2. * h_tau ) );
  }
  else
  {
    // use Euler-Maruyama method
    V_.P1_ = 1;
    V_.P2_ = h_tau ;
    V_.input_noise_factor_ = std::sqrt( h_tau );
  }
}

/* ----------------------------------------------------------------
 * Update and event handling functions
 */

bool
tvbnest::tvb_rate_redwongwang_inh::update_( Time const& origin,
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
    // store rate
    new_S[ lag ] = S_.S_ ;
    // get noise
    S_.noise_ = P_.sigma_ * B_.random_numbers[ lag ];
    // propagate rate to new time step
    S_.S_ = V_.P1_ * new_S[ lag ]
      + V_.input_noise_factor_ * S_.noise_;

    double delayed_currents_in = 0;
    double delayed_currents_ex = 0;
    if ( called_from_wfr_update )
    {
      // use get_value_wfr_update to keep values in buffer
      delayed_currents_in = B_.delayed_currents_in_.get_value_wfr_update( lag );
      delayed_currents_ex = B_.delayed_currents_ex_.get_value_wfr_update( lag );
    }
    else
    {
      // use get_value to clear values in buffer after reading
      delayed_currents_in = B_.delayed_currents_in_.get_value( lag );
      delayed_currents_ex = B_.delayed_currents_ex_.get_value( lag );
    }
    double instant_currents_in = B_.instant_currents_in_[ lag ];
    double instant_currents_ex = B_.instant_currents_ex_[ lag ];

    // total synaptic current
    S_.I_syn_ = V_.W_I_Io_I_e_ - V_.w_rec_J_i_ * S_.S_ +
                delayed_currents_ex + instant_currents_ex +
                delayed_currents_in + instant_currents_in;

    // rate
    S_.r_ = sigmoid(S_.I_syn_);

    // Update with coupling
    S_.S_ += V_.P2_ * P_.g_ * S_.r_ ;

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
        or fabs( S_.S_ - B_.last_y_values[ lag ] ) > wfr_tol;
      // update last_y_values for next wfr iteration
      B_.last_y_values[ lag ] = S_.S_;
    }
    else
    {
      // rate logging
      B_.logger_.record_data( origin.get_steps() + lag );
    }
  }

  // Prepare for secondary data event transmission:
  // !!! transmits current (I * S), not synaptic gating (S) !!!
  for ( long temp = from; temp < to; ++temp )
  {
    new_S[ temp ] *= P_.J_i_ ;
  }

  if ( not called_from_wfr_update )
  {
    // Send delay-rate-neuron-event. This only happens in the final iteration
    // to avoid accumulation in the buffers of the receiving neurons.
    DelayedRateConnectionEvent drve;
    drve.set_coeffarray(new_S);
    kernel().event_delivery_manager.send_secondary( *this, drve );

    // clear last_y_values
    std::vector< double >( buffer_size, 0.0 ).swap( B_.last_y_values );

    // modifiy new_S for rate-neuron-event as proxy for next min_delay
    for ( long temp = from; temp < to; ++temp )
    {
      new_S[ temp ] = P_.J_i_ * S_.S_; // !!! transmits current (I * S), not synaptic gating (S) !!!
    }

    // create new random numbers
    B_.random_numbers.resize( buffer_size, numerics::nan );
    for ( unsigned int i = 0; i < buffer_size; i++ )
    {
      B_.random_numbers[ i ] =
        V_.normal_dev_( kernel().rng_manager.get_rng( get_thread() ) );
    }
  }

  // Send rate-neuron-event
  InstantaneousRateConnectionEvent rve;
  rve.set_coeffarray(new_S);
  kernel().event_delivery_manager.send_secondary( *this, rve );

  // Reset variables
  std::vector< double >( buffer_size, 0.0 ).swap( B_.instant_currents_ex_ );
  std::vector< double >( buffer_size, 0.0 ).swap( B_.instant_currents_in_ );

  return wfr_tol_exceeded;
}

void
tvbnest::tvb_rate_redwongwang_inh::handle(
  InstantaneousRateConnectionEvent& e )
{
  const double weight = e.get_weight();

  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();
  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( it != e.end() )
  {
    if ( weight >= 0.0 )
    {
      B_.instant_currents_ex_[ i ] += weight * e.get_coeffvalue( it );
    }
    else
    {
      B_.instant_currents_in_[ i ] += weight * e.get_coeffvalue( it );
    }
    i++;
  }
}

void
tvbnest::tvb_rate_redwongwang_inh::handle(
  DelayedRateConnectionEvent& e )
{
  const double weight = e.get_weight();
  const long delay = e.get_delay_steps();

  size_t i = 0;
  std::vector< unsigned int >::iterator it = e.begin();
  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( it != e.end() )
  {
    if ( weight >= 0.0 )
    {
      B_.delayed_currents_ex_.add_value(
        delay + i, weight * e.get_coeffvalue( it ) );
    }
    else
    {
      B_.delayed_currents_in_.add_value(
        delay + i, weight * e.get_coeffvalue( it ) );
    }
    ++i;
  }
}

void
tvbnest::tvb_rate_redwongwang_inh::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}
