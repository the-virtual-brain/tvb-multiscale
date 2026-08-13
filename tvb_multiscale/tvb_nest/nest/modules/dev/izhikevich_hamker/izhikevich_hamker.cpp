/*
 *  izhikevich_hamker.cpp
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

#include "izhikevich_hamker.h"

// C++ includes:
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "event_delivery_manager_impl.h"
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< nest::izhikevich_hamker > nest::izhikevich_hamker::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< izhikevich_hamker >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &izhikevich_hamker::get_V_m_ );
  insert_( names::U_m, &izhikevich_hamker::get_U_m_ );
  insert_( names::g_L, &izhikevich_hamker::get_g_L_ );
  insert_( names::g_AMPA, &izhikevich_hamker::get_g_AMPA_ );
  insert_( names::g_GABA_A, &izhikevich_hamker::get_g_GABA_A_ );
  insert_( names::I_syn_ex, &izhikevich_hamker::get_I_syn_ex_ );
  insert_( names::I_syn_in, &izhikevich_hamker::get_I_syn_in_ );
  insert_( names::I_syn, &izhikevich_hamker::get_I_syn_ );
  insert_( names::I, &izhikevich_hamker::get_I_ );
}
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */
nest::izhikevich_hamker::Parameters_::Parameters_()
  : E_rev_AMPA_( 0.0 )                              // mV
  , E_rev_GABA_A_( -90.0 )                          // mV
  , V_th_( 30.0 )                                   // mV
  , V_min_( -std::numeric_limits< double >::max() ) // mV
  , V_r_( 0.0 )                                     // mV
  , C_m_( 1.0 )                                     // real
  , I_e_( 0.0 )                                     // pA
  , t_ref_( 10.0 )                                  // ms
  , tau_rise_( 1.0 )                                // ms
  , tau_rise_AMPA_( 10.0 )                          // ms
  , tau_rise_GABA_A_( 10.0 )                        // ms
  , n0_( 140.0 )                                    // n0
  , n1_( 5.0 )                                      // n1
  , n2_( 0.04 )                                     // n2
  , a_( 0.02 )                                      // a
  , b_( 0.2 )                                       // b
  , c_( -72.0 )                                     // c without unit
  , d_( 6.0 )                                       // d
  , current_stimulus_scale_( 1.0 )                  // current_stimulus_scale
  , current_stimulus_mode_( 0 )                     // current_stimulus_mode
  , consistent_integration_( true )
{
}

nest::izhikevich_hamker::State_::State_()
  : v_( -70.0 )      // membrane potential
  , u_( -18.55 )        // membrane recovery variable
  , g_L_( 0.0 )      // baseline conductance
  , g_AMPA_( 0.0 )   // AMPA conductance
  , g_GABA_A_( 0.0 ) // GABA conductance
  , I_syn_ex_( 0.0 ) // total AMPA current
  , I_syn_in_( 0.0 ) // total GABA current
  , I_syn_( 0.0 )    // total synaptic current
  , I_( 0.0 )        // input current
  , r_( 0 )          // number of refractory steps remaining
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::izhikevich_hamker::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::E_rev_AMPA, E_rev_AMPA_ );
  def< double >( d, names::E_rev_GABA_A, E_rev_GABA_A_ );
  def< double >( d, names::V_th, V_th_ ); // threshold value
  def< double >( d, names::V_min, V_min_ );
  def< double >( d, "V_r", V_r_ );
  def< double >( d, names::C_m, C_m_ );
  def< double >( d, names::I_e, I_e_ );
  def< double >( d, names::t_ref, t_ref_ );
  def< double >( d, names::tau_rise, tau_rise_ );
  def< double >( d, names::tau_rise_AMPA, tau_rise_AMPA_ );
  def< double >( d, names::tau_rise_GABA_A, tau_rise_GABA_A_ );
  def< double >( d, "n0", n0_ );
  def< double >( d, "n1", n1_ );
  def< double >( d, "n2", n2_ );
  def< double >( d, names::a, a_ );
  def< double >( d, names::b, b_ );
  def< double >( d, names::c, c_ );
  def< double >( d, names::d, d_ );
  def< double >( d, "current_stimulus_scale", current_stimulus_scale_ );
  def< long >( d, "current_stimulus_mode", current_stimulus_mode_ );
  def< bool >( d, names::consistent_integration, consistent_integration_ );
}

void
nest::izhikevich_hamker::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::E_rev_AMPA, E_rev_AMPA_ );
  updateValue< double >( d, names::E_rev_GABA_A, E_rev_GABA_A_ );
  updateValue< double >( d, names::V_th, V_th_ );
  updateValue< double >( d, names::V_min, V_min_ );
  updateValue< double >( d, "V_r", V_r_ );
  updateValue< double >( d, names::C_m, C_m_ );
  updateValue< double >( d, names::I_e, I_e_ );
  updateValue< double >( d, names::t_ref, t_ref_ );
  updateValue< double >( d, names::tau_rise, tau_rise_ );
  updateValue< double >( d, names::tau_rise_AMPA, tau_rise_AMPA_ );
  updateValue< double >( d, names::tau_rise_GABA_A, tau_rise_GABA_A_ );
  updateValue< double >( d, "n0", n0_ );
  updateValue< double >( d, "n1", n1_ );
  updateValue< double >( d, "n2", n2_ );
  updateValue< double >( d, names::a, a_ );
  updateValue< double >( d, names::b, b_ );
  updateValue< double >( d, names::c, c_ );
  updateValue< double >( d, names::d, d_ );
  updateValue< double >( d, "current_stimulus_scale", current_stimulus_scale_ );
  updateValue< long >( d, "current_stimulus_mode", current_stimulus_mode_ );
  updateValue< bool >( d, names::consistent_integration, consistent_integration_ );
  const double h = Time::get_resolution().get_ms();
  if ( not consistent_integration_ && h != 1.0 )
  {
    LOG( M_INFO, "Parameters_::set", "Use 1.0 ms as resolution for consistency." );
  }
  if ( t_ref_ <= 0.0 )
  {
    throw BadProperty( "tau_ref has to be positive!" );
  }
  if ( tau_rise_ <= 0.0 )
  {
    throw BadProperty( "tau_rise has to be positive!" );
  }
  if ( tau_rise_AMPA_ <= 0.0 )
  {
    throw BadProperty( "tau_rise_AMPA has to be positive!" );
  }
  if ( tau_rise_GABA_A_ <= 0.0 )
  {
    throw BadProperty( "tau_rise_GABA_A has to be positive!" );
  }
  if ( C_m_ <= 0.0 )
  {
    throw BadProperty( "C_m has to be positive!" );
  }
  if ( ( current_stimulus_mode_ < 0 ) || ( current_stimulus_mode_ > 2 ) )
  {
    throw BadProperty( "current_stimulus_mode has to be an integer in the interval [0,2]!" );
  }
}

void
nest::izhikevich_hamker::State_::get( DictionaryDatum& d, const Parameters_& ) const
{
  def< double >( d, names::V_m, v_ );             // Membrane potential
  def< double >( d, names::U_m, u_ );             // Membrane potential recovery variable
  def< double >( d, names::g_L, g_L_ );           // Baseline conductance
  def< double >( d, names::g_AMPA, g_AMPA_ );     // AMPA conductance
  def< double >( d, names::g_GABA_A, g_GABA_A_ ); // GABA conductance
  def< double >( d, names::I_syn_ex, I_syn_ex_ ); // AMPA current
  def< double >( d, names::I_syn_in, I_syn_in_ ); // GABA current
  def< double >( d, names::I_syn, I_syn_ );       // Total synaptic current
  def< double >( d, names::I, I_ );               // Input current
  def< double >( d, "r", r_ );               // number of refractory steps remaining
}

void
nest::izhikevich_hamker::State_::set( const DictionaryDatum& d, const Parameters_& )
{
  updateValue< double >( d, names::U_m, u_ );
  updateValue< double >( d, names::V_m, v_ );
  updateValue< double >( d, names::g_L, g_L_ );
  updateValue< double >( d, names::g_AMPA, g_AMPA_ );
  updateValue< double >( d, names::g_GABA_A, g_GABA_A_ );
  updateValue< double >( d, names::I_syn_ex, I_syn_ex_ );
  updateValue< double >( d, names::I_syn_in, I_syn_in_ );
  updateValue< double >( d, names::I_syn, I_syn_ );
  updateValue< double >( d, names::I, I_ );
  updateValue< double >( d, "r", r_ );
  if ( g_L_ < 0.0 )
  {
    throw BadProperty( "g_L can not be negative!" );
  }
  if ( g_AMPA_ < 0.0 )
  {
    throw BadProperty( "g_AMPA can not be negative!" );
  }
  if ( g_GABA_A_ < 0.0 )
  {
    throw BadProperty( "g_GABA_A can not be negative!" );
  }
  if ( r_ < 0.0 )
  {
    throw BadProperty( "r can not be negative!" );
  }
}

nest::izhikevich_hamker::Buffers_::Buffers_( izhikevich_hamker& n )
  : logger_( n )
{
}

nest::izhikevich_hamker::Buffers_::Buffers_( const Buffers_&, izhikevich_hamker& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::izhikevich_hamker::izhikevich_hamker()
  : ArchivingNode()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

nest::izhikevich_hamker::izhikevich_hamker( const izhikevich_hamker& n )
  : ArchivingNode( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
nest::izhikevich_hamker::init_state_( const Node& proto )
{
  const izhikevich_hamker& pr = downcast< izhikevich_hamker >( proto );
  S_ = pr.S_;
  S_.v_ = P_.c_;
  S_.u_ = P_.b_ * P_.c_;
}

void
nest::izhikevich_hamker::init_buffers_()
{
  B_.spikes_base_.clear();   // includes resize
  B_.spikes_exc_.clear();   // includes resize
  B_.spikes_inh_.clear();   // includes resize
  B_.currents_.clear(); // includes resize
  B_.logger_.reset();   // includes resize
  ArchivingNode::clear_history();
}

void
nest::izhikevich_hamker::pre_run_hook()
{
  B_.logger_.init();
  V_.refractory_counts_ = Time( Time::ms( P_.t_ref_ ) ).get_steps();
  assert( V_.refractory_counts_ >= 0 ); // since t_ref >= 0, this can only fail in error
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 */

void
nest::izhikevich_hamker::update( Time const& origin, const long from, const long to )
{
  assert( to >= 0 && from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  const double h = Time::get_resolution().get_ms();
  double v_old, u_old;

  for ( long lag = from; lag < to; ++lag )
  {

    // Check if the neuron is in refractory period:
    const bool is_refractory = S_.r_ > 0 ;

    // use standard forward Euler numerics in this case
    if ( P_.consistent_integration_ )
    {

      // Use U_m, V_m, from previous time step t:
      v_old = S_.v_;
      u_old = S_.u_;

      if ( !is_refractory ) {  // If the neuron is NOT refractory...
        // ...integrate V_m and U_m using old values at time t
        S_.v_ +=
            h * ( P_.n2_ * v_old * v_old + P_.n1_ * v_old + P_.n0_ - u_old / P_.C_m_+ S_.I_ + P_.I_e_ + S_.I_syn_ );
        S_.u_ += h * P_.a_ * ( P_.b_ * (v_old - P_.V_r_) - u_old );
      }

      // Add the spikes:
      S_.g_L_ += B_.spikes_base_.get_value( lag );
      S_.g_AMPA_ += B_.spikes_exc_.get_value( lag );
      S_.g_GABA_A_ += B_.spikes_inh_.get_value( lag );

      // Integrate conductances using values at time t + spikes at time t:
      S_.g_L_ -= h * S_.g_L_ / P_.tau_rise_;
      S_.g_AMPA_ -= h * S_.g_AMPA_/ P_.tau_rise_AMPA_;
      S_.g_GABA_A_ -= h *  S_.g_GABA_A_ / P_.tau_rise_GABA_A_;

      // lower bound of conductances
      S_.g_L_ = ( S_.g_L_ < 0.0 ? 0.0 : S_.g_L_ );
      S_.g_AMPA_ = ( S_.g_AMPA_ < 0.0 ? 0.0 : S_.g_AMPA_ );
      S_.g_GABA_A_ = ( S_.g_GABA_A_ < 0.0 ? 0.0 : S_.g_GABA_A_ );

      // Compute synaptic currents at time step t+1:
      S_.I_syn_ex_ = - S_.g_AMPA_ * ( S_.v_ - P_.E_rev_AMPA_ );
      S_.I_syn_in_ = - S_.g_GABA_A_ * ( S_.v_ - P_.E_rev_GABA_A_ );
      S_.I_syn_ = S_.I_syn_ex_ + S_.I_syn_in_  - S_.g_L_ * S_.v_;

    }
    // use numerics published in Izhikevich (2003) in this case (not
    // recommended)
    else
    {
      // Add the spikes:
      S_.g_L_ += B_.spikes_base_.get_value( lag );
      S_.g_AMPA_ += B_.spikes_exc_.get_value( lag );
      S_.g_GABA_A_ += B_.spikes_inh_.get_value( lag );

      // Integrate conductances using values at time t + spikes at time t:
      S_.g_L_ -= h * S_.g_L_ / P_.tau_rise_;
      S_.g_AMPA_ -= h * S_.g_AMPA_/ P_.tau_rise_AMPA_;
      S_.g_GABA_A_ -= h *  S_.g_GABA_A_ / P_.tau_rise_GABA_A_;

      // lower bound of conductances
      S_.g_L_ = ( S_.g_L_ < 0.0 ? 0.0 : S_.g_L_ );
      S_.g_AMPA_ = ( S_.g_AMPA_ < 0.0 ? 0.0 : S_.g_AMPA_ );
      S_.g_GABA_A_ = ( S_.g_GABA_A_ < 0.0 ? 0.0 : S_.g_GABA_A_ );

      // Compute synaptic currents at time step t+1:
      S_.I_syn_ex_ = - S_.g_AMPA_ * ( S_.v_ - P_.E_rev_AMPA_ );
      S_.I_syn_in_ = - S_.g_GABA_A_ * ( S_.v_ - P_.E_rev_GABA_A_ );
      S_.I_syn_ = S_.I_syn_ex_ + S_.I_syn_in_  - S_.g_L_ * S_.v_;

      if ( !is_refractory ) {  // If the neuron is NOT refractory...
          // ...integrate U_m, V_m using new values (t+1)
          S_.v_ += h * 0.5 * (
                         P_.n2_ * S_.v_ * S_.v_ + P_.n1_ * S_.v_ + P_.n0_ - S_.u_ / P_.C_m_ + S_.I_ + P_.I_e_ + S_.I_syn_ );
          S_.v_ += h * 0.5 * (
                         P_.n2_ * S_.v_ * S_.v_ + P_.n1_ * S_.v_ + P_.n0_ - S_.u_ / P_.C_m_ + S_.I_ + P_.I_e_ + S_.I_syn_ );
          S_.u_ += h * P_.a_ * ( P_.b_ * (S_.v_ - P_.V_r_) - S_.u_ );
      }
    }

    // lower bound of membrane potential
    S_.v_ = ( S_.v_ < P_.V_min_ ? P_.V_min_ : S_.v_ );

    // threshold crossing
    if ( is_refractory ) // if neuron is still in refractory period
    {
      --S_.r_;
    }
    else if ( S_.v_ >= P_.V_th_ )
    {

      // Spike!

      // Clamp v and u
      S_.v_ = P_.c_;
      S_.u_ += P_.d_;

      // compute spike time
      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      // Apply spike
      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );

      /* Initialize refractory step counter.
       * - We need to add 1 to compensate for count-down immediately after
       *   while loop.
       * - If neuron has no refractory time, set to 0 to avoid refractory
       *   artifact inside while loop.
       */
      S_.r_ = V_.refractory_counts_ > 0 ? V_.refractory_counts_ + 1 : 0;

    }

    // set new input current
    S_.I_ = B_.currents_.get_value( lag );

    // optional transformation of input current
    if ( P_.current_stimulus_mode_ == 1 )
    {
        S_.I_ = std::abs(S_.I_) ;
    }
    else if ( P_.current_stimulus_mode_ == 2 )
    {
        S_.I_ = ( S_.I_ > 0.0 ? 1.0 : 0.0 ) ;
    }
    S_.I_ *= P_.current_stimulus_scale_ ;

    // voltage logging
    B_.logger_.record_data( origin.get_steps() + lag );
  }
}

void
nest::izhikevich_hamker::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );
  assert( ( e.get_rport() >= MIN_SPIKE_RECEPTOR ) && ( e.get_rport() <= SUP_SPIKE_RECEPTOR ) );

  const double weight = e.get_weight() * e.get_multiplicity();

 if (e.get_rport() == NOISE) {
    B_.spikes_base_.add_value(
            e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), weight );
 } else if (e.get_rport() == ACTIVITY) {
    if ( weight < 0.0 ) {
       B_.spikes_inh_.add_value(
           e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), -weight );
    } else {
       B_.spikes_exc_.add_value(
           e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), weight );
    }
 }

}

void
nest::izhikevich_hamker::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  const double c = e.get_current();
  const double w = e.get_weight();
  B_.currents_.add_value( e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), w * c );
}

void
nest::izhikevich_hamker::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}
