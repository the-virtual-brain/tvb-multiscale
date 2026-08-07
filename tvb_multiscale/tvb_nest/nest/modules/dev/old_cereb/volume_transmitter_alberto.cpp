/*
 *  volume_transmitter_alberto.cpp
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

#include "volume_transmitter_alberto.h"

// C++ includes:
#include <numeric>

// Includes from nestkernel:
#include "connector_base.h"
#include "exceptions.h"
#include "kernel_manager.h"
#include "spikecounter.h"

// Includes from sli:
#include "arraydatum.h"
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "mynames.h"

/* ----------------------------------------------------------------
 * Default constructor defining default parameters
 * ---------------------------------------------------------------- */

mynest::volume_transmitter_alberto::Parameters_::Parameters_()
  : deliver_interval_( 1 ) // in steps of mindelay
  , vt_num_ ( 0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
mynest::volume_transmitter_alberto::Parameters_::get( DictionaryDatum& d ) const
{
  def< long >( d, nest::names::deliver_interval, deliver_interval_ );
  def< long >( d, "vt_num", vt_num_ );
}

void mynest::volume_transmitter_alberto::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< long >( d, nest::names::deliver_interval, deliver_interval_ );
  updateValue< long >( d, "vt_num", vt_num_ );
}

/* ----------------------------------------------------------------
 * Default and copy constructor for volume transmitter
 * ---------------------------------------------------------------- */

mynest::volume_transmitter_alberto::volume_transmitter_alberto()
  : Archiving_Node()
  , P_()
{
}

mynest::volume_transmitter_alberto::volume_transmitter_alberto( const volume_transmitter_alberto& n )
  : Archiving_Node( n )
  , P_( n.P_ )
{
}

void
mynest::volume_transmitter_alberto::init_state_( const Node& )
{
}

void
mynest::volume_transmitter_alberto::init_buffers_()
{
  B_.neuromodulatory_spikes_.clear();
  B_.spikecounter_.clear();
  B_.spikecounter_.push_back(
  nest::spikecounter( 0.0, 0.0 ) ); // insert pseudo last dopa spike at t = 0.0
  Archiving_Node::clear_history();
}

void
mynest::volume_transmitter_alberto::calibrate()
{
  // +1 as pseudo dopa spike at t_trig is inserted after trigger_update_weight
  B_.spikecounter_.reserve(
    nest::kernel().connection_manager.get_min_delay() * P_.deliver_interval_ + 1 );
}

void
mynest::volume_transmitter_alberto::update( const nest::Time&, const long from, const long to )
{
  // spikes that arrive in this time slice are stored in spikecounter_
  double t_spike;
  double multiplicity;
  long lag = from;
  multiplicity = B_.neuromodulatory_spikes_.get_value( lag );
  if ( multiplicity > 0 )
  {
    t_spike =  nest::Time( nest::Time::step( nest::kernel().simulation_manager.get_slice_origin().get_steps()+ lag + 1 ) ).get_ms();
    B_.spikecounter_.push_back( nest::spikecounter( t_spike, P_.vt_num_ ) );

    // all spikes stored in spikecounter_ are delivered to the target synapses
    if ( ( nest::kernel().simulation_manager.get_slice_origin().get_steps() + to ) % ( P_.deliver_interval_ * nest::kernel().connection_manager.get_min_delay() ) == 0 )
    {
      double t_trig = nest::Time(nest::Time::step( nest::kernel().simulation_manager.get_slice_origin().get_steps() + to ) ).get_ms();
      if ( not( B_.spikecounter_.empty() ) )
      {
        //std::cout << P_.vt_num_ << " << P_.vt_num_ " << get_gid() << " << get_gid() " << std::endl;
        nest::kernel().connection_manager.trigger_update_weight(get_gid()-P_.vt_num_, B_.spikecounter_, t_trig );
      }
      // clear spikecounter
      B_.spikecounter_.clear();

      // as with trigger_update_weight dopamine trace has been updated to t_trig,
      // insert pseudo last dopa spike at t_trig
      B_.spikecounter_.push_back( nest::spikecounter( t_trig, 0.0 ) );
    }
  }
}

void
mynest::volume_transmitter_alberto::handle( nest::SpikeEvent& e )
{
  B_.neuromodulatory_spikes_.add_value(
    e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ),
    static_cast< double >( e.get_multiplicity() ) );
}
