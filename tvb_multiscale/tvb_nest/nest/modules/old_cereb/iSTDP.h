/*
 *  stdp_connection.h
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

#ifndef iSTDP_H
#define iSTDP_H

/* BeginDocumentation
  Name: stdp_synapse - Synapse type for spike-timing dependent
   plasticity.

  Description:
   stdp_synapse is a connector to create synapses with spike time
   dependent plasticity (as defined in [1]). Here the weight dependence
   exponent can be set separately for potentiation and depression.

  Examples:
   multiplicative STDP [2]  mu_plus = mu_minus = 1.0
   additive STDP       [3]  mu_plus = mu_minus = 0.0
   Guetig STDP         [1]  mu_plus = mu_minus = [0.0,1.0]
   van Rossum STDP     [4]  mu_plus = 0.0 mu_minus = 1.0

  Parameters:
   tau_plus   double - Time constant of STDP window, potentiation in ms
                       (tau_minus defined in post-synaptic neuron)
   lambda     double - Step size
   alpha      double - Asymmetry parameter (scales depressing increments as
                       alpha*lambda)
   mu_plus    double - Weight dependence exponent, potentiation
   mu_minus   double - Weight dependence exponent, depression
   Wmax       double - Maximum allowed weight

  Transmits: SpikeEvent

  References:
   [1] Guetig et al. (2003) Learning Input Correlations through Nonlinear
       Temporally Asymmetric Hebbian Plasticity. Journal of Neuroscience

   [2] Rubin, J., Lee, D. and Sompolinsky, H. (2001). Equilibrium
       properties of temporally asymmetric Hebbian plasticity, PRL
       86,364-367

   [3] Song, S., Miller, K. D. and Abbott, L. F. (2000). Competitive
       Hebbian learning through spike-timing-dependent synaptic
       plasticity,Nature Neuroscience 3:9,919--926

   [4] van Rossum, M. C. W., Bi, G-Q and Turrigiano, G. G. (2000).
       Stable Hebbian learning from spike timing-dependent
       plasticity, Journal of Neuroscience, 20:23,8812--8821

  FirstVersion: March 2006
  Author: Moritz Helias, Abigail Morrison
  Adapted by: Philipp Weidel
  SeeAlso: synapsedict, tsodyks_synapse, static_synapse
*/

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"


namespace mynest
{


// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template
template < typename targetidentifierT >
class iSTDP : public nest ::Connection< targetidentifierT >
{

public:
  typedef nest::CommonSynapseProperties CommonPropertiesType;
  typedef nest::Connection< targetidentifierT > ConnectionBase;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  iSTDP();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  iSTDP( const iSTDP& );

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_delay;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, nest::ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param t_lastspike Point in time of last spike sent.
   * \param cp common properties of all synapses (empty).
   */
  void send( nest::Event& e,
    nest::thread t,
    const nest::CommonSynapseProperties& cp );


  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using nest::ConnTestDummyNodeBase::handles_test_event;
    nest::port
    handles_test_event( nest::SpikeEvent&, nest::rport )
    {
      return nest::invalid_port_;
    }
  };

  void
  check_connection( nest::Node& s,
    nest::Node& t,
    nest::rport receptor_type,
    const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;

    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );

    t.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }

private:
  double
  calculate_k_( double dt )
  {
    double k = Wmax_ * 0.03 *std::exp( - 1.0 * ( std::abs(dt) / tau_plus_)) * (1 + cos(2*(dt)/tau_plus_)) -lambda_ * Wmax_ * 0.03 * mu_plus_ * std::exp(  - 1.0 *      (std::abs (dt) / mu_minus_)) * (1 - cos(2*(dt)/mu_minus_));
   // std::cout << cos(2*(dt)/195.6) << std::endl;
    //std::cout << cos(2*(dt)/125) << std::endl;
    return k;
  }
  double
  facilitate_( double w, double kplus )
  {
    double norm_w;
    double norm_w1 = 0.5 * kplus;
    if ( w >= 0 )
    {
      norm_w = norm_w1 + w;
    }
    if ( w < 0 )
    {
      norm_w = norm_w1 + w;
    }

    //std::cout << norm_w1<< std::endl;
    //std::cout << kplus << std::endl;

    return norm_w;
  }

  // data members of each connection
  double weight_;
  double tau_plus_;
  double lambda_;
  double alpha_;
  double mu_plus_;
  double mu_minus_;
  double Wmax_;
  double Kplus_;
  double Wmin_;
  double t_lastspike_;
};


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param t_lastspike_ Time point of last spike emitted
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
iSTDP< targetidentifierT >::send( nest::Event& e,
  nest::thread t,
  const nest::CommonSynapseProperties& )
{
  // synapse STDP depressing/facilitation dynamics
  double t_spike = e.get_stamp().get_ms();
  
  // use accessor functions (inherited from Connection< >) to obtain delay and
  // target
  nest::Node* target = get_target( t );
  double dendritic_delay = get_delay();

  // get spike history in relevant range (t1, t2] from post-synaptic neuron
  std::deque< nest::histentry >::iterator start;
  std::deque< nest::histentry >::iterator finish;


  // For a new synapse, t_lastspike_ contains the point in time of the last
  // spike. So we initially read the
  // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
  // which increases the access counter for these entries.
  // At registration, all entries' access counters of
  // history[0, ..., t_last_spike - dendritic_delay] have been
  // incremented by Archiving_Node::register_stdp_connection(). See bug #218 for
  // details.
  target->get_history(
    t_lastspike_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish );

  // facilitation due to post-synaptic spikes since last pre-synaptic spike
  double dtp_;
  double dtn_;
  while ( start != finish )
  {
    //std::cout << start->t_ << "\t"<< t_lastspike_ << "\t" << t_spike  << std::endl;
    dtp_ = t_spike - ( start -> t_ );
    Kplus_ = calculate_k_( dtp_ );
    // std::cout << Kplus_ << std::endl;
    weight_ = facilitate_( weight_, Kplus_ );

    if( t_lastspike_ > 0 )
    {
      dtn_ = ( start -> t_ ) - t_lastspike_;
      Kplus_ = calculate_k_( dtn_ );
      // std::cout << start ->t_<< std::endl;
      weight_ = facilitate_( weight_, Kplus_ );
    }
    ++start;
  }


  e.set_receiver( *target );
  if ( weight_ < Wmin_ )
  {
    weight_ = Wmin_;
  }
  if ( weight_ > Wmax_ )
  {
    weight_ = Wmax_;
  }
  e.set_weight( weight_ );
  // use accessor functions (inherited from Connection< >) to obtain delay in
  // steps and rport
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();

  t_lastspike_ = t_spike;

}


template < typename targetidentifierT >
iSTDP< targetidentifierT >::iSTDP()
  : ConnectionBase()
  , weight_( 1.0 )
  , tau_plus_( 125.0 )
  , lambda_( 0.01 )
  , alpha_( 1.0 )
  , mu_plus_( 1.0 )
  , mu_minus_( 1.0 )
  , Wmax_( 100.0 )
  , Kplus_( 0.0 )
  , Wmin_(-100.0)
  , t_lastspike_( 0.0 )
{
}

template < typename targetidentifierT >
iSTDP< targetidentifierT >::iSTDP(
  const iSTDP< targetidentifierT >& rhs )
  : ConnectionBase( rhs )
  , weight_( rhs.weight_ )
  , tau_plus_( rhs.tau_plus_ )
  , lambda_( rhs.lambda_ )
  , alpha_( rhs.alpha_ )
  , mu_plus_( rhs.mu_plus_ )
  , mu_minus_( rhs.mu_minus_ )
  , Wmax_( rhs.Wmax_ )
  , Kplus_( rhs.Kplus_ )
  , Wmin_( rhs.Wmin_ )
  , t_lastspike_( rhs.t_lastspike_ )
{
}

template < typename targetidentifierT >
void
iSTDP< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, nest::names::weight, weight_ );
  def< double >( d, nest::names::tau_plus, tau_plus_ );
  def< double >( d, nest::names::lambda, lambda_ );
  def< double >( d, nest::names::alpha, alpha_ );
  def< double >( d, nest::names::mu_plus, mu_plus_ );
  def< double >( d, nest::names::mu_minus, mu_minus_ );
  def< double >( d, nest::names::Wmax, Wmax_ );
  def< double >( d, nest::names::Wmin, Wmin_ );
  def< long >( d, nest::names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
iSTDP< targetidentifierT >::set_status( const DictionaryDatum& d,
  nest::ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, nest::names::weight, weight_ );
  updateValue< double >( d, nest::names::tau_plus, tau_plus_ );
  updateValue< double >( d, nest::names::lambda, lambda_ );
  updateValue< double >( d, nest::names::alpha, alpha_ );
  updateValue< double >( d, nest::names::mu_plus, mu_plus_ );
  updateValue< double >( d, nest::names::mu_minus, mu_minus_ );
  updateValue< double >( d, nest::names::Wmax, Wmax_ );
  updateValue< double >( d, nest::names::Wmin, Wmin_ );

  // check if weight_ and Wmax_ has the same sign
  if ( not( ( ( weight_ >= 0 ) - ( weight_ < 0 ) )
         == ( ( Wmax_ >= 0 ) - ( Wmax_ < 0 ) ) ) )
  {
    throw nest::BadProperty( "Weight and Wmax must have same sign." );
  }
}

} // of namespace nest

#endif // of #ifndef iSTDP_H
