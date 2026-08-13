/*
 *  mynames.h
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

#ifndef MY_NAMES_H
#define MY_NAMES_H

// Includes from sli:
#include "name.h"

namespace mynest
{

/**
 * This namespace contains global Name objects. These can be used in
 * Node::get_status and Node::set_status to make data exchange
 * more efficient and consistent. Creating a Name from a std::string
 * is in O(log n), for n the number of Names already created. Using
 * predefined names should make data exchange much more efficient.
 */
namespace mynames
{
extern const Name first_dcn;
extern const Name gain;
extern const Name num_dcn;            //!< Number of Deep Cerebellar Nuclei
                                      //!< (closed_loop_neuron)
extern const Name positive;
extern const Name protocol;
extern const Name Tduration; //!< US duration for Closed Loop Neuron
extern const Name sdev;

extern const Name vt_num;

extern const Name trial_length;
extern const Name target;
extern const Name prism_deviation;
extern const Name baseline_rate;
extern const Name gain_rate;
extern const Name joint_id;
extern const Name fiber_id;
extern const Name fibers_per_joint;
extern const Name rbf_sdev;
}
}

#endif //MY_NAMES_H
