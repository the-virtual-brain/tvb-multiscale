/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#define _pval pval
// clang-format off
#include "md1redef.h"
#include "section_fwd.hpp"
#include "nrniv_mf.h"
#include "md2redef.h"
#include "nrnconf.h"
// clang-format on
#include "neuron/cache/mechanism_range.hpp"
static constexpr auto number_of_datum_variables = 3;
static constexpr auto number_of_floating_point_variables = 5;
namespace {
template <typename T>
using _nrn_mechanism_std_vector = std::vector<T>;
using _nrn_model_sorted_token = neuron::model_sorted_token;
using _nrn_mechanism_cache_range = neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_mechanism_cache_instance = neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_non_owning_id_without_container = neuron::container::non_owning_identifier_without_container;
template <typename T>
using _nrn_mechanism_field = neuron::mechanism::field<T>;
template <typename... Args>
void _nrn_mechanism_register_data_fields(Args&&... args) {
  neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
}
}
 
#if !NRNGPU
#undef exp
#define exp hoc_Exp
#if NRN_ENABLE_ARCH_INDEP_EXP_POW
#undef pow
#define pow hoc_pow
#endif
#endif
 
#define nrn_init _nrn_init__DynamicVecStim
#define _nrn_initial _nrn_initial__DynamicVecStim
#define nrn_cur _nrn_cur__DynamicVecStim
#define _nrn_current _nrn_current__DynamicVecStim
#define nrn_jacob _nrn_jacob__DynamicVecStim
#define nrn_state _nrn_state__DynamicVecStim
#define _net_receive _net_receive__DynamicVecStim 
#define inferNextEvent inferNextEvent__DynamicVecStim 
#define play play__DynamicVecStim 
 
#define _threadargscomma_ _ml, _iml, _ppvar, _thread, _globals, _nt,
#define _threadargsprotocomma_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _internalthreadargsprotocomma_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _threadargs_ _ml, _iml, _ppvar, _thread, _globals, _nt
#define _threadargsproto_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
#define _internalthreadargsproto_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 
#define t _nt->_t
#define dt _nt->_dt
#define index _ml->template fpfield<0>(_iml)
#define index_columnindex 0
#define intervalEnd _ml->template fpfield<1>(_iml)
#define intervalEnd_columnindex 1
#define etime _ml->template fpfield<2>(_iml)
#define etime_columnindex 2
#define v _ml->template fpfield<3>(_iml)
#define v_columnindex 3
#define _tsav _ml->template fpfield<4>(_iml)
#define _tsav_columnindex 4
#define _nd_area *_ml->dptr_field<0>(_iml)
#define ptr	*_ppvar[2].get<double*>()
#define _p_ptr _ppvar[2].literal_value<void*>()
 /* Thread safe. No static _ml, _iml or _ppvar. */
 static int hoc_nrnpointerindex =  2;
 static _nrn_mechanism_std_vector<Datum> _extcall_thread;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_inferNextEvent(void*);
 static double _hoc_play(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mechtype);
#endif
 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 static void _hoc_setdata(void*);
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 {0, 0}
};
 static Member_func _member_func[] = {
 {"loc", _hoc_loc_pnt},
 {"has_loc", _hoc_has_loc},
 {"get_loc", _hoc_get_loc_pnt},
 {"inferNextEvent", _hoc_inferNextEvent},
 {"play", _hoc_play},
 {0, 0}
};
 /* declare global and static user variables */
 #define gind 0
 #define _gth 0
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {0, 0}
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {0, 0, 0}
};
 static double _sav_indep;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 static void nrn_alloc(Prop*);
static void nrn_init(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_state(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 static void _destructor(Prop*);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"DynamicVecStim",
 0,
 0,
 0,
 "ptr",
 0};
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
 }; 
 
 
extern Prop* need_memb(Symbol*);
static void nrn_alloc(Prop* _prop) {
  Prop *prop_ion{};
  Datum *_ppvar{};
  if (nrn_point_prop_) {
    _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
    _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
  } else {
   _ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
    _nrn_mechanism_access_dparam(_prop) = _ppvar;
     _nrn_mechanism_cache_instance _ml_real{_prop};
    auto* const _ml = &_ml_real;
    size_t const _iml{};
    assert(_nrn_mechanism_get_num_vars(_prop) == 5);
 	/*initialize range parameters*/
  }
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 5);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[3])
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _dynvecstim_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nullptr, nullptr, nullptr, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 	register_destructor(_destructor);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"index"} /* 0 */,
                                       _nrn_mechanism_field<double>{"intervalEnd"} /* 1 */,
                                       _nrn_mechanism_field<double>{"etime"} /* 2 */,
                                       _nrn_mechanism_field<double>{"v"} /* 3 */,
                                       _nrn_mechanism_field<double>{"_tsav"} /* 4 */,
                                       _nrn_mechanism_field<double*>{"_nd_area", "area"} /* 0 */,
                                       _nrn_mechanism_field<Point_process*>{"_pntproc", "pntproc"} /* 1 */,
                                       _nrn_mechanism_field<double*>{"ptr", "pointer"} /* 2 */,
                                       _nrn_mechanism_field<void*>{"_tqitem", "netsend"} /* 3 */);
  hoc_register_prop_size(_mechtype, 5, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
 add_nrn_artcell(_mechtype, 3);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 DynamicVecStim /home/docker/packages/tvb-multiscale/tvb_multiscale/tvb_netpyne/netpyne/mod/dynvecstim.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int inferNextEvent(_internalthreadargsproto_);
static int play(_internalthreadargsproto_);
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  Prop* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _nrn_mechanism_cache_instance _ml_real{_pnt->_prop};
  auto* const _ml = &_ml_real;
  size_t const _iml{};
   _thread = nullptr; double* _globals = nullptr; _nt = (NrnThread*)_pnt->_vnt;   _ppvar = _nrn_mechanism_access_dparam(_pnt->_prop);
  if (_tsav > t){ hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = nullptr;}
 {
   if ( _lflag  == 1.0  || _lflag  == 2.0 ) {
     if ( _lflag  == 1.0 ) {
       net_event ( _pnt, t ) ;
       }
     inferNextEvent ( _threadargs_ ) ;
     if ( index > 0.0 ) {
       if ( etime < t ) {
         etime = t ;
         }
       artcell_net_send ( _tqitem, _args, _pnt, t +  etime - t , 1.0 ) ;
       }
     else {
       double _lendTime ;
 _lendTime = intervalEnd - t ;
       intervalEnd = - 1.0 ;
       artcell_net_send ( _tqitem, _args, _pnt, t +  _lendTime , 3.0 ) ;
       }
     }
   else if ( _lflag  == 3.0 ) {
     if ( intervalEnd  == - 1.0 ) {
       artcell_net_send ( _tqitem, _args, _pnt, t +  dt , 3.0 ) ;
       }
     else {
       artcell_net_send ( _tqitem, _args, _pnt, t +  0.0 , 2.0 ) ;
       }
     }
   } }
 
static int  inferNextEvent ( _internalthreadargsproto_ ) {
   
/*VERBATIM*/

  { void* vv; int i, size; double* px;
	i = (int)index;
	if (i >= 0) {
		vv = (void*)(_p_ptr);
		if (vv) {
			size = vector_capacity(vv);
			px = vector_vec(vv);
			if (i < size) {
				etime = px[i];
				if (etime < intervalEnd) {
					index += 1.;
				} else {
					printf("WARNING in DynamicVecStim. Spike at %f will be skipped, as well as all later spikes in this interval, since they go after the interval end\n", etime);
					index = -1.;
				}
			}else{
				index = -1.;
			}
		}else{
			index = -1.;
		}
	}
  }
  return 0; }
 
static double _hoc_inferNextEvent(void* _vptr) {
 double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   auto* const _pnt = static_cast<Point_process*>(_vptr);
  auto* const _p = _pnt->_prop;
  if (!_p) {
    hoc_execerror("POINT_PROCESS data instance not valid", NULL);
  }
   _nrn_mechanism_cache_instance _ml_real{_p};
  auto* const _ml = &_ml_real;
  size_t const _iml{};
  _ppvar = _nrn_mechanism_access_dparam(_p);
  _thread = _extcall_thread.data();
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  _nt = static_cast<NrnThread*>(_pnt->_vnt);
 _r = 1.;
 inferNextEvent ( _threadargs_ );
 return(_r);
}
 
static int  play ( _internalthreadargsproto_ ) {
   index = 0.0 ;
   
/*VERBATIM*/
	void** pv;
	void* ptmp = NULL;
	if (ifarg(2)) {
		intervalEnd = *getarg(2);
	} else {
		printf("ERROR in DynamicVecStim! End of interval should be specified as second arg in play() !!");
		exit(1);
	}
	if (ifarg(1)) {
		ptmp = vector_arg(1);
		hoc_obj_ref(*vector_pobj(ptmp));
	}
	pv = (void**)(&_p_ptr);
	if (*pv) {
		hoc_obj_unref(*vector_pobj(*pv));
	}
	*pv = ptmp;
  return 0; }
 
static double _hoc_play(void* _vptr) {
 double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   auto* const _pnt = static_cast<Point_process*>(_vptr);
  auto* const _p = _pnt->_prop;
  if (!_p) {
    hoc_execerror("POINT_PROCESS data instance not valid", NULL);
  }
   _nrn_mechanism_cache_instance _ml_real{_p};
  auto* const _ml = &_ml_real;
  size_t const _iml{};
  _ppvar = _nrn_mechanism_access_dparam(_p);
  _thread = _extcall_thread.data();
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  _nt = static_cast<NrnThread*>(_pnt->_vnt);
 _r = 1.;
 play ( _threadargs_ );
 return(_r);
}
 
static void _destructor(Prop* _prop) {
  _nrn_mechanism_cache_instance _ml_real{_prop};
  auto* const _ml = &_ml_real;
  size_t const _iml{};
  Datum *_ppvar{_nrn_mechanism_access_dparam(_prop)}, *_thread{};
  {
 {
   
/*VERBATIM*/
	void* vv = (void*)(_p_ptr);  
        if (vv) {
		hoc_obj_unref(*vector_pobj(vv));
	}
 }
 
}
}

static void initmodel(_internalthreadargsproto_) {
  int _i; double _save;{
 {
   index = 0.0 ;
   inferNextEvent ( _threadargs_ ) ;
   if ( index > 0.0 ) {
     artcell_net_send ( _tqitem, nullptr, _ppvar[1].get<Point_process*>(), t +  etime - t , 1.0 ) ;
     }
   else {
     artcell_net_send ( _tqitem, nullptr, _ppvar[1].get<Point_process*>(), t +  intervalEnd - t , 3.0 ) ;
     }
   }

}
}

static void nrn_init(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
 _tsav = -1e20;
 initmodel(_threadargs_);
}
}

static double _nrn_current(_internalthreadargsprotocomma_ double _v) {
double _current=0.; v=_v;
{
} return _current;
}

static void nrn_state(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni;
_ni = _ml_arg->_nodeindices;
size_t _cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (size_t _iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
 _nd = _ml_arg->_nodelist[_iml];
 v=_v;
{
}}

}

static void terminal(){}

static void _initlists(){
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/docker/packages/tvb-multiscale/tvb_multiscale/tvb_netpyne/netpyne/mod/dynvecstim.mod";
    const char* nmodl_file_text = 
  ":  Vector stream of events\n"
  "\n"
  "NEURON {\n"
  "	THREADSAFE\n"
  "	ARTIFICIAL_CELL DynamicVecStim\n"
  "	POINTER ptr\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	index\n"
  "	intervalEnd\n"
  "	etime (ms)\n"
  "	ptr\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "	index = 0\n"
  "	inferNextEvent()\n"
  "	if (index > 0) {\n"
  "		net_send(etime - t, 1) : schedule spike at time etime\n"
  "	} else {\n"
  "		: no spikes. wait for inetrval end to be able to initialize next interval\n"
  "		net_send(intervalEnd - t, 3)\n"
  "	}\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "	: flag 1 - emit a spike and prepare for the next one\n"
  "	: flag 2 - interval just started, prepare for the first spike\n"
  "	: flag 3 - interval ended, waiting for initialization of the next one\n"
  "	if (flag == 1 || flag == 2) { \n"
  "		if (flag == 1) {\n"
  "			net_event(t) : emit spike\n"
  "		}\n"
  "		inferNextEvent()\n"
  "\n"
  "		if (index > 0) {\n"
  "			if (etime < t) {\n"
  "				: this may happen if the first spike in this interval is earlier than dt. Need to do this correction:\n"
  "				etime = t\n"
  "			}\n"
  "			net_send(etime - t, 1) : schedule next spike\n"
  "			\n"
  "		} else {\n"
  "			: no more spikes. wait for inetrval end to be able to initialize next interval\n"
  "			LOCAL endTime\n"
  "			endTime = intervalEnd - t\n"
  "			intervalEnd = -1\n"
  "			net_send(endTime, 3)\n"
  "		}\n"
  "	} else if (flag == 3) {\n"
  "		if (intervalEnd == -1) {\n"
  "			: still waiting for next interval initialization\n"
  "			net_send(dt, 3)\n"
  "		} else {\n"
  "			: next interval is already initialized ( from play() )! Proceed to first spike in it\n"
  "			net_send(0, 2)\n"
  "		}\n"
  "	}\n"
  "}\n"
  "\n"
  "DESTRUCTOR {\n"
  "VERBATIM\n"
  "	void* vv = (void*)(_p_ptr);  \n"
  "        if (vv) {\n"
  "		hoc_obj_unref(*vector_pobj(vv));\n"
  "	}\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE inferNextEvent() {\n"
  "VERBATIM	\n"
  "  { void* vv; int i, size; double* px;\n"
  "	i = (int)index;\n"
  "	if (i >= 0) {\n"
  "		vv = (void*)(_p_ptr);\n"
  "		if (vv) {\n"
  "			size = vector_capacity(vv);\n"
  "			px = vector_vec(vv);\n"
  "			if (i < size) {\n"
  "				etime = px[i];\n"
  "				if (etime < intervalEnd) {\n"
  "					index += 1.;\n"
  "				} else {\n"
  "					printf(\"WARNING in DynamicVecStim. Spike at %f will be skipped, as well as all later spikes in this interval, since they go after the interval end\\n\", etime);\n"
  "					index = -1.;\n"
  "				}\n"
  "			}else{\n"
  "				index = -1.;\n"
  "			}\n"
  "		}else{\n"
  "			index = -1.;\n"
  "		}\n"
  "	}\n"
  "  }\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE play() {\n"
  "	index = 0\n"
  "VERBATIM\n"
  "	void** pv;\n"
  "	void* ptmp = NULL;\n"
  "	if (ifarg(2)) {\n"
  "		intervalEnd = *getarg(2);\n"
  "	} else {\n"
  "		printf(\"ERROR in DynamicVecStim! End of interval should be specified as second arg in play() !!\");\n"
  "		exit(1);\n"
  "	}\n"
  "	if (ifarg(1)) {\n"
  "		ptmp = vector_arg(1);\n"
  "		hoc_obj_ref(*vector_pobj(ptmp));\n"
  "	}\n"
  "	pv = (void**)(&_p_ptr);\n"
  "	if (*pv) {\n"
  "		hoc_obj_unref(*vector_pobj(*pv));\n"
  "	}\n"
  "	*pv = ptmp;\n"
  "ENDVERBATIM\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
