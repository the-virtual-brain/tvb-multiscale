#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern "C" void _dynvecstim_reg(void);

extern "C" void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"/home/docker/packages/tvb-multiscale/tvb_multiscale/tvb_netpyne/netpyne/mod/dynvecstim.mod\"");
    fprintf(stderr, "\n");
  }
  _dynvecstim_reg();
}
