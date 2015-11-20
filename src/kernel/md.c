/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 * Copyright (c) 2012,2013, by the GROMACS development team, led by
 * David van der Spoel, Berk Hess, Erik Lindahl, and including many
 * others, as listed in the AUTHORS file in the top-level source
 * directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define MAIN_FILE

#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "statutil.h"
#include "vcm.h"
#include "mdebin.h"
#include "nrnb.h"
#include "calcmu.h"
#include "index.h"
#include "vsite.h"
#include "update.h"
#include "ns.h"
#include "trnio.h"
#include "xtcio.h"
#include "mdrun.h"
#include "md_support.h"
#include "md_logging.h"
#include "confio.h"
#include "network.h"
#include "pull.h"
#include "xvgr.h"
#include "physics.h"
#include "names.h"
#include "xmdrun.h"
#include "ionize.h"
#include "disre.h"
#include "orires.h"
#include "pme.h"
#include "mdatoms.h"
#include "repl_ex.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "domdec_network.h"
#include "partdec.h"
#include "topsort.h"
#include "coulomb.h"
#include "constr.h"
#include "shellfc.h"
#include "compute_io.h"
#include "mvdata.h"
#include "checkpoint.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "txtdump.h"
#include "string2.h"
#include "pme_loadbal.h"
#include "bondf.h"
#include "membed.h"
#include "types/nlistheuristics.h"
#include "types/iteratedconstraints.h"
#include "nbnxn_cuda_data_mgmt.h"
#include "rbin.h"

#ifdef GMX_LIB_MPI
#include <mpi.h>
#endif
#ifdef GMX_THREAD_MPI
#include "tmpi.h"
#endif

#ifdef GMX_FAHCORE
#include "corewrap.h"
#endif

static void moveit(t_commrec *cr,
		   int left,int right,const char *s,rvec xx[])
{
  if (!xx) 
    return;

  move_rvecs(cr,FALSE,FALSE,left,right,
	     xx,NULL,(cr->nnodes-cr->npmenodes)-1,NULL);
}

static void reset_all_counters(FILE *fplog, t_commrec *cr,
                               gmx_large_int_t step,
                               gmx_large_int_t *step_rel, t_inputrec *ir,
                               gmx_wallcycle_t wcycle, t_nrnb *nrnb,
                               gmx_runtime_t *runtime,
                               nbnxn_cuda_ptr_t cu_nbv)
{
    char sbuf[STEPSTRSIZE];

    /* Reset all the counters related to performance over the run */
    md_print_warn(cr, fplog, "step %s: resetting all time and cycle counters\n",
                  gmx_step_str(step, sbuf));

    if (cu_nbv)
    {
        nbnxn_cuda_reset_timings(cu_nbv);
    }

    wallcycle_stop(wcycle, ewcRUN);
    wallcycle_reset_all(wcycle);
    if (DOMAINDECOMP(cr))
    {
        reset_dd_statistics_counters(cr->dd);
    }
    init_nrnb(nrnb);
    ir->init_step += *step_rel;
    ir->nsteps    -= *step_rel;
    *step_rel      = 0;
    wallcycle_start(wcycle, ewcRUN);
    runtime_start(runtime);
    print_date_and_time(fplog, cr->nodeid, "Restarted time", runtime);
}

double do_md(FILE *fplog, t_commrec *cr, int nfile, const t_filenm fnm[],
             const output_env_t oenv, gmx_bool bVerbose, gmx_bool bCompact,
             int nstglobalcomm,
             gmx_vsite_t *vsite, gmx_constr_t constr,
             int stepout, t_inputrec *ir,
             gmx_mtop_t *top_global,
             t_fcdata *fcd,
             t_state *state_global,
             t_mdatoms *mdatoms,
             t_nrnb *nrnb, gmx_wallcycle_t wcycle,
             gmx_edsam_t ed, t_forcerec *fr,
             int repl_ex_nst, int repl_ex_nex, int repl_ex_seed, gmx_membed_t membed,
             real cpt_period, real max_hours,
             const char *deviceOptions,
             unsigned long Flags,
             gmx_runtime_t *runtime)
{
    gmx_mdoutf_t   *outf;
    gmx_large_int_t step, step_rel;
    double          run_time;
    double          t, t0, lam0[efptNR];
    gmx_bool        bGStatEveryStep, bGStat, bCalcVir, bCalcEner;
    gmx_bool        bNS, bNStList, bSimAnn, bStopCM, bRerunMD, bNotLastFrame = FALSE,
                    bFirstStep, bStateFromCP, bStateFromTPX, bInitStep, bLastStep,
                    bBornRadii, bStartingFromCpt;
    gmx_bool          bDoDHDL = FALSE, bDoFEP = FALSE, bDoExpanded = FALSE;
    gmx_bool          do_ene, do_log, do_verbose, bRerunWarnNoV = TRUE,
                      bForceUpdate = FALSE, bCPT;
    int               mdof_flags;
    gmx_bool          bMasterState;
    int               force_flags, cglo_flags;
    tensor            force_vir, shake_vir, total_vir, tmp_vir, pres;
    int               i, m;
    t_trxstatus      *status;
    rvec              mu_tot;
    t_vcm            *vcm;
    t_state          *bufstate = NULL;
    matrix           *scale_tot, pcoupl_mu, M, ebox;
    gmx_nlheur_t      nlh;
    t_trxframe        rerun_fr;
    gmx_repl_ex_t     repl_ex = NULL;
    int               nchkpt  = 1;
    gmx_localtop_t   *top;
    t_mdebin         *mdebin = NULL;
    df_history_t      df_history;
    t_state          *state    = NULL;
    rvec             *f_global = NULL;
    int               n_xtc    = -1;
    rvec             *x_xtc    = NULL;
    gmx_enerdata_t   *enerd;
    rvec             *f = NULL;
    gmx_global_stat_t gstat;
    gmx_update_t      upd   = NULL;
    t_graph          *graph = NULL;
    globsig_t         gs;
    gmx_rng_t         mcrng = NULL;
    gmx_bool          bFFscan;
    gmx_groups_t     *groups;
    gmx_ekindata_t   *ekind, *ekind_save;
    gmx_shellfc_t     shellfc;
    int               count, nconverged = 0;
    real              timestep = 0;
    double            tcount   = 0;
    gmx_bool          bIonize  = FALSE;
    gmx_bool          bTCR     = FALSE, bConverged = TRUE, bOK, bSumEkinhOld, bExchanged;
    gmx_bool          bAppend;
    gmx_bool          bResetCountersHalfMaxH = FALSE;
    gmx_bool          bVV, bIterativeCase, bFirstIterate, bTemp, bPres, bTrotter;
    gmx_bool          bUpdateDoLR;
    real              mu_aver = 0, dvdl_constr;
    int               a0, a1, gnx = 0, ii;
    atom_id          *grpindex = NULL;
    char             *grpname;
    t_coupl_rec      *tcr     = NULL;
    rvec             *xcopy   = NULL, *vcopy = NULL, *cbuf = NULL;
    matrix            boxcopy = {{0}}, lastbox;
    tensor            tmpvir;
    real              fom, oldfom, veta_save, pcurr, scalevir, tracevir;
    real              vetanew = 0;
    int               lamnew  = 0;
    /* for FEP */
    int               nstfep;
    real              rate;
    double            cycles;
    real              saved_conserved_quantity = 0;
    real              last_ekin                = 0;
    int               iter_i;
    t_extmass         MassQ;
    int             **trotter_seq;
    char              sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];
    int               handled_stop_condition = gmx_stop_cond_none; /* compare to get_stop_condition*/
    int nf = 0;
    gmx_iterate_t     iterate;
    gmx_large_int_t   multisim_nsteps = -1;                        /* number of steps to do  before first multisim
                                                                      simulation stops. If equal to zero, don't
                                                                      communicate any more between multisims.*/
    /* PME load balancing data for GPU kernels */
    pme_load_balancing_t pme_loadbal = NULL;
    double               cycles_pmes;
    gmx_bool             bPMETuneTry = FALSE, bPMETuneRunning = FALSE;

#ifdef GMX_FAHCORE
    /* Temporary addition for FAHCORE checkpointing */
    int chkpt_ret;
#endif
    MPI_Request req[4];
    MPI_Status  stat[4];
    int nreq;

index_r[0] = 0;
index_r[1] = 0;
index_r[2] = 0;
index_r[3] = 0;
index_r[4] = 0;
index_r[5] = 0;
index_r[6] = 0;
index_r[7] = 0;
index_r[8] = 0;
index_r[9] = 0;
index_r[10] = 0;
index_r[11] = 0;
index_r[12] = 0;
index_r[13] = 0;
index_r[14] = 0;
index_r[15] = 0;
index_r[16] = 0;
index_r[17] = 0;
index_r[18] = 0;
index_r[19] = 0;
index_r[20] = 0;
index_r[21] = 0;
index_r[22] = 1;
index_r[23] = 1;
index_r[24] = 1;
index_r[25] = 1;
index_r[26] = 1;
index_r[27] = 1;
index_r[28] = 1;
index_r[29] = 1;
index_r[30] = 1;
index_r[31] = 1;
index_r[32] = 1;
index_r[33] = 1;
index_r[34] = 1;
index_r[35] = 1;
index_r[36] = 1;
index_r[37] = 1;
index_r[38] = 1;
index_r[39] = 1;
index_r[40] = 1;
index_r[41] = 2;
index_r[42] = 2;
index_r[43] = 2;
index_r[44] = 2;
index_r[45] = 2;
index_r[46] = 2;
index_r[47] = 2;
index_r[48] = 3;
index_r[49] = 3;
index_r[50] = 3;
index_r[51] = 3;
index_r[52] = 3;
index_r[53] = 3;
index_r[54] = 3;
index_r[55] = 3;
index_r[56] = 3;
index_r[57] = 3;
index_r[58] = 3;
index_r[59] = 3;
index_r[60] = 3;
index_r[61] = 3;
index_r[62] = 3;
index_r[63] = 4;
index_r[64] = 4;
index_r[65] = 4;
index_r[66] = 4;
index_r[67] = 4;
index_r[68] = 4;
index_r[69] = 4;
index_r[70] = 4;
index_r[71] = 4;
index_r[72] = 4;
index_r[73] = 4;
index_r[74] = 4;
index_r[75] = 4;
index_r[76] = 4;
index_r[77] = 4;
index_r[78] = 5;
index_r[79] = 5;
index_r[80] = 5;
index_r[81] = 5;
index_r[82] = 5;
index_r[83] = 5;
index_r[84] = 5;
index_r[85] = 5;
index_r[86] = 5;
index_r[87] = 5;
index_r[88] = 5;
index_r[89] = 5;
index_r[90] = 6;
index_r[91] = 6;
index_r[92] = 6;
index_r[93] = 6;
index_r[94] = 6;
index_r[95] = 6;
index_r[96] = 6;
index_r[97] = 6;
index_r[98] = 6;
index_r[99] = 6;
index_r[100] = 6;
index_r[101] = 6;
index_r[102] = 6;
index_r[103] = 6;
index_r[104] = 6;
index_r[105] = 6;
index_r[106] = 6;
index_r[107] = 6;
index_r[108] = 6;
index_r[109] = 7;
index_r[110] = 7;
index_r[111] = 7;
index_r[112] = 7;
index_r[113] = 7;
index_r[114] = 7;
index_r[115] = 7;
index_r[116] = 7;
index_r[117] = 7;
index_r[118] = 7;
index_r[119] = 7;
index_r[120] = 7;
index_r[121] = 7;
index_r[122] = 7;
index_r[123] = 8;
index_r[124] = 8;
index_r[125] = 8;
index_r[126] = 8;
index_r[127] = 8;
index_r[128] = 8;
index_r[129] = 8;
index_r[130] = 8;
index_r[131] = 8;
index_r[132] = 8;
index_r[133] = 8;
index_r[134] = 8;
index_r[135] = 8;
index_r[136] = 8;
index_r[137] = 8;
index_r[138] = 8;
index_r[139] = 8;
index_r[140] = 8;
index_r[141] = 8;
index_r[142] = 8;
index_r[143] = 8;
index_r[144] = 8;
index_r[145] = 8;
index_r[146] = 8;
index_r[147] = 9;
index_r[148] = 9;
index_r[149] = 9;
index_r[150] = 9;
index_r[151] = 9;
index_r[152] = 9;
index_r[153] = 9;
index_r[154] = 9;
index_r[155] = 9;
index_r[156] = 9;
index_r[157] = 9;
index_r[158] = 9;
index_r[159] = 9;
index_r[160] = 9;
index_r[161] = 9;
index_r[162] = 10;
index_r[163] = 10;
index_r[164] = 10;
index_r[165] = 10;
index_r[166] = 10;
index_r[167] = 10;
index_r[168] = 10;
index_r[169] = 10;
index_r[170] = 10;
index_r[171] = 10;
index_r[172] = 10;
index_r[173] = 10;
index_r[174] = 10;
index_r[175] = 10;
index_r[176] = 11;
index_r[177] = 11;
index_r[178] = 11;
index_r[179] = 11;
index_r[180] = 11;
index_r[181] = 11;
index_r[182] = 11;
index_r[183] = 11;
index_r[184] = 11;
index_r[185] = 11;
index_r[186] = 11;
index_r[187] = 11;
index_r[188] = 11;
index_r[189] = 11;
index_r[190] = 11;
index_r[191] = 11;
index_r[192] = 11;
index_r[193] = 11;
index_r[194] = 11;
index_r[195] = 11;
index_r[196] = 11;
index_r[197] = 11;
index_r[198] = 11;
index_r[199] = 11;
index_r[200] = 12;
index_r[201] = 12;
index_r[202] = 12;
index_r[203] = 12;
index_r[204] = 12;
index_r[205] = 12;
index_r[206] = 12;
index_r[207] = 12;
index_r[208] = 12;
index_r[209] = 12;
index_r[210] = 12;
index_r[211] = 12;
index_r[212] = 12;
index_r[213] = 12;
index_r[214] = 12;
index_r[215] = 12;
index_r[216] = 12;
index_r[217] = 12;
index_r[218] = 12;
index_r[219] = 12;
index_r[220] = 12;
index_r[221] = 12;
index_r[222] = 12;
index_r[223] = 12;
index_r[224] = 13;
index_r[225] = 13;
index_r[226] = 13;
index_r[227] = 13;
index_r[228] = 13;
index_r[229] = 13;
index_r[230] = 13;
index_r[231] = 13;
index_r[232] = 13;
index_r[233] = 13;
index_r[234] = 13;
index_r[235] = 13;
index_r[236] = 13;
index_r[237] = 13;
index_r[238] = 13;
index_r[239] = 13;
index_r[240] = 13;
index_r[241] = 13;
index_r[242] = 13;
index_r[243] = 14;
index_r[244] = 14;
index_r[245] = 14;
index_r[246] = 14;
index_r[247] = 14;
index_r[248] = 14;
index_r[249] = 14;
index_r[250] = 14;
index_r[251] = 14;
index_r[252] = 14;
index_r[253] = 14;
index_r[254] = 14;
index_r[255] = 14;
index_r[256] = 14;
index_r[257] = 14;
index_r[258] = 14;
index_r[259] = 15;
index_r[260] = 15;
index_r[261] = 15;
index_r[262] = 15;
index_r[263] = 15;
index_r[264] = 15;
index_r[265] = 15;
index_r[266] = 15;
index_r[267] = 15;
index_r[268] = 15;
index_r[269] = 15;
index_r[270] = 15;
index_r[271] = 15;
index_r[272] = 15;
index_r[273] = 15;
index_r[274] = 15;
index_r[275] = 15;
index_r[276] = 15;
index_r[277] = 15;
index_r[278] = 16;
index_r[279] = 16;
index_r[280] = 16;
index_r[281] = 16;
index_r[282] = 16;
index_r[283] = 16;
index_r[284] = 16;
index_r[285] = 16;
index_r[286] = 16;
index_r[287] = 16;
index_r[288] = 16;
index_r[289] = 16;
index_r[290] = 16;
index_r[291] = 16;
index_r[292] = 16;
index_r[293] = 16;
index_r[294] = 16;
index_r[295] = 17;
index_r[296] = 17;
index_r[297] = 17;
index_r[298] = 17;
index_r[299] = 17;
index_r[300] = 17;
index_r[301] = 17;
index_r[302] = 17;
index_r[303] = 17;
index_r[304] = 17;
index_r[305] = 17;
index_r[306] = 17;
index_r[307] = 17;
index_r[308] = 17;
index_r[309] = 17;
index_r[310] = 17;
index_r[311] = 17;
index_r[312] = 17;
index_r[313] = 17;
index_r[314] = 17;
index_r[315] = 17;
index_r[316] = 17;
index_r[317] = 17;
index_r[318] = 17;
index_r[319] = 18;
index_r[320] = 18;
index_r[321] = 18;
index_r[322] = 18;
index_r[323] = 18;
index_r[324] = 18;
index_r[325] = 18;
index_r[326] = 19;
index_r[327] = 19;
index_r[328] = 19;
index_r[329] = 19;
index_r[330] = 19;
index_r[331] = 19;
index_r[332] = 19;
index_r[333] = 19;
index_r[334] = 19;
index_r[335] = 19;
index_r[336] = 19;
index_r[337] = 20;
index_r[338] = 20;
index_r[339] = 20;
index_r[340] = 20;
index_r[341] = 20;
index_r[342] = 20;
index_r[343] = 20;
index_r[344] = 20;
index_r[345] = 20;
index_r[346] = 20;
index_r[347] = 20;
index_r[348] = 20;
index_r[349] = 20;
index_r[350] = 20;
index_r[351] = 21;
index_r[352] = 21;
index_r[353] = 21;
index_r[354] = 21;
index_r[355] = 21;
index_r[356] = 21;
index_r[357] = 21;
index_r[358] = 22;
index_r[359] = 22;
index_r[360] = 22;
index_r[361] = 22;
index_r[362] = 22;
index_r[363] = 22;
index_r[364] = 22;
index_r[365] = 22;
index_r[366] = 22;
index_r[367] = 22;
index_r[368] = 22;
index_r[369] = 22;
index_r[370] = 22;
index_r[371] = 22;
index_r[372] = 22;
index_r[373] = 22;
index_r[374] = 22;
index_r[375] = 22;
index_r[376] = 22;
index_r[377] = 23;
index_r[378] = 23;
index_r[379] = 23;
index_r[380] = 23;
index_r[381] = 23;
index_r[382] = 23;
index_r[383] = 23;
index_r[384] = 24;
index_r[385] = 24;
index_r[386] = 24;
index_r[387] = 24;
index_r[388] = 24;
index_r[389] = 24;
index_r[390] = 24;
index_r[391] = 24;
index_r[392] = 24;
index_r[393] = 24;
index_r[394] = 24;
index_r[395] = 24;
index_r[396] = 24;
index_r[397] = 24;
index_r[398] = 24;
index_r[399] = 24;
index_r[400] = 24;
index_r[401] = 24;
index_r[402] = 24;
index_r[403] = 24;
index_r[404] = 25;
index_r[405] = 25;
index_r[406] = 25;
index_r[407] = 25;
index_r[408] = 25;
index_r[409] = 25;
index_r[410] = 25;
index_r[411] = 25;
index_r[412] = 25;
index_r[413] = 25;
index_r[414] = 25;
index_r[415] = 25;
index_r[416] = 25;
index_r[417] = 25;
index_r[418] = 26;
index_r[419] = 26;
index_r[420] = 26;
index_r[421] = 26;
index_r[422] = 26;
index_r[423] = 26;
index_r[424] = 26;
index_r[425] = 26;
index_r[426] = 26;
index_r[427] = 26;
index_r[428] = 26;
index_r[429] = 26;
index_r[430] = 26;
index_r[431] = 26;
index_r[432] = 26;
index_r[433] = 26;
index_r[434] = 26;
index_r[435] = 26;
index_r[436] = 26;
index_r[437] = 27;
index_r[438] = 27;
index_r[439] = 27;
index_r[440] = 27;
index_r[441] = 27;
index_r[442] = 27;
index_r[443] = 27;
index_r[444] = 27;
index_r[445] = 27;
index_r[446] = 27;
index_r[447] = 27;
index_r[448] = 27;
index_r[449] = 27;
index_r[450] = 27;
index_r[451] = 27;
index_r[452] = 27;
index_r[453] = 27;
index_r[454] = 27;
index_r[455] = 27;
index_r[456] = 28;
index_r[457] = 28;
index_r[458] = 28;
index_r[459] = 28;
index_r[460] = 28;
index_r[461] = 28;
index_r[462] = 28;
index_r[463] = 29;
index_r[464] = 29;
index_r[465] = 29;
index_r[466] = 29;
index_r[467] = 29;
index_r[468] = 29;
index_r[469] = 29;
index_r[470] = 30;
index_r[471] = 30;
index_r[472] = 30;
index_r[473] = 30;
index_r[474] = 30;
index_r[475] = 30;
index_r[476] = 30;
index_r[477] = 30;
index_r[478] = 30;
index_r[479] = 30;
index_r[480] = 30;
index_r[481] = 30;
index_r[482] = 30;
index_r[483] = 30;
index_r[484] = 30;
index_r[485] = 31;
index_r[486] = 31;
index_r[487] = 31;
index_r[488] = 31;
index_r[489] = 31;
index_r[490] = 31;
index_r[491] = 31;
index_r[492] = 31;
index_r[493] = 31;
index_r[494] = 31;
index_r[495] = 31;
index_r[496] = 31;
index_r[497] = 32;
index_r[498] = 32;
index_r[499] = 32;
index_r[500] = 32;
index_r[501] = 32;
index_r[502] = 32;
index_r[503] = 32;
index_r[504] = 33;
index_r[505] = 33;
index_r[506] = 33;
index_r[507] = 33;
index_r[508] = 33;
index_r[509] = 33;
index_r[510] = 33;
index_r[511] = 33;
index_r[512] = 33;
index_r[513] = 33;
index_r[514] = 33;
index_r[515] = 33;
index_r[516] = 33;
index_r[517] = 33;
index_r[518] = 33;
index_r[519] = 34;
index_r[520] = 34;
index_r[521] = 34;
index_r[522] = 34;
index_r[523] = 34;
index_r[524] = 34;
index_r[525] = 34;
index_r[526] = 35;
index_r[527] = 35;
index_r[528] = 35;
index_r[529] = 35;
index_r[530] = 35;
index_r[531] = 35;
index_r[532] = 35;
index_r[533] = 35;
index_r[534] = 35;
index_r[535] = 35;
index_r[536] = 35;
index_r[537] = 35;
index_r[538] = 35;
index_r[539] = 35;
index_r[540] = 35;
index_r[541] = 35;
index_r[542] = 35;
index_r[543] = 35;
index_r[544] = 35;
index_r[545] = 36;
index_r[546] = 36;
index_r[547] = 36;
index_r[548] = 36;
index_r[549] = 36;
index_r[550] = 36;
index_r[551] = 36;
index_r[552] = 36;
index_r[553] = 36;
index_r[554] = 36;
index_r[555] = 36;
index_r[556] = 36;
index_r[557] = 36;
index_r[558] = 36;
index_r[559] = 36;
index_r[560] = 36;
index_r[561] = 36;
index_r[562] = 36;
index_r[563] = 36;
index_r[564] = 36;
index_r[565] = 37;
index_r[566] = 37;
index_r[567] = 37;
index_r[568] = 37;
index_r[569] = 37;
index_r[570] = 37;
index_r[571] = 37;
index_r[572] = 37;
index_r[573] = 37;
index_r[574] = 37;
index_r[575] = 37;
index_r[576] = 37;
index_r[577] = 37;
index_r[578] = 37;
index_r[579] = 37;
index_r[580] = 37;
index_r[581] = 37;
index_r[582] = 37;
index_r[583] = 37;
index_r[584] = 38;
index_r[585] = 38;
index_r[586] = 38;
index_r[587] = 38;
index_r[588] = 38;
index_r[589] = 38;
index_r[590] = 38;
index_r[591] = 38;
index_r[592] = 38;
index_r[593] = 38;
index_r[594] = 38;
index_r[595] = 39;
index_r[596] = 39;
index_r[597] = 39;
index_r[598] = 39;
index_r[599] = 39;
index_r[600] = 39;
index_r[601] = 39;
index_r[602] = 39;
index_r[603] = 39;
index_r[604] = 39;
index_r[605] = 39;
index_r[606] = 39;
index_r[607] = 39;
index_r[608] = 39;
index_r[609] = 39;
index_r[610] = 39;
index_r[611] = 39;
index_r[612] = 39;
index_r[613] = 39;
index_r[614] = 39;
index_r[615] = 40;
index_r[616] = 40;
index_r[617] = 40;
index_r[618] = 40;
index_r[619] = 40;
index_r[620] = 40;
index_r[621] = 40;
index_r[622] = 40;
index_r[623] = 40;
index_r[624] = 40;
index_r[625] = 40;
index_r[626] = 40;
index_r[627] = 40;
index_r[628] = 40;
index_r[629] = 40;
index_r[630] = 40;
index_r[631] = 40;
index_r[632] = 40;
index_r[633] = 40;
index_r[634] = 41;
index_r[635] = 41;
index_r[636] = 41;
index_r[637] = 41;
index_r[638] = 41;
index_r[639] = 41;
index_r[640] = 41;
index_r[641] = 41;
index_r[642] = 41;
index_r[643] = 41;
index_r[644] = 41;
index_r[645] = 41;
index_r[646] = 41;
index_r[647] = 41;
index_r[648] = 41;
index_r[649] = 41;
index_r[650] = 41;
index_r[651] = 41;
index_r[652] = 41;
index_r[653] = 42;
index_r[654] = 42;
index_r[655] = 42;
index_r[656] = 42;
index_r[657] = 42;
index_r[658] = 42;
index_r[659] = 42;
index_r[660] = 42;
index_r[661] = 42;
index_r[662] = 42;
index_r[663] = 43;
index_r[664] = 43;
index_r[665] = 43;
index_r[666] = 43;
index_r[667] = 43;
index_r[668] = 43;
index_r[669] = 43;
index_r[670] = 44;
index_r[671] = 44;
index_r[672] = 44;
index_r[673] = 44;
index_r[674] = 44;
index_r[675] = 44;
index_r[676] = 44;
index_r[677] = 45;
index_r[678] = 45;
index_r[679] = 45;
index_r[680] = 45;
index_r[681] = 45;
index_r[682] = 45;
index_r[683] = 45;
index_r[684] = 45;
index_r[685] = 45;
index_r[686] = 45;
index_r[687] = 45;
index_r[688] = 45;
index_r[689] = 45;
index_r[690] = 45;
index_r[691] = 46;
index_r[692] = 46;
index_r[693] = 46;
index_r[694] = 46;
index_r[695] = 46;
index_r[696] = 46;
index_r[697] = 46;
index_r[698] = 46;
index_r[699] = 46;
index_r[700] = 46;
index_r[701] = 47;
index_r[702] = 47;
index_r[703] = 47;
index_r[704] = 47;
index_r[705] = 47;
index_r[706] = 47;
index_r[707] = 47;
index_r[708] = 47;
index_r[709] = 47;
index_r[710] = 47;
index_r[711] = 47;
index_r[712] = 47;
index_r[713] = 48;
index_r[714] = 48;
index_r[715] = 48;
index_r[716] = 48;
index_r[717] = 48;
index_r[718] = 48;
index_r[719] = 48;
index_r[720] = 48;
index_r[721] = 48;
index_r[722] = 48;
index_r[723] = 48;
index_r[724] = 48;
index_r[725] = 48;
index_r[726] = 48;
index_r[727] = 48;
index_r[728] = 48;
index_r[729] = 48;
index_r[730] = 48;
index_r[731] = 48;
index_r[732] = 49;
index_r[733] = 49;
index_r[734] = 49;
index_r[735] = 49;
index_r[736] = 49;
index_r[737] = 49;
index_r[738] = 49;
index_r[739] = 49;
index_r[740] = 49;
index_r[741] = 49;
index_r[742] = 49;
index_r[743] = 50;
index_r[744] = 50;
index_r[745] = 50;
index_r[746] = 50;
index_r[747] = 50;
index_r[748] = 50;
index_r[749] = 50;
index_r[750] = 51;
index_r[751] = 51;
index_r[752] = 51;
index_r[753] = 51;
index_r[754] = 51;
index_r[755] = 51;
index_r[756] = 51;
index_r[757] = 51;
index_r[758] = 51;
index_r[759] = 51;
index_r[760] = 51;
index_r[761] = 51;
index_r[762] = 51;
index_r[763] = 51;
index_r[764] = 51;
index_r[765] = 52;
index_r[766] = 52;
index_r[767] = 52;
index_r[768] = 52;
index_r[769] = 52;
index_r[770] = 52;
index_r[771] = 52;
index_r[772] = 52;
index_r[773] = 52;
index_r[774] = 52;
index_r[775] = 52;
index_r[776] = 52;
index_r[777] = 52;
index_r[778] = 52;
index_r[779] = 52;
index_r[780] = 52;
index_r[781] = 52;
index_r[782] = 52;
index_r[783] = 52;
index_r[784] = 53;
index_r[785] = 53;
index_r[786] = 53;
index_r[787] = 53;
index_r[788] = 53;
index_r[789] = 53;
index_r[790] = 53;
index_r[791] = 53;
index_r[792] = 53;
index_r[793] = 53;
index_r[794] = 53;
index_r[795] = 53;
index_r[796] = 53;
index_r[797] = 53;
index_r[798] = 53;
index_r[799] = 53;
index_r[800] = 53;
index_r[801] = 53;
index_r[802] = 53;
index_r[803] = 53;
index_r[804] = 53;
index_r[805] = 53;
index_r[806] = 53;
index_r[807] = 53;
index_r[808] = 54;
index_r[809] = 54;
index_r[810] = 54;
index_r[811] = 54;
index_r[812] = 54;
index_r[813] = 54;
index_r[814] = 54;
index_r[815] = 54;
index_r[816] = 54;
index_r[817] = 54;
index_r[818] = 54;
index_r[819] = 54;
index_r[820] = 54;
index_r[821] = 54;
index_r[822] = 54;
index_r[823] = 54;
index_r[824] = 54;
index_r[825] = 54;
index_r[826] = 54;
index_r[827] = 54;
index_r[828] = 54;
index_r[829] = 54;
index_r[830] = 55;
index_r[831] = 55;
index_r[832] = 55;
index_r[833] = 55;
index_r[834] = 55;
index_r[835] = 55;
index_r[836] = 55;
index_r[837] = 56;
index_r[838] = 56;
index_r[839] = 56;
index_r[840] = 56;
index_r[841] = 56;
index_r[842] = 56;
index_r[843] = 56;
index_r[844] = 56;
index_r[845] = 56;
index_r[846] = 56;
index_r[847] = 56;
index_r[848] = 56;
index_r[849] = 57;
index_r[850] = 57;
index_r[851] = 57;
index_r[852] = 57;
index_r[853] = 57;
index_r[854] = 57;
index_r[855] = 57;
index_r[856] = 57;
index_r[857] = 57;
index_r[858] = 57;
index_r[859] = 57;
index_r[860] = 57;
index_r[861] = 57;
index_r[862] = 57;
index_r[863] = 57;
index_r[864] = 57;
index_r[865] = 57;
index_r[866] = 58;
index_r[867] = 58;
index_r[868] = 58;
index_r[869] = 58;
index_r[870] = 58;
index_r[871] = 58;
index_r[872] = 58;
index_r[873] = 58;
index_r[874] = 58;
index_r[875] = 58;
index_r[876] = 58;
index_r[877] = 58;
index_r[878] = 58;
index_r[879] = 58;
index_r[880] = 58;
index_r[881] = 58;
index_r[882] = 58;
index_r[883] = 58;
index_r[884] = 58;
index_r[885] = 59;
index_r[886] = 59;
index_r[887] = 59;
index_r[888] = 59;
index_r[889] = 59;
index_r[890] = 59;
index_r[891] = 59;
index_r[892] = 59;
index_r[893] = 59;
index_r[894] = 59;
index_r[895] = 59;
index_r[896] = 59;
index_r[897] = 59;
index_r[898] = 59;
index_r[899] = 59;
index_r[900] = 59;
index_r[901] = 59;
index_r[902] = 59;
index_r[903] = 59;
index_r[904] = 60;
index_r[905] = 60;
index_r[906] = 60;
index_r[907] = 60;
index_r[908] = 60;
index_r[909] = 60;
index_r[910] = 60;
index_r[911] = 60;
index_r[912] = 60;
index_r[913] = 60;
index_r[914] = 60;
index_r[915] = 61;
index_r[916] = 61;
index_r[917] = 61;
index_r[918] = 61;
index_r[919] = 61;
index_r[920] = 61;
index_r[921] = 61;
index_r[922] = 61;
index_r[923] = 61;
index_r[924] = 61;
index_r[925] = 61;
index_r[926] = 61;
index_r[927] = 61;
index_r[928] = 61;
index_r[929] = 61;
index_r[930] = 61;
index_r[931] = 62;
index_r[932] = 62;
index_r[933] = 62;
index_r[934] = 62;
index_r[935] = 62;
index_r[936] = 62;
index_r[937] = 62;
index_r[938] = 62;
index_r[939] = 62;
index_r[940] = 62;
index_r[941] = 62;
index_r[942] = 62;
index_r[943] = 62;
index_r[944] = 62;
index_r[945] = 63;
index_r[946] = 63;
index_r[947] = 63;
index_r[948] = 63;
index_r[949] = 63;
index_r[950] = 63;
index_r[951] = 63;
index_r[952] = 64;
index_r[953] = 64;
index_r[954] = 64;
index_r[955] = 64;
index_r[956] = 64;
index_r[957] = 64;
index_r[958] = 64;
index_r[959] = 64;
index_r[960] = 64;
index_r[961] = 64;
index_r[962] = 64;
index_r[963] = 64;
index_r[964] = 64;
index_r[965] = 64;
index_r[966] = 64;
index_r[967] = 64;
index_r[968] = 65;
index_r[969] = 65;
index_r[970] = 65;
index_r[971] = 65;
index_r[972] = 65;
index_r[973] = 65;
index_r[974] = 65;
index_r[975] = 65;
index_r[976] = 65;
index_r[977] = 65;
index_r[978] = 65;
index_r[979] = 65;
index_r[980] = 66;
index_r[981] = 66;
index_r[982] = 66;
index_r[983] = 66;
index_r[984] = 66;
index_r[985] = 66;
index_r[986] = 66;
index_r[987] = 66;
index_r[988] = 66;
index_r[989] = 66;
index_r[990] = 66;
index_r[991] = 66;
index_r[992] = 66;
index_r[993] = 66;
index_r[994] = 66;
index_r[995] = 66;
index_r[996] = 66;
index_r[997] = 66;
index_r[998] = 66;
index_r[999] = 67;
index_r[1000] = 67;
index_r[1001] = 67;
index_r[1002] = 67;
index_r[1003] = 67;
index_r[1004] = 67;
index_r[1005] = 67;
index_r[1006] = 67;
index_r[1007] = 67;
index_r[1008] = 67;
index_r[1009] = 67;
index_r[1010] = 67;
index_r[1011] = 67;
index_r[1012] = 67;
index_r[1013] = 67;
index_r[1014] = 67;
index_r[1015] = 67;
index_r[1016] = 67;
index_r[1017] = 67;
index_r[1018] = 67;
index_r[1019] = 67;
index_r[1020] = 67;
index_r[1021] = 67;
index_r[1022] = 67;
index_r[1023] = 68;
index_r[1024] = 68;
index_r[1025] = 68;
index_r[1026] = 68;
index_r[1027] = 68;
index_r[1028] = 68;
index_r[1029] = 68;
index_r[1030] = 68;
index_r[1031] = 68;
index_r[1032] = 68;
index_r[1033] = 68;
index_r[1034] = 68;
index_r[1035] = 68;
index_r[1036] = 68;
index_r[1037] = 69;
index_r[1038] = 69;
index_r[1039] = 69;
index_r[1040] = 69;
index_r[1041] = 69;
index_r[1042] = 69;
index_r[1043] = 69;
index_r[1044] = 69;
index_r[1045] = 69;
index_r[1046] = 69;
index_r[1047] = 70;
index_r[1048] = 70;
index_r[1049] = 70;
index_r[1050] = 70;
index_r[1051] = 70;
index_r[1052] = 70;
index_r[1053] = 70;
index_r[1054] = 70;
index_r[1055] = 70;
index_r[1056] = 70;
index_r[1057] = 70;
index_r[1058] = 71;
index_r[1059] = 71;
index_r[1060] = 71;
index_r[1061] = 71;
index_r[1062] = 71;
index_r[1063] = 71;
index_r[1064] = 71;
index_r[1065] = 71;
index_r[1066] = 71;
index_r[1067] = 71;
index_r[1068] = 71;
index_r[1069] = 71;
index_r[1070] = 71;
index_r[1071] = 71;
index_r[1072] = 71;
index_r[1073] = 71;
index_r[1074] = 71;
index_r[1075] = 72;
index_r[1076] = 72;
index_r[1077] = 72;
index_r[1078] = 72;
index_r[1079] = 72;
index_r[1080] = 72;
index_r[1081] = 72;
index_r[1082] = 72;
index_r[1083] = 72;
index_r[1084] = 72;
index_r[1085] = 72;
index_r[1086] = 72;
index_r[1087] = 72;
index_r[1088] = 72;
index_r[1089] = 72;
index_r[1090] = 73;
index_r[1091] = 73;
index_r[1092] = 73;
index_r[1093] = 73;
index_r[1094] = 73;
index_r[1095] = 73;
index_r[1096] = 73;
index_r[1097] = 73;
index_r[1098] = 73;
index_r[1099] = 73;
index_r[1100] = 73;
index_r[1101] = 73;
index_r[1102] = 73;
index_r[1103] = 73;
index_r[1104] = 73;
index_r[1105] = 73;
index_r[1106] = 73;
index_r[1107] = 74;
index_r[1108] = 74;
index_r[1109] = 74;
index_r[1110] = 74;
index_r[1111] = 74;
index_r[1112] = 74;
index_r[1113] = 74;
index_r[1114] = 74;
index_r[1115] = 74;
index_r[1116] = 74;
index_r[1117] = 75;
index_r[1118] = 75;
index_r[1119] = 75;
index_r[1120] = 75;
index_r[1121] = 75;
index_r[1122] = 75;
index_r[1123] = 75;
index_r[1124] = 75;
index_r[1125] = 75;
index_r[1126] = 75;
index_r[1127] = 76;
index_r[1128] = 76;
index_r[1129] = 76;
index_r[1130] = 76;
index_r[1131] = 76;
index_r[1132] = 76;
index_r[1133] = 76;
index_r[1134] = 76;
index_r[1135] = 76;
index_r[1136] = 76;
index_r[1137] = 76;
index_r[1138] = 76;
index_r[1139] = 76;
index_r[1140] = 76;
index_r[1141] = 76;
index_r[1142] = 76;
index_r[1143] = 76;
index_r[1144] = 76;
index_r[1145] = 76;
index_r[1146] = 77;
index_r[1147] = 77;
index_r[1148] = 77;
index_r[1149] = 77;
index_r[1150] = 77;
index_r[1151] = 77;
index_r[1152] = 77;
index_r[1153] = 77;
index_r[1154] = 77;
index_r[1155] = 77;
index_r[1156] = 78;
index_r[1157] = 78;
index_r[1158] = 78;
index_r[1159] = 78;
index_r[1160] = 78;
index_r[1161] = 78;
index_r[1162] = 78;
index_r[1163] = 78;
index_r[1164] = 78;
index_r[1165] = 78;
index_r[1166] = 78;
index_r[1167] = 78;
index_r[1168] = 78;
index_r[1169] = 78;
index_r[1170] = 78;
index_r[1171] = 78;
index_r[1172] = 78;
index_r[1173] = 78;
index_r[1174] = 78;
index_r[1175] = 79;
index_r[1176] = 79;
index_r[1177] = 79;
index_r[1178] = 79;
index_r[1179] = 79;
index_r[1180] = 79;
index_r[1181] = 79;
index_r[1182] = 79;
index_r[1183] = 79;
index_r[1184] = 79;
index_r[1185] = 79;
index_r[1186] = 79;
index_r[1187] = 79;
index_r[1188] = 79;
index_r[1189] = 79;
index_r[1190] = 79;
index_r[1191] = 79;
index_r[1192] = 79;
index_r[1193] = 79;
index_r[1194] = 79;
index_r[1195] = 79;
index_r[1196] = 79;
index_r[1197] = 80;
index_r[1198] = 80;
index_r[1199] = 80;
index_r[1200] = 80;
index_r[1201] = 80;
index_r[1202] = 80;
index_r[1203] = 80;
index_r[1204] = 80;
index_r[1205] = 80;
index_r[1206] = 80;
index_r[1207] = 80;
index_r[1208] = 80;
index_r[1209] = 80;
index_r[1210] = 80;
index_r[1211] = 81;
index_r[1212] = 81;
index_r[1213] = 81;
index_r[1214] = 81;
index_r[1215] = 81;
index_r[1216] = 81;
index_r[1217] = 81;
index_r[1218] = 81;
index_r[1219] = 81;
index_r[1220] = 81;
index_r[1221] = 82;
index_r[1222] = 82;
index_r[1223] = 82;
index_r[1224] = 82;
index_r[1225] = 82;
index_r[1226] = 82;
index_r[1227] = 82;
index_r[1228] = 83;
index_r[1229] = 83;
index_r[1230] = 83;
index_r[1231] = 83;
index_r[1232] = 83;
index_r[1233] = 83;
index_r[1234] = 83;
index_r[1235] = 83;
index_r[1236] = 83;
index_r[1237] = 83;
index_r[1238] = 83;
index_r[1239] = 83;
index_r[1240] = 83;
index_r[1241] = 83;
index_r[1242] = 83;
index_r[1243] = 83;
index_r[1244] = 83;
index_r[1245] = 84;
index_r[1246] = 84;
index_r[1247] = 84;
index_r[1248] = 84;
index_r[1249] = 84;
index_r[1250] = 84;
index_r[1251] = 84;
index_r[1252] = 84;
index_r[1253] = 84;
index_r[1254] = 84;
index_r[1255] = 84;
index_r[1256] = 84;
index_r[1257] = 84;
index_r[1258] = 84;
index_r[1259] = 85;
index_r[1260] = 85;
index_r[1261] = 85;
index_r[1262] = 85;
index_r[1263] = 85;
index_r[1264] = 85;
index_r[1265] = 85;
index_r[1266] = 85;
index_r[1267] = 85;
index_r[1268] = 85;
index_r[1269] = 85;
index_r[1270] = 85;
index_r[1271] = 85;
index_r[1272] = 85;
index_r[1273] = 85;
index_r[1274] = 85;
index_r[1275] = 86;
index_r[1276] = 86;
index_r[1277] = 86;
index_r[1278] = 86;
index_r[1279] = 86;
index_r[1280] = 86;
index_r[1281] = 86;
index_r[1282] = 86;
index_r[1283] = 86;
index_r[1284] = 86;
index_r[1285] = 86;
index_r[1286] = 86;
index_r[1287] = 86;
index_r[1288] = 86;
index_r[1289] = 87;
index_r[1290] = 87;
index_r[1291] = 87;
index_r[1292] = 87;
index_r[1293] = 87;
index_r[1294] = 87;
index_r[1295] = 87;
index_r[1296] = 87;
index_r[1297] = 87;
index_r[1298] = 87;
index_r[1299] = 87;
index_r[1300] = 87;
index_r[1301] = 87;
index_r[1302] = 87;
index_r[1303] = 87;
index_r[1304] = 87;
index_r[1305] = 87;
index_r[1306] = 87;
index_r[1307] = 87;
index_r[1308] = 88;
index_r[1309] = 88;
index_r[1310] = 88;
index_r[1311] = 88;
index_r[1312] = 88;
index_r[1313] = 88;
index_r[1314] = 88;
index_r[1315] = 88;
index_r[1316] = 88;
index_r[1317] = 88;
index_r[1318] = 88;
index_r[1319] = 88;
index_r[1320] = 88;
index_r[1321] = 88;
index_r[1322] = 88;
index_r[1323] = 88;
index_r[1324] = 88;
index_r[1325] = 88;
index_r[1326] = 88;
index_r[1327] = 89;
index_r[1328] = 89;
index_r[1329] = 89;
index_r[1330] = 89;
index_r[1331] = 89;
index_r[1332] = 89;
index_r[1333] = 89;
index_r[1334] = 89;
index_r[1335] = 89;
index_r[1336] = 89;
index_r[1337] = 90;
index_r[1338] = 90;
index_r[1339] = 90;
index_r[1340] = 90;
index_r[1341] = 90;
index_r[1342] = 90;
index_r[1343] = 90;
index_r[1344] = 90;
index_r[1345] = 90;
index_r[1346] = 90;
index_r[1347] = 90;
index_r[1348] = 90;
index_r[1349] = 90;
index_r[1350] = 90;
index_r[1351] = 90;
index_r[1352] = 90;
index_r[1353] = 90;
index_r[1354] = 91;
index_r[1355] = 91;
index_r[1356] = 91;
index_r[1357] = 91;
index_r[1358] = 91;
index_r[1359] = 91;
index_r[1360] = 91;
index_r[1361] = 91;
index_r[1362] = 91;
index_r[1363] = 91;
index_r[1364] = 91;
index_r[1365] = 91;
index_r[1366] = 91;
index_r[1367] = 91;
index_r[1368] = 91;
index_r[1369] = 91;
index_r[1370] = 91;
index_r[1371] = 91;
index_r[1372] = 91;
index_r[1373] = 91;
index_r[1374] = 91;
index_r[1375] = 92;
index_r[1376] = 92;
index_r[1377] = 92;
index_r[1378] = 92;
index_r[1379] = 92;
index_r[1380] = 92;
index_r[1381] = 92;
index_r[1382] = 92;
index_r[1383] = 92;
index_r[1384] = 92;
index_r[1385] = 92;
index_r[1386] = 92;
index_r[1387] = 92;
index_r[1388] = 92;
index_r[1389] = 92;
index_r[1390] = 92;
index_r[1391] = 92;
index_r[1392] = 92;
index_r[1393] = 92;
index_r[1394] = 92;
index_r[1395] = 92;
index_r[1396] = 92;
index_r[1397] = 93;
index_r[1398] = 93;
index_r[1399] = 93;
index_r[1400] = 93;
index_r[1401] = 93;
index_r[1402] = 93;
index_r[1403] = 93;
index_r[1404] = 93;
index_r[1405] = 93;
index_r[1406] = 93;
index_r[1407] = 93;
index_r[1408] = 93;
index_r[1409] = 93;
index_r[1410] = 93;
index_r[1411] = 94;
index_r[1412] = 94;
index_r[1413] = 94;
index_r[1414] = 94;
index_r[1415] = 94;
index_r[1416] = 94;
index_r[1417] = 94;
index_r[1418] = 94;
index_r[1419] = 94;
index_r[1420] = 94;
index_r[1421] = 94;
index_r[1422] = 94;
index_r[1423] = 94;
index_r[1424] = 94;
index_r[1425] = 94;
index_r[1426] = 95;
index_r[1427] = 95;
index_r[1428] = 95;
index_r[1429] = 95;
index_r[1430] = 95;
index_r[1431] = 95;
index_r[1432] = 95;
index_r[1433] = 95;
index_r[1434] = 95;
index_r[1435] = 95;
index_r[1436] = 95;
index_r[1437] = 95;
index_r[1438] = 95;
index_r[1439] = 95;
index_r[1440] = 95;
index_r[1441] = 96;
index_r[1442] = 96;
index_r[1443] = 96;
index_r[1444] = 96;
index_r[1445] = 96;
index_r[1446] = 96;
index_r[1447] = 96;
index_r[1448] = 96;
index_r[1449] = 96;
index_r[1450] = 96;
index_r[1451] = 96;
index_r[1452] = 96;
index_r[1453] = 96;
index_r[1454] = 96;
index_r[1455] = 96;
index_r[1456] = 96;
index_r[1457] = 96;
index_r[1458] = 96;
index_r[1459] = 96;
index_r[1460] = 96;
index_r[1461] = 96;
index_r[1462] = 97;
index_r[1463] = 97;
index_r[1464] = 97;
index_r[1465] = 97;
index_r[1466] = 97;
index_r[1467] = 97;
index_r[1468] = 97;
index_r[1469] = 97;
index_r[1470] = 97;
index_r[1471] = 97;
index_r[1472] = 97;
index_r[1473] = 98;
index_r[1474] = 98;
index_r[1475] = 98;
index_r[1476] = 98;
index_r[1477] = 98;
index_r[1478] = 98;
index_r[1479] = 98;
index_r[1480] = 98;
index_r[1481] = 98;
index_r[1482] = 98;
index_r[1483] = 98;
index_r[1484] = 98;
index_r[1485] = 98;
index_r[1486] = 98;
index_r[1487] = 98;
index_r[1488] = 98;
index_r[1489] = 98;
index_r[1490] = 98;
index_r[1491] = 98;
index_r[1492] = 98;
index_r[1493] = 98;
index_r[1494] = 98;
index_r[1495] = 98;
index_r[1496] = 98;
index_r[1497] = 99;
index_r[1498] = 99;
index_r[1499] = 99;
index_r[1500] = 99;
index_r[1501] = 99;
index_r[1502] = 99;
index_r[1503] = 99;
index_r[1504] = 99;
index_r[1505] = 99;
index_r[1506] = 99;
index_r[1507] = 99;
index_r[1508] = 99;
index_r[1509] = 99;
index_r[1510] = 99;
index_r[1511] = 99;
index_r[1512] = 99;
index_r[1513] = 99;
index_r[1514] = 99;
index_r[1515] = 99;
index_r[1516] = 99;
index_r[1517] = 100;
index_r[1518] = 100;
index_r[1519] = 100;
index_r[1520] = 100;
index_r[1521] = 100;
index_r[1522] = 100;
index_r[1523] = 100;
index_r[1524] = 100;
index_r[1525] = 100;
index_r[1526] = 100;
index_r[1527] = 100;
index_r[1528] = 100;
index_r[1529] = 100;
index_r[1530] = 100;
index_r[1531] = 100;
index_r[1532] = 101;
index_r[1533] = 101;
index_r[1534] = 101;
index_r[1535] = 101;
index_r[1536] = 101;
index_r[1537] = 101;
index_r[1538] = 101;
index_r[1539] = 101;
index_r[1540] = 101;
index_r[1541] = 101;
index_r[1542] = 102;
index_r[1543] = 102;
index_r[1544] = 102;
index_r[1545] = 102;
index_r[1546] = 102;
index_r[1547] = 102;
index_r[1548] = 102;
index_r[1549] = 102;
index_r[1550] = 102;
index_r[1551] = 102;
index_r[1552] = 102;
index_r[1553] = 102;
index_r[1554] = 102;
index_r[1555] = 102;
index_r[1556] = 103;
index_r[1557] = 103;
index_r[1558] = 103;
index_r[1559] = 103;
index_r[1560] = 103;
index_r[1561] = 103;
index_r[1562] = 103;
index_r[1563] = 103;
index_r[1564] = 103;
index_r[1565] = 103;
index_r[1566] = 103;
index_r[1567] = 104;
index_r[1568] = 104;
index_r[1569] = 104;
index_r[1570] = 104;
index_r[1571] = 104;
index_r[1572] = 104;
index_r[1573] = 104;
index_r[1574] = 104;
index_r[1575] = 104;
index_r[1576] = 104;
index_r[1577] = 104;
index_r[1578] = 104;
index_r[1579] = 104;
index_r[1580] = 104;
index_r[1581] = 104;
index_r[1582] = 104;
index_r[1583] = 104;
index_r[1584] = 104;
index_r[1585] = 104;
index_r[1586] = 104;
index_r[1587] = 104;
index_r[1588] = 104;
index_r[1589] = 104;
index_r[1590] = 104;
index_r[1591] = 105;
index_r[1592] = 105;
index_r[1593] = 105;
index_r[1594] = 105;
index_r[1595] = 105;
index_r[1596] = 105;
index_r[1597] = 105;
index_r[1598] = 105;
index_r[1599] = 105;
index_r[1600] = 105;
index_r[1601] = 105;
index_r[1602] = 105;
index_r[1603] = 105;
index_r[1604] = 105;
index_r[1605] = 105;
index_r[1606] = 105;
index_r[1607] = 106;
index_r[1608] = 106;
index_r[1609] = 106;
index_r[1610] = 106;
index_r[1611] = 106;
index_r[1612] = 106;
index_r[1613] = 106;
index_r[1614] = 106;
index_r[1615] = 106;
index_r[1616] = 106;
index_r[1617] = 106;
index_r[1618] = 106;
index_r[1619] = 106;
index_r[1620] = 106;
index_r[1621] = 107;
index_r[1622] = 107;
index_r[1623] = 107;
index_r[1624] = 107;
index_r[1625] = 107;
index_r[1626] = 107;
index_r[1627] = 107;
index_r[1628] = 107;
index_r[1629] = 107;
index_r[1630] = 107;
index_r[1631] = 107;
index_r[1632] = 108;
index_r[1633] = 108;
index_r[1634] = 108;
index_r[1635] = 108;
index_r[1636] = 108;
index_r[1637] = 108;
index_r[1638] = 108;
index_r[1639] = 108;
index_r[1640] = 108;
index_r[1641] = 108;
index_r[1642] = 108;
index_r[1643] = 109;
index_r[1644] = 109;
index_r[1645] = 109;
index_r[1646] = 109;
index_r[1647] = 109;
index_r[1648] = 109;
index_r[1649] = 109;
index_r[1650] = 110;
index_r[1651] = 110;
index_r[1652] = 110;
index_r[1653] = 110;
index_r[1654] = 110;
index_r[1655] = 110;
index_r[1656] = 110;
index_r[1657] = 110;
index_r[1658] = 110;
index_r[1659] = 110;
index_r[1660] = 110;
index_r[1661] = 110;
index_r[1662] = 110;
index_r[1663] = 110;
index_r[1664] = 110;
index_r[1665] = 110;
index_r[1666] = 110;
index_r[1667] = 110;
index_r[1668] = 110;
index_r[1669] = 110;
index_r[1670] = 110;
index_r[1671] = 110;
index_r[1672] = 110;
index_r[1673] = 110;
index_r[1674] = 111;
index_r[1675] = 111;
index_r[1676] = 111;
index_r[1677] = 111;
index_r[1678] = 111;
index_r[1679] = 111;
index_r[1680] = 111;
index_r[1681] = 111;
index_r[1682] = 111;
index_r[1683] = 111;
index_r[1684] = 111;
index_r[1685] = 111;
index_r[1686] = 111;
index_r[1687] = 111;
index_r[1688] = 111;
index_r[1689] = 111;
index_r[1690] = 111;
index_r[1691] = 111;
index_r[1692] = 111;
index_r[1693] = 112;
index_r[1694] = 112;
index_r[1695] = 112;
index_r[1696] = 112;
index_r[1697] = 112;
index_r[1698] = 112;
index_r[1699] = 112;
index_r[1700] = 112;
index_r[1701] = 112;
index_r[1702] = 112;
index_r[1703] = 112;
index_r[1704] = 112;
index_r[1705] = 112;
index_r[1706] = 112;
index_r[1707] = 112;
index_r[1708] = 112;
index_r[1709] = 113;
index_r[1710] = 113;
index_r[1711] = 113;
index_r[1712] = 113;
index_r[1713] = 113;
index_r[1714] = 113;
index_r[1715] = 113;
index_r[1716] = 113;
index_r[1717] = 113;
index_r[1718] = 113;
index_r[1719] = 113;
index_r[1720] = 113;
index_r[1721] = 113;
index_r[1722] = 113;
index_r[1723] = 114;
index_r[1724] = 114;
index_r[1725] = 114;
index_r[1726] = 114;
index_r[1727] = 114;
index_r[1728] = 114;
index_r[1729] = 114;
index_r[1730] = 114;
index_r[1731] = 114;
index_r[1732] = 114;
index_r[1733] = 114;
index_r[1734] = 114;
index_r[1735] = 114;
index_r[1736] = 114;
index_r[1737] = 114;

int nres = 116;

/*index_r[0] = 0;
index_r[1] = 0;
index_r[2] = 0;
index_r[3] = 0;
index_r[4] = 0;
index_r[5] = 0;
index_r[6] = 0;
index_r[7] = 0;
index_r[8] = 0;
index_r[9] = 0;
index_r[10] = 0;
index_r[11] = 0;
index_r[12] = 0;
index_r[13] = 0;
index_r[14] = 0;
index_r[15] = 0;
index_r[16] = 1;
index_r[17] = 1;
index_r[18] = 1;
index_r[19] = 1;
index_r[20] = 1;
index_r[21] = 1;
index_r[22] = 1;
index_r[23] = 1;
index_r[24] = 1;
index_r[25] = 1;
index_r[26] = 1;
index_r[27] = 1;
index_r[28] = 2;
index_r[29] = 2;
index_r[30] = 2;
index_r[31] = 2;
index_r[32] = 2;
index_r[33] = 2;
index_r[34] = 2;
index_r[35] = 2;
index_r[36] = 2;
index_r[37] = 2;
index_r[38] = 2;
index_r[39] = 2;
index_r[40] = 2;
index_r[41] = 2;
index_r[42] = 3;
index_r[43] = 3;
index_r[44] = 3;
index_r[45] = 3;
index_r[46] = 3;
index_r[47] = 3;
index_r[48] = 3;
index_r[49] = 3;
index_r[50] = 3;
index_r[51] = 3;
index_r[52] = 3;
index_r[53] = 3;
index_r[54] = 3;
index_r[55] = 3;
index_r[56] = 4;
index_r[57] = 4;
index_r[58] = 4;
index_r[59] = 4;
index_r[60] = 4;
index_r[61] = 4;
index_r[62] = 4;
index_r[63] = 4;
index_r[64] = 4;
index_r[65] = 4;
index_r[66] = 4;
index_r[67] = 4;
index_r[68] = 4;
index_r[69] = 4;
index_r[70] = 4;
index_r[71] = 4;
index_r[72] = 4;
index_r[73] = 4;
index_r[74] = 4;
index_r[75] = 5;
index_r[76] = 5;
index_r[77] = 5;
index_r[78] = 5;
index_r[79] = 5;
index_r[80] = 5;
index_r[81] = 5;
index_r[82] = 5;
index_r[83] = 5;
index_r[84] = 5;
index_r[85] = 5;
index_r[86] = 5;
index_r[87] = 5;
index_r[88] = 5;
index_r[89] = 5;
index_r[90] = 6;
index_r[91] = 6;
index_r[92] = 6;
index_r[93] = 6;
index_r[94] = 6;
index_r[95] = 6;
index_r[96] = 6;
index_r[97] = 6;
index_r[98] = 6;
index_r[99] = 6;
index_r[100] = 6;
index_r[101] = 6;
index_r[102] = 6;
index_r[103] = 6;
index_r[104] = 6;
index_r[105] = 6;
index_r[106] = 6;
index_r[107] = 6;
index_r[108] = 6;
index_r[109] = 6;
index_r[110] = 6;
index_r[111] = 6;
index_r[112] = 6;
index_r[113] = 6;
index_r[114] = 7;
index_r[115] = 7;
index_r[116] = 7;
index_r[117] = 7;
index_r[118] = 7;
index_r[119] = 7;
index_r[120] = 7;
index_r[121] = 7;
index_r[122] = 7;
index_r[123] = 7;
index_r[124] = 7;
index_r[125] = 7;
index_r[126] = 7;
index_r[127] = 7;
index_r[128] = 7;
index_r[129] = 7;
index_r[130] = 7;
index_r[131] = 7;
index_r[132] = 7;
index_r[133] = 7;
index_r[134] = 8;
index_r[135] = 8;
index_r[136] = 8;
index_r[137] = 8;
index_r[138] = 8;
index_r[139] = 8;
index_r[140] = 8;
index_r[141] = 8;
index_r[142] = 8;
index_r[143] = 8;
index_r[144] = 8;
index_r[145] = 8;
index_r[146] = 8;
index_r[147] = 8;
index_r[148] = 8;
index_r[149] = 8;
index_r[150] = 8;
index_r[151] = 8;
index_r[152] = 8;
index_r[153] = 9;
index_r[154] = 9;
index_r[155] = 9;
index_r[156] = 9;
index_r[157] = 9;
index_r[158] = 9;
index_r[159] = 9;
index_r[160] = 9;
index_r[161] = 9;
index_r[162] = 9;
index_r[163] = 9;
index_r[164] = 10;
index_r[165] = 10;
index_r[166] = 10;
index_r[167] = 10;
index_r[168] = 10;
index_r[169] = 10;
index_r[170] = 10;
index_r[171] = 10;
index_r[172] = 10;
index_r[173] = 10;
index_r[174] = 10;
index_r[175] = 10;
index_r[176] = 10;
index_r[177] = 10;
index_r[178] = 10;
index_r[179] = 10;
index_r[180] = 10;
index_r[181] = 11;
index_r[182] = 11;
index_r[183] = 11;
index_r[184] = 11;
index_r[185] = 11;
index_r[186] = 11;
index_r[187] = 11;
index_r[188] = 11;
index_r[189] = 11;
index_r[190] = 11;
index_r[191] = 11;
index_r[192] = 12;
index_r[193] = 12;
index_r[194] = 12;
index_r[195] = 12;
index_r[196] = 12;
index_r[197] = 12;
index_r[198] = 12;
index_r[199] = 12;
index_r[200] = 12;
index_r[201] = 12;
index_r[202] = 12;
index_r[203] = 12;
index_r[204] = 12;
index_r[205] = 12;
index_r[206] = 12;
index_r[207] = 12;
index_r[208] = 12;
index_r[209] = 13;
index_r[210] = 13;
index_r[211] = 13;
index_r[212] = 13;
index_r[213] = 13;
index_r[214] = 13;
index_r[215] = 13;
index_r[216] = 13;
index_r[217] = 13;
index_r[218] = 13;
index_r[219] = 13;
index_r[220] = 13;
index_r[221] = 13;
index_r[222] = 13;
index_r[223] = 13;
index_r[224] = 13;
index_r[225] = 13;
index_r[226] = 13;
index_r[227] = 13;
index_r[228] = 14;
index_r[229] = 14;
index_r[230] = 14;
index_r[231] = 14;
index_r[232] = 14;
index_r[233] = 14;
index_r[234] = 14;
index_r[235] = 14;
index_r[236] = 14;
index_r[237] = 14;
index_r[238] = 14;
index_r[239] = 14;
index_r[240] = 14;
index_r[241] = 14;
index_r[242] = 14;
index_r[243] = 14;
index_r[244] = 14;
index_r[245] = 15;
index_r[246] = 15;
index_r[247] = 15;
index_r[248] = 15;
index_r[249] = 15;
index_r[250] = 15;
index_r[251] = 15;
index_r[252] = 15;
index_r[253] = 15;
index_r[254] = 15;
index_r[255] = 15;
index_r[256] = 15;
index_r[257] = 15;
index_r[258] = 15;
index_r[259] = 15;
index_r[260] = 15;
index_r[261] = 15;
index_r[262] = 15;
index_r[263] = 15;
index_r[264] = 15;
index_r[265] = 15;
index_r[266] = 15;
index_r[267] = 16;
index_r[268] = 16;
index_r[269] = 16;
index_r[270] = 16;
index_r[271] = 16;
index_r[272] = 16;
index_r[273] = 16;
index_r[274] = 16;
index_r[275] = 16;
index_r[276] = 16;
index_r[277] = 16;
index_r[278] = 16;
index_r[279] = 16;
index_r[280] = 16;
index_r[281] = 16;
index_r[282] = 16;
index_r[283] = 16;
index_r[284] = 16;
index_r[285] = 16;
index_r[286] = 16;
index_r[287] = 16;
index_r[288] = 17;
index_r[289] = 17;
index_r[290] = 17;
index_r[291] = 17;
index_r[292] = 17;
index_r[293] = 17;
index_r[294] = 17;
index_r[295] = 17;
index_r[296] = 17;
index_r[297] = 17;
index_r[298] = 17;
index_r[299] = 17;
index_r[300] = 17;
index_r[301] = 17;
index_r[302] = 18;
index_r[303] = 18;
index_r[304] = 18;
index_r[305] = 18;
index_r[306] = 18;
index_r[307] = 18;
index_r[308] = 18;
index_r[309] = 18;
index_r[310] = 18;
index_r[311] = 18;
index_r[312] = 18;
index_r[313] = 19;
index_r[314] = 19;
index_r[315] = 19;
index_r[316] = 19;
index_r[317] = 19;
index_r[318] = 19;
index_r[319] = 19;
index_r[320] = 19;
index_r[321] = 19;
index_r[322] = 19;
index_r[323] = 19;
index_r[324] = 19;
index_r[325] = 19;
index_r[326] = 19;
index_r[327] = 19;
index_r[328] = 19;
index_r[329] = 19;
index_r[330] = 19;
index_r[331] = 19;
index_r[332] = 19;
index_r[333] = 19;
index_r[334] = 19;
index_r[335] = 20;
index_r[336] = 20;
index_r[337] = 20;
index_r[338] = 20;
index_r[339] = 20;
index_r[340] = 20;
index_r[341] = 20;
index_r[342] = 20;
index_r[343] = 20;
index_r[344] = 20;
index_r[345] = 20;
index_r[346] = 21;
index_r[347] = 21;
index_r[348] = 21;
index_r[349] = 21;
index_r[350] = 21;
index_r[351] = 21;
index_r[352] = 21;
index_r[353] = 21;
index_r[354] = 21;
index_r[355] = 21;
index_r[356] = 21;
index_r[357] = 21;
index_r[358] = 21;
index_r[359] = 21;
index_r[360] = 22;
index_r[361] = 22;
index_r[362] = 22;
index_r[363] = 22;
index_r[364] = 22;
index_r[365] = 22;
index_r[366] = 22;
index_r[367] = 22;
index_r[368] = 22;
index_r[369] = 22;
index_r[370] = 22;
index_r[371] = 22;
index_r[372] = 22;
index_r[373] = 22;
index_r[374] = 22;
index_r[375] = 22;
index_r[376] = 22;
index_r[377] = 22;
index_r[378] = 22;
index_r[379] = 23;
index_r[380] = 23;
index_r[381] = 23;
index_r[382] = 23;
index_r[383] = 23;
index_r[384] = 23;
index_r[385] = 23;
index_r[386] = 23;
index_r[387] = 23;
index_r[388] = 23;
index_r[389] = 23;
index_r[390] = 23;
index_r[391] = 23;
index_r[392] = 23;
index_r[393] = 23;
index_r[394] = 23;
index_r[395] = 23;
index_r[396] = 23;
index_r[397] = 23;
index_r[398] = 24;
index_r[399] = 24;
index_r[400] = 24;
index_r[401] = 24;
index_r[402] = 24;
index_r[403] = 24;
index_r[404] = 24;
index_r[405] = 24;
index_r[406] = 24;
index_r[407] = 24;
index_r[408] = 24;
index_r[409] = 24;
index_r[410] = 24;
index_r[411] = 24;
index_r[412] = 24;
index_r[413] = 24;
index_r[414] = 24;
index_r[415] = 25;
index_r[416] = 25;
index_r[417] = 25;
index_r[418] = 25;
index_r[419] = 25;
index_r[420] = 25;
index_r[421] = 25;
index_r[422] = 25;
index_r[423] = 25;
index_r[424] = 25;
index_r[425] = 25;
index_r[426] = 25;
index_r[427] = 25;
index_r[428] = 25;
index_r[429] = 25;
index_r[430] = 25;
index_r[431] = 25;
index_r[432] = 26;
index_r[433] = 26;
index_r[434] = 26;
index_r[435] = 26;
index_r[436] = 26;
index_r[437] = 26;
index_r[438] = 26;
index_r[439] = 27;
index_r[440] = 27;
index_r[441] = 27;
index_r[442] = 27;
index_r[443] = 27;
index_r[444] = 27;
index_r[445] = 27;
index_r[446] = 27;
index_r[447] = 27;
index_r[448] = 27;
index_r[449] = 27;
index_r[450] = 27;
index_r[451] = 27;
index_r[452] = 27;
index_r[453] = 27;
index_r[454] = 28;
index_r[455] = 28;
index_r[456] = 28;
index_r[457] = 28;
index_r[458] = 28;
index_r[459] = 28;
index_r[460] = 28;
index_r[461] = 28;
index_r[462] = 28;
index_r[463] = 28;
index_r[464] = 28;
index_r[465] = 28;
index_r[466] = 28;
index_r[467] = 28;
index_r[468] = 28;
index_r[469] = 28;
index_r[470] = 28;
index_r[471] = 28;
index_r[472] = 28;
index_r[473] = 28;
index_r[474] = 28;
index_r[475] = 28;
index_r[476] = 29;
index_r[477] = 29;
index_r[478] = 29;
index_r[479] = 29;
index_r[480] = 29;
index_r[481] = 29;
index_r[482] = 29;
index_r[483] = 29;
index_r[484] = 29;
index_r[485] = 29;
index_r[486] = 30;
index_r[487] = 30;
index_r[488] = 30;
index_r[489] = 30;
index_r[490] = 30;
index_r[491] = 30;
index_r[492] = 30;
index_r[493] = 30;
index_r[494] = 30;
index_r[495] = 30;
index_r[496] = 30;
index_r[497] = 30;
index_r[498] = 30;
index_r[499] = 30;
index_r[500] = 30;
index_r[501] = 31;
index_r[502] = 31;
index_r[503] = 31;
index_r[504] = 31;
index_r[505] = 31;
index_r[506] = 31;
index_r[507] = 31;
index_r[508] = 31;
index_r[509] = 31;
index_r[510] = 31;
index_r[511] = 31;
index_r[512] = 31;
index_r[513] = 31;
index_r[514] = 31;
index_r[515] = 32;
index_r[516] = 32;
index_r[517] = 32;
index_r[518] = 32;
index_r[519] = 32;
index_r[520] = 32;
index_r[521] = 32;
index_r[522] = 32;
index_r[523] = 32;
index_r[524] = 32;
index_r[525] = 32;
index_r[526] = 32;
index_r[527] = 32;
index_r[528] = 32;
index_r[529] = 32;
index_r[530] = 32;
index_r[531] = 32;
index_r[532] = 32;
index_r[533] = 32;
index_r[534] = 33;
index_r[535] = 33;
index_r[536] = 33;
index_r[537] = 33;
index_r[538] = 33;
index_r[539] = 33;
index_r[540] = 33;
index_r[541] = 33;
index_r[542] = 33;
index_r[543] = 33;
index_r[544] = 33;
index_r[545] = 33;
index_r[546] = 33;
index_r[547] = 33;
index_r[548] = 33;
index_r[549] = 33;
index_r[550] = 33;
index_r[551] = 33;
index_r[552] = 33;
index_r[553] = 33;
index_r[554] = 33;
index_r[555] = 34;
index_r[556] = 34;
index_r[557] = 34;
index_r[558] = 34;
index_r[559] = 34;
index_r[560] = 34;
index_r[561] = 34;
index_r[562] = 34;
index_r[563] = 34;
index_r[564] = 34;
index_r[565] = 34;
index_r[566] = 34;
index_r[567] = 34;
index_r[568] = 34;
index_r[569] = 34;
index_r[570] = 34;
index_r[571] = 34;
index_r[572] = 34;
index_r[573] = 34;
index_r[574] = 34;
index_r[575] = 34;
index_r[576] = 35;
index_r[577] = 35;
index_r[578] = 35;
index_r[579] = 35;
index_r[580] = 35;
index_r[581] = 35;
index_r[582] = 35;
index_r[583] = 35;
index_r[584] = 35;
index_r[585] = 35;
index_r[586] = 35;
index_r[587] = 35;
index_r[588] = 35;
index_r[589] = 35;
index_r[590] = 35;
index_r[591] = 35;
index_r[592] = 35;
index_r[593] = 35;
index_r[594] = 35;
index_r[595] = 36;
index_r[596] = 36;
index_r[597] = 36;
index_r[598] = 36;
index_r[599] = 36;
index_r[600] = 36;
index_r[601] = 36;
index_r[602] = 36;
index_r[603] = 36;
index_r[604] = 36;
index_r[605] = 36;
index_r[606] = 36;
index_r[607] = 36;
index_r[608] = 36;
index_r[609] = 36;
index_r[610] = 36;
index_r[611] = 37;
index_r[612] = 37;
index_r[613] = 37;
index_r[614] = 37;
index_r[615] = 37;
index_r[616] = 37;
index_r[617] = 37;
index_r[618] = 37;
index_r[619] = 37;
index_r[620] = 37;
index_r[621] = 37;
index_r[622] = 37;
index_r[623] = 37;
index_r[624] = 37;
index_r[625] = 37;
index_r[626] = 37;
index_r[627] = 37;
index_r[628] = 37;
index_r[629] = 37;
index_r[630] = 37;
index_r[631] = 37;
index_r[632] = 37;
index_r[633] = 38;
index_r[634] = 38;
index_r[635] = 38;
index_r[636] = 38;
index_r[637] = 38;
index_r[638] = 38;
index_r[639] = 38;
index_r[640] = 39;
index_r[641] = 39;
index_r[642] = 39;
index_r[643] = 39;
index_r[644] = 39;
index_r[645] = 39;
index_r[646] = 39;
index_r[647] = 39;
index_r[648] = 39;
index_r[649] = 39;
index_r[650] = 39;
index_r[651] = 40;
index_r[652] = 40;
index_r[653] = 40;
index_r[654] = 40;
index_r[655] = 40;
index_r[656] = 40;
index_r[657] = 40;
index_r[658] = 40;
index_r[659] = 40;
index_r[660] = 40;
index_r[661] = 40;
index_r[662] = 40;
index_r[663] = 40;
index_r[664] = 40;
index_r[665] = 40;
index_r[666] = 40;
index_r[667] = 41;
index_r[668] = 41;
index_r[669] = 41;
index_r[670] = 41;
index_r[671] = 41;
index_r[672] = 41;
index_r[673] = 41;
index_r[674] = 41;
index_r[675] = 41;
index_r[676] = 41;
index_r[677] = 42;
index_r[678] = 42;
index_r[679] = 42;
index_r[680] = 42;
index_r[681] = 42;
index_r[682] = 42;
index_r[683] = 42;
index_r[684] = 42;
index_r[685] = 42;
index_r[686] = 42;
index_r[687] = 42;
index_r[688] = 42;
index_r[689] = 42;
index_r[690] = 42;
index_r[691] = 42;
index_r[692] = 42;
index_r[693] = 43;
index_r[694] = 43;
index_r[695] = 43;
index_r[696] = 43;
index_r[697] = 43;
index_r[698] = 43;
index_r[699] = 43;
index_r[700] = 43;
index_r[701] = 43;
index_r[702] = 43;
index_r[703] = 43;
index_r[704] = 43;
index_r[705] = 43;
index_r[706] = 43;
index_r[707] = 43;
index_r[708] = 43;
index_r[709] = 43;
index_r[710] = 43;
index_r[711] = 43;
index_r[712] = 44;
index_r[713] = 44;
index_r[714] = 44;
index_r[715] = 44;
index_r[716] = 44;
index_r[717] = 44;
index_r[718] = 44;
index_r[719] = 44;
index_r[720] = 44;
index_r[721] = 44;
index_r[722] = 44;
index_r[723] = 44;
index_r[724] = 44;
index_r[725] = 44;
index_r[726] = 44;
index_r[727] = 44;
index_r[728] = 44;
index_r[729] = 44;
index_r[730] = 44;
index_r[731] = 45;
index_r[732] = 45;
index_r[733] = 45;
index_r[734] = 45;
index_r[735] = 45;
index_r[736] = 45;
index_r[737] = 45;
index_r[738] = 45;
index_r[739] = 45;
index_r[740] = 45;
index_r[741] = 45;
index_r[742] = 45;
index_r[743] = 45;
index_r[744] = 45;
index_r[745] = 45;
index_r[746] = 45;
index_r[747] = 45;
index_r[748] = 45;
index_r[749] = 45;
index_r[750] = 45;
index_r[751] = 45;
index_r[752] = 45;
index_r[753] = 46;
index_r[754] = 46;
index_r[755] = 46;
index_r[756] = 46;
index_r[757] = 46;
index_r[758] = 46;
index_r[759] = 46;
index_r[760] = 46;
index_r[761] = 46;
index_r[762] = 46;
index_r[763] = 46;
index_r[764] = 46;
index_r[765] = 47;
index_r[766] = 47;
index_r[767] = 47;
index_r[768] = 47;
index_r[769] = 47;
index_r[770] = 47;
index_r[771] = 47;
index_r[772] = 47;
index_r[773] = 47;
index_r[774] = 47;
index_r[775] = 47;
index_r[776] = 47;
index_r[777] = 47;
index_r[778] = 47;
index_r[779] = 47;
index_r[780] = 48;
index_r[781] = 48;
index_r[782] = 48;
index_r[783] = 48;
index_r[784] = 48;
index_r[785] = 48;
index_r[786] = 48;
index_r[787] = 48;
index_r[788] = 48;
index_r[789] = 48;
index_r[790] = 48;
index_r[791] = 48;
index_r[792] = 48;
index_r[793] = 48;
index_r[794] = 48;
index_r[795] = 49;
index_r[796] = 49;
index_r[797] = 49;
index_r[798] = 49;
index_r[799] = 49;
index_r[800] = 49;
index_r[801] = 49;
index_r[802] = 50;
index_r[803] = 50;
index_r[804] = 50;
index_r[805] = 50;
index_r[806] = 50;
index_r[807] = 50;
index_r[808] = 50;
index_r[809] = 50;
index_r[810] = 50;
index_r[811] = 50;
index_r[812] = 50;
index_r[813] = 50;
index_r[814] = 50;
index_r[815] = 50;
index_r[816] = 50;
index_r[817] = 50;
index_r[818] = 50;
index_r[819] = 50;
index_r[820] = 50;
index_r[821] = 50;
index_r[822] = 50;
index_r[823] = 50;
index_r[824] = 51;
index_r[825] = 51;
index_r[826] = 51;
index_r[827] = 51;
index_r[828] = 51;
index_r[829] = 51;
index_r[830] = 51;
index_r[831] = 51;
index_r[832] = 51;
index_r[833] = 51;
index_r[834] = 51;
index_r[835] = 51;
index_r[836] = 51;
index_r[837] = 51;
index_r[838] = 51;
index_r[839] = 52;
index_r[840] = 52;
index_r[841] = 52;
index_r[842] = 52;
index_r[843] = 52;
index_r[844] = 52;
index_r[845] = 52;
index_r[846] = 52;
index_r[847] = 52;
index_r[848] = 52;
index_r[849] = 52;
index_r[850] = 52;
index_r[851] = 52;
index_r[852] = 52;
index_r[853] = 52;
index_r[854] = 52;
index_r[855] = 52;
index_r[856] = 53;
index_r[857] = 53;
index_r[858] = 53;
index_r[859] = 53;
index_r[860] = 53;
index_r[861] = 53;
index_r[862] = 53;
index_r[863] = 53;
index_r[864] = 53;
index_r[865] = 53;
index_r[866] = 53;
index_r[867] = 53;
index_r[868] = 53;
index_r[869] = 53;
index_r[870] = 53;
index_r[871] = 53;
index_r[872] = 53;
index_r[873] = 53;
index_r[874] = 53;
index_r[875] = 54;
index_r[876] = 54;
index_r[877] = 54;
index_r[878] = 54;
index_r[879] = 54;
index_r[880] = 54;
index_r[881] = 54;
index_r[882] = 54;
index_r[883] = 54;
index_r[884] = 54;
index_r[885] = 54;
index_r[886] = 54;
index_r[887] = 54;
index_r[888] = 54;
index_r[889] = 54;
index_r[890] = 54;
index_r[891] = 54;
index_r[892] = 54;
index_r[893] = 54;
index_r[894] = 55;
index_r[895] = 55;
index_r[896] = 55;
index_r[897] = 55;
index_r[898] = 55;
index_r[899] = 55;
index_r[900] = 55;
index_r[901] = 55;
index_r[902] = 55;
index_r[903] = 55;
index_r[904] = 55;
index_r[905] = 56;
index_r[906] = 56;
index_r[907] = 56;
index_r[908] = 56;
index_r[909] = 56;
index_r[910] = 56;
index_r[911] = 56;
index_r[912] = 56;
index_r[913] = 56;
index_r[914] = 56;
index_r[915] = 56;
index_r[916] = 56;
index_r[917] = 56;
index_r[918] = 56;
index_r[919] = 56;
index_r[920] = 56;
index_r[921] = 56;
index_r[922] = 56;
index_r[923] = 56;
index_r[924] = 56;
index_r[925] = 56;
index_r[926] = 57;
index_r[927] = 57;
index_r[928] = 57;
index_r[929] = 57;
index_r[930] = 57;
index_r[931] = 57;
index_r[932] = 57;
index_r[933] = 57;
index_r[934] = 57;
index_r[935] = 57;
index_r[936] = 57;
index_r[937] = 57;
index_r[938] = 57;
index_r[939] = 57;
index_r[940] = 57;
index_r[941] = 57;
index_r[942] = 57;
index_r[943] = 57;
index_r[944] = 57;
index_r[945] = 58;
index_r[946] = 58;
index_r[947] = 58;
index_r[948] = 58;
index_r[949] = 58;
index_r[950] = 58;
index_r[951] = 58;
index_r[952] = 58;
index_r[953] = 58;
index_r[954] = 58;
index_r[955] = 58;
index_r[956] = 58;
index_r[957] = 58;
index_r[958] = 58;
index_r[959] = 59;
index_r[960] = 59;
index_r[961] = 59;
index_r[962] = 59;
index_r[963] = 59;
index_r[964] = 59;
index_r[965] = 59;
index_r[966] = 59;
index_r[967] = 59;
index_r[968] = 59;
index_r[969] = 59;
index_r[970] = 59;
index_r[971] = 59;
index_r[972] = 59;
index_r[973] = 59;
index_r[974] = 59;
index_r[975] = 59;
index_r[976] = 60;
index_r[977] = 60;
index_r[978] = 60;
index_r[979] = 60;
index_r[980] = 60;
index_r[981] = 60;
index_r[982] = 60;
index_r[983] = 61;
index_r[984] = 61;
index_r[985] = 61;
index_r[986] = 61;
index_r[987] = 61;
index_r[988] = 61;
index_r[989] = 61;
index_r[990] = 61;
index_r[991] = 61;
index_r[992] = 61;
index_r[993] = 61;
index_r[994] = 61;
index_r[995] = 62;
index_r[996] = 62;
index_r[997] = 62;
index_r[998] = 62;
index_r[999] = 62;
index_r[1000] = 62;
index_r[1001] = 62;
index_r[1002] = 62;
index_r[1003] = 62;
index_r[1004] = 62;
index_r[1005] = 62;
index_r[1006] = 62;
index_r[1007] = 62;
index_r[1008] = 62;
index_r[1009] = 62;
index_r[1010] = 62;
index_r[1011] = 62;
index_r[1012] = 62;
index_r[1013] = 62;
index_r[1014] = 62;
index_r[1015] = 63;
index_r[1016] = 63;
index_r[1017] = 63;
index_r[1018] = 63;
index_r[1019] = 63;
index_r[1020] = 63;
index_r[1021] = 63;
index_r[1022] = 63;
index_r[1023] = 63;
index_r[1024] = 63;
index_r[1025] = 63;
index_r[1026] = 63;
index_r[1027] = 63;
index_r[1028] = 63;
index_r[1029] = 63;
index_r[1030] = 63;
index_r[1031] = 63;
index_r[1032] = 63;
index_r[1033] = 63;
index_r[1034] = 64;
index_r[1035] = 64;
index_r[1036] = 64;
index_r[1037] = 64;
index_r[1038] = 64;
index_r[1039] = 64;
index_r[1040] = 64;
index_r[1041] = 65;
index_r[1042] = 65;
index_r[1043] = 65;
index_r[1044] = 65;
index_r[1045] = 65;
index_r[1046] = 65;
index_r[1047] = 65;
index_r[1048] = 65;
index_r[1049] = 65;
index_r[1050] = 65;
index_r[1051] = 65;
index_r[1052] = 65;
index_r[1053] = 65;
index_r[1054] = 65;
index_r[1055] = 65;
index_r[1056] = 66;
index_r[1057] = 66;
index_r[1058] = 66;
index_r[1059] = 66;
index_r[1060] = 66;
index_r[1061] = 66;
index_r[1062] = 66;
index_r[1063] = 66;
index_r[1064] = 66;
index_r[1065] = 66;
index_r[1066] = 66;
index_r[1067] = 66;
index_r[1068] = 66;
index_r[1069] = 66;
index_r[1070] = 66;
index_r[1071] = 66;
index_r[1072] = 66;
index_r[1073] = 66;
index_r[1074] = 66;
index_r[1075] = 67;
index_r[1076] = 67;
index_r[1077] = 67;
index_r[1078] = 67;
index_r[1079] = 67;
index_r[1080] = 67;
index_r[1081] = 67;
index_r[1082] = 68;
index_r[1083] = 68;
index_r[1084] = 68;
index_r[1085] = 68;
index_r[1086] = 68;
index_r[1087] = 68;
index_r[1088] = 68;
index_r[1089] = 68;
index_r[1090] = 68;
index_r[1091] = 68;
index_r[1092] = 68;
index_r[1093] = 68;
index_r[1094] = 68;
index_r[1095] = 68;
index_r[1096] = 68;
index_r[1097] = 68;
index_r[1098] = 68;
index_r[1099] = 68;
index_r[1100] = 68;
index_r[1101] = 69;
index_r[1102] = 69;
index_r[1103] = 69;
index_r[1104] = 69;
index_r[1105] = 69;
index_r[1106] = 69;
index_r[1107] = 69;
index_r[1108] = 69;
index_r[1109] = 69;
index_r[1110] = 69;
index_r[1111] = 69;
index_r[1112] = 69;
index_r[1113] = 69;
index_r[1114] = 69;
index_r[1115] = 69;
index_r[1116] = 69;
index_r[1117] = 69;
index_r[1118] = 69;
index_r[1119] = 69;
index_r[1120] = 69;
index_r[1121] = 70;
index_r[1122] = 70;
index_r[1123] = 70;
index_r[1124] = 70;
index_r[1125] = 70;
index_r[1126] = 70;
index_r[1127] = 70;
index_r[1128] = 70;
index_r[1129] = 70;
index_r[1130] = 70;
index_r[1131] = 70;
index_r[1132] = 70;
index_r[1133] = 70;
index_r[1134] = 70;
index_r[1135] = 70;
index_r[1136] = 71;
index_r[1137] = 71;
index_r[1138] = 71;
index_r[1139] = 71;
index_r[1140] = 71;
index_r[1141] = 71;
index_r[1142] = 71;
index_r[1143] = 71;
index_r[1144] = 71;
index_r[1145] = 71;
index_r[1146] = 71;
index_r[1147] = 71;
index_r[1148] = 71;
index_r[1149] = 71;
index_r[1150] = 71;
index_r[1151] = 72;
index_r[1152] = 72;
index_r[1153] = 72;
index_r[1154] = 72;
index_r[1155] = 72;
index_r[1156] = 72;
index_r[1157] = 72;
index_r[1158] = 73;
index_r[1159] = 73;
index_r[1160] = 73;
index_r[1161] = 73;
index_r[1162] = 73;
index_r[1163] = 73;
index_r[1164] = 73;
index_r[1165] = 73;
index_r[1166] = 73;
index_r[1167] = 73;
index_r[1168] = 73;
index_r[1169] = 73;
index_r[1170] = 73;
index_r[1171] = 73;
index_r[1172] = 73;
index_r[1173] = 73;
index_r[1174] = 73;
index_r[1175] = 74;
index_r[1176] = 74;
index_r[1177] = 74;
index_r[1178] = 74;
index_r[1179] = 74;
index_r[1180] = 74;
index_r[1181] = 74;
index_r[1182] = 74;
index_r[1183] = 74;
index_r[1184] = 74;
index_r[1185] = 74;
index_r[1186] = 74;
index_r[1187] = 74;
index_r[1188] = 74;
index_r[1189] = 74;
index_r[1190] = 75;
index_r[1191] = 75;
index_r[1192] = 75;
index_r[1193] = 75;
index_r[1194] = 75;
index_r[1195] = 75;
index_r[1196] = 75;
index_r[1197] = 75;
index_r[1198] = 75;
index_r[1199] = 75;
index_r[1200] = 75;
index_r[1201] = 75;
index_r[1202] = 75;
index_r[1203] = 75;
index_r[1204] = 75;
index_r[1205] = 75;
index_r[1206] = 75;
index_r[1207] = 75;
index_r[1208] = 75;
index_r[1209] = 75;
index_r[1210] = 75;
index_r[1211] = 75;
index_r[1212] = 75;
index_r[1213] = 75;
index_r[1214] = 76;
index_r[1215] = 76;
index_r[1216] = 76;
index_r[1217] = 76;
index_r[1218] = 76;
index_r[1219] = 76;
index_r[1220] = 76;
index_r[1221] = 76;
index_r[1222] = 76;
index_r[1223] = 76;
index_r[1224] = 76;
index_r[1225] = 77;
index_r[1226] = 77;
index_r[1227] = 77;
index_r[1228] = 77;
index_r[1229] = 77;
index_r[1230] = 77;
index_r[1231] = 77;
index_r[1232] = 77;
index_r[1233] = 77;
index_r[1234] = 77;
index_r[1235] = 78;
index_r[1236] = 78;
index_r[1237] = 78;
index_r[1238] = 78;
index_r[1239] = 78;
index_r[1240] = 78;
index_r[1241] = 78;
index_r[1242] = 78;
index_r[1243] = 78;
index_r[1244] = 78;
index_r[1245] = 78;
index_r[1246] = 78;
index_r[1247] = 78;
index_r[1248] = 78;
index_r[1249] = 78;
index_r[1250] = 78;
index_r[1251] = 78;
index_r[1252] = 78;
index_r[1253] = 78;
index_r[1254] = 78;
index_r[1255] = 78;
index_r[1256] = 78;
index_r[1257] = 78;
index_r[1258] = 78;
index_r[1259] = 79;
index_r[1260] = 79;
index_r[1261] = 79;
index_r[1262] = 79;
index_r[1263] = 79;
index_r[1264] = 79;
index_r[1265] = 79;
index_r[1266] = 79;
index_r[1267] = 79;
index_r[1268] = 79;
index_r[1269] = 79;
index_r[1270] = 79;
index_r[1271] = 79;
index_r[1272] = 79;
index_r[1273] = 79;
index_r[1274] = 79;
index_r[1275] = 80;
index_r[1276] = 80;
index_r[1277] = 80;
index_r[1278] = 80;
index_r[1279] = 80;
index_r[1280] = 80;
index_r[1281] = 80;
index_r[1282] = 80;
index_r[1283] = 80;
index_r[1284] = 80;
index_r[1285] = 80;
index_r[1286] = 80;
index_r[1287] = 80;
index_r[1288] = 80;
index_r[1289] = 80;
index_r[1290] = 80;
index_r[1291] = 80;
index_r[1292] = 80;
index_r[1293] = 80;
index_r[1294] = 80;
index_r[1295] = 80;
index_r[1296] = 80;
index_r[1297] = 80;
index_r[1298] = 80;
index_r[1299] = 81;
index_r[1300] = 81;
index_r[1301] = 81;
index_r[1302] = 81;
index_r[1303] = 81;
index_r[1304] = 81;
index_r[1305] = 81;
index_r[1306] = 81;
index_r[1307] = 81;
index_r[1308] = 81;
index_r[1309] = 82;
index_r[1310] = 82;
index_r[1311] = 82;
index_r[1312] = 82;
index_r[1313] = 82;
index_r[1314] = 82;
index_r[1315] = 82;
index_r[1316] = 82;
index_r[1317] = 82;
index_r[1318] = 82;
index_r[1319] = 82;
index_r[1320] = 82;
index_r[1321] = 82;
index_r[1322] = 82;
index_r[1323] = 82;
index_r[1324] = 82;
index_r[1325] = 82;
index_r[1326] = 82;
index_r[1327] = 82;
index_r[1328] = 82;
index_r[1329] = 82;
index_r[1330] = 82;
index_r[1331] = 83;
index_r[1332] = 83;
index_r[1333] = 83;
index_r[1334] = 83;
index_r[1335] = 83;
index_r[1336] = 83;
index_r[1337] = 83;
index_r[1338] = 83;
index_r[1339] = 83;
index_r[1340] = 83;
index_r[1341] = 83;
index_r[1342] = 83;
index_r[1343] = 83;
index_r[1344] = 83;
index_r[1345] = 84;
index_r[1346] = 84;
index_r[1347] = 84;
index_r[1348] = 84;
index_r[1349] = 84;
index_r[1350] = 84;
index_r[1351] = 84;
index_r[1352] = 84;
index_r[1353] = 84;
index_r[1354] = 84;
index_r[1355] = 85;
index_r[1356] = 85;
index_r[1357] = 85;
index_r[1358] = 85;
index_r[1359] = 85;
index_r[1360] = 85;
index_r[1361] = 85;
index_r[1362] = 85;
index_r[1363] = 85;
index_r[1364] = 85;
index_r[1365] = 85;
index_r[1366] = 86;
index_r[1367] = 86;
index_r[1368] = 86;
index_r[1369] = 86;
index_r[1370] = 86;
index_r[1371] = 86;
index_r[1372] = 86;
index_r[1373] = 86;
index_r[1374] = 86;
index_r[1375] = 86;
index_r[1376] = 86;
index_r[1377] = 86;
index_r[1378] = 86;
index_r[1379] = 86;
index_r[1380] = 86;
index_r[1381] = 87;
index_r[1382] = 87;
index_r[1383] = 87;
index_r[1384] = 87;
index_r[1385] = 87;
index_r[1386] = 87;
index_r[1387] = 87;
index_r[1388] = 87;
index_r[1389] = 87;
index_r[1390] = 87;
index_r[1391] = 87;
index_r[1392] = 87;
index_r[1393] = 87;
index_r[1394] = 87;
index_r[1395] = 87;
index_r[1396] = 87;
index_r[1397] = 88;
index_r[1398] = 88;
index_r[1399] = 88;
index_r[1400] = 88;
index_r[1401] = 88;
index_r[1402] = 88;
index_r[1403] = 88;
index_r[1404] = 88;
index_r[1405] = 88;
index_r[1406] = 88;
index_r[1407] = 89;
index_r[1408] = 89;
index_r[1409] = 89;
index_r[1410] = 89;
index_r[1411] = 89;
index_r[1412] = 89;
index_r[1413] = 89;
index_r[1414] = 89;
index_r[1415] = 89;
index_r[1416] = 89;
index_r[1417] = 89;
index_r[1418] = 89;
index_r[1419] = 89;
index_r[1420] = 89;
index_r[1421] = 89;
index_r[1422] = 90;
index_r[1423] = 90;
index_r[1424] = 90;
index_r[1425] = 90;
index_r[1426] = 90;
index_r[1427] = 90;
index_r[1428] = 90;
index_r[1429] = 90;
index_r[1430] = 90;
index_r[1431] = 90;
index_r[1432] = 90;
index_r[1433] = 90;
index_r[1434] = 90;
index_r[1435] = 90;
index_r[1436] = 90;
index_r[1437] = 90;
index_r[1438] = 90;
index_r[1439] = 90;
index_r[1440] = 90;
index_r[1441] = 91;
index_r[1442] = 91;
index_r[1443] = 91;
index_r[1444] = 91;
index_r[1445] = 91;
index_r[1446] = 91;
index_r[1447] = 91;
index_r[1448] = 91;
index_r[1449] = 91;
index_r[1450] = 91;
index_r[1451] = 91;
index_r[1452] = 92;
index_r[1453] = 92;
index_r[1454] = 92;
index_r[1455] = 92;
index_r[1456] = 92;
index_r[1457] = 92;
index_r[1458] = 92;
index_r[1459] = 92;
index_r[1460] = 92;
index_r[1461] = 92;
index_r[1462] = 92;
index_r[1463] = 92;
index_r[1464] = 92;
index_r[1465] = 92;
index_r[1466] = 92;
index_r[1467] = 92;
index_r[1468] = 92;
index_r[1469] = 92;
index_r[1470] = 92;
index_r[1471] = 92;
index_r[1472] = 92;
index_r[1473] = 93;
index_r[1474] = 93;
index_r[1475] = 93;
index_r[1476] = 93;
index_r[1477] = 93;
index_r[1478] = 93;
index_r[1479] = 93;
index_r[1480] = 93;
index_r[1481] = 93;
index_r[1482] = 93;
index_r[1483] = 93;
index_r[1484] = 93;
index_r[1485] = 93;
index_r[1486] = 93;
index_r[1487] = 93;
index_r[1488] = 93;
index_r[1489] = 93;
index_r[1490] = 93;
index_r[1491] = 93;
index_r[1492] = 93;
index_r[1493] = 93;
index_r[1494] = 93;
index_r[1495] = 94;
index_r[1496] = 94;
index_r[1497] = 94;
index_r[1498] = 94;
index_r[1499] = 94;
index_r[1500] = 94;
index_r[1501] = 94;
index_r[1502] = 94;
index_r[1503] = 94;
index_r[1504] = 94;
index_r[1505] = 94;
index_r[1506] = 94;
index_r[1507] = 94;
index_r[1508] = 94;
index_r[1509] = 94;
index_r[1510] = 94;
index_r[1511] = 94;
index_r[1512] = 94;
index_r[1513] = 94;
index_r[1514] = 94;
index_r[1515] = 94;
index_r[1516] = 94;
index_r[1517] = 95;
index_r[1518] = 95;
index_r[1519] = 95;
index_r[1520] = 95;
index_r[1521] = 95;
index_r[1522] = 95;
index_r[1523] = 95;
index_r[1524] = 95;
index_r[1525] = 95;
index_r[1526] = 95;
index_r[1527] = 95;
index_r[1528] = 95;
index_r[1529] = 95;
index_r[1530] = 95;
index_r[1531] = 95;
index_r[1532] = 95;
index_r[1533] = 95;
index_r[1534] = 95;
index_r[1535] = 95;
index_r[1536] = 95;
index_r[1537] = 96;
index_r[1538] = 96;
index_r[1539] = 96;
index_r[1540] = 96;
index_r[1541] = 96;
index_r[1542] = 96;
index_r[1543] = 96;
index_r[1544] = 96;
index_r[1545] = 96;
index_r[1546] = 96;
index_r[1547] = 96;
index_r[1548] = 96;
index_r[1549] = 96;
index_r[1550] = 96;
index_r[1551] = 96;
index_r[1552] = 96;
index_r[1553] = 96;
index_r[1554] = 96;
index_r[1555] = 96;
index_r[1556] = 96;
index_r[1557] = 96;
index_r[1558] = 96;
index_r[1559] = 96;
index_r[1560] = 96;
index_r[1561] = 97;
index_r[1562] = 97;
index_r[1563] = 97;
index_r[1564] = 97;
index_r[1565] = 97;
index_r[1566] = 97;
index_r[1567] = 97;
index_r[1568] = 97;
index_r[1569] = 97;
index_r[1570] = 97;
index_r[1571] = 97;
index_r[1572] = 97;
index_r[1573] = 97;
index_r[1574] = 97;
index_r[1575] = 97;
index_r[1576] = 97;
index_r[1577] = 97;
index_r[1578] = 98;
index_r[1579] = 98;
index_r[1580] = 98;
index_r[1581] = 98;
index_r[1582] = 98;
index_r[1583] = 98;
index_r[1584] = 98;
index_r[1585] = 98;
index_r[1586] = 98;
index_r[1587] = 98;
index_r[1588] = 98;
index_r[1589] = 98;
index_r[1590] = 98;
index_r[1591] = 98;
index_r[1592] = 98;
index_r[1593] = 98;
index_r[1594] = 98;
index_r[1595] = 98;
index_r[1596] = 98;
index_r[1597] = 99;
index_r[1598] = 99;
index_r[1599] = 99;
index_r[1600] = 99;
index_r[1601] = 99;
index_r[1602] = 99;
index_r[1603] = 99;
index_r[1604] = 99;
index_r[1605] = 99;
index_r[1606] = 99;
index_r[1607] = 99;
index_r[1608] = 99;
index_r[1609] = 99;
index_r[1610] = 99;
index_r[1611] = 99;
index_r[1612] = 99;
index_r[1613] = 99;
index_r[1614] = 99;
index_r[1615] = 99;
index_r[1616] = 100;
index_r[1617] = 100;
index_r[1618] = 100;
index_r[1619] = 100;
index_r[1620] = 100;
index_r[1621] = 100;
index_r[1622] = 100;
index_r[1623] = 100;
index_r[1624] = 100;
index_r[1625] = 100;
index_r[1626] = 100;
index_r[1627] = 100;
index_r[1628] = 100;
index_r[1629] = 100;
index_r[1630] = 100;
index_r[1631] = 100;
index_r[1632] = 100;
index_r[1633] = 101;
index_r[1634] = 101;
index_r[1635] = 101;
index_r[1636] = 101;
index_r[1637] = 101;
index_r[1638] = 101;
index_r[1639] = 101;
index_r[1640] = 101;
index_r[1641] = 101;
index_r[1642] = 101;
index_r[1643] = 101;
index_r[1644] = 101;
index_r[1645] = 101;
index_r[1646] = 101;
index_r[1647] = 101;
index_r[1648] = 101;
index_r[1649] = 102;
index_r[1650] = 102;
index_r[1651] = 102;
index_r[1652] = 102;
index_r[1653] = 102;
index_r[1654] = 102;
index_r[1655] = 102;
index_r[1656] = 102;
index_r[1657] = 102;
index_r[1658] = 102;
index_r[1659] = 102;
index_r[1660] = 102;
index_r[1661] = 102;
index_r[1662] = 102;
index_r[1663] = 103;
index_r[1664] = 103;
index_r[1665] = 103;
index_r[1666] = 103;
index_r[1667] = 103;
index_r[1668] = 103;
index_r[1669] = 103;
index_r[1670] = 103;
index_r[1671] = 103;
index_r[1672] = 103;
index_r[1673] = 103;
index_r[1674] = 103;
index_r[1675] = 103;
index_r[1676] = 103;
index_r[1677] = 104;
index_r[1678] = 104;
index_r[1679] = 104;
index_r[1680] = 104;
index_r[1681] = 104;
index_r[1682] = 104;
index_r[1683] = 104;
index_r[1684] = 104;
index_r[1685] = 104;
index_r[1686] = 104;
index_r[1687] = 104;
index_r[1688] = 104;
index_r[1689] = 105;
index_r[1690] = 105;
index_r[1691] = 105;
index_r[1692] = 105;
index_r[1693] = 105;
index_r[1694] = 105;
index_r[1695] = 105;
index_r[1696] = 105;
index_r[1697] = 105;
index_r[1698] = 105;
index_r[1699] = 105;
index_r[1700] = 105;
index_r[1701] = 105;
index_r[1702] = 105;
index_r[1703] = 105;
index_r[1704] = 105;
index_r[1705] = 105;
index_r[1706] = 105;
index_r[1707] = 105;
index_r[1708] = 106;
index_r[1709] = 106;
index_r[1710] = 106;
index_r[1711] = 106;
index_r[1712] = 106;
index_r[1713] = 106;
index_r[1714] = 106;
index_r[1715] = 106;
index_r[1716] = 106;
index_r[1717] = 106;
index_r[1718] = 106;
index_r[1719] = 106;
index_r[1720] = 106;
index_r[1721] = 106;
index_r[1722] = 106;
index_r[1723] = 106;
index_r[1724] = 106;
index_r[1725] = 106;
index_r[1726] = 106;
index_r[1727] = 107;
index_r[1728] = 107;
index_r[1729] = 107;
index_r[1730] = 107;
index_r[1731] = 107;
index_r[1732] = 107;
index_r[1733] = 107;
index_r[1734] = 107;
index_r[1735] = 107;
index_r[1736] = 107;
index_r[1737] = 107;
index_r[1738] = 107;
index_r[1739] = 107;
index_r[1740] = 107;
index_r[1741] = 107;
index_r[1742] = 107;
index_r[1743] = 107;
index_r[1744] = 108;
index_r[1745] = 108;
index_r[1746] = 108;
index_r[1747] = 108;
index_r[1748] = 108;
index_r[1749] = 108;
index_r[1750] = 108;
index_r[1751] = 108;
index_r[1752] = 108;
index_r[1753] = 108;
index_r[1754] = 108;
index_r[1755] = 108;
index_r[1756] = 108;
index_r[1757] = 108;
index_r[1758] = 108;
index_r[1759] = 108;
index_r[1760] = 108;
index_r[1761] = 108;
index_r[1762] = 108;
index_r[1763] = 108;
index_r[1764] = 108;
index_r[1765] = 108;
index_r[1766] = 108;
index_r[1767] = 108;
index_r[1768] = 109;
index_r[1769] = 109;
index_r[1770] = 109;
index_r[1771] = 109;
index_r[1772] = 109;
index_r[1773] = 109;
index_r[1774] = 109;
index_r[1775] = 109;
index_r[1776] = 109;
index_r[1777] = 109;
index_r[1778] = 109;
index_r[1779] = 109;
index_r[1780] = 109;
index_r[1781] = 109;
index_r[1782] = 109;
index_r[1783] = 109;
index_r[1784] = 109;
index_r[1785] = 109;
index_r[1786] = 109;
index_r[1787] = 110;
index_r[1788] = 110;
index_r[1789] = 110;
index_r[1790] = 110;
index_r[1791] = 110;
index_r[1792] = 110;
index_r[1793] = 110;
index_r[1794] = 110;
index_r[1795] = 110;
index_r[1796] = 110;
index_r[1797] = 110;
index_r[1798] = 111;
index_r[1799] = 111;
index_r[1800] = 111;
index_r[1801] = 111;
index_r[1802] = 111;
index_r[1803] = 111;
index_r[1804] = 111;
index_r[1805] = 111;
index_r[1806] = 111;
index_r[1807] = 111;
index_r[1808] = 112;
index_r[1809] = 112;
index_r[1810] = 112;
index_r[1811] = 112;
index_r[1812] = 112;
index_r[1813] = 112;
index_r[1814] = 112;
index_r[1815] = 112;
index_r[1816] = 112;
index_r[1817] = 112;
index_r[1818] = 112;
index_r[1819] = 112;
index_r[1820] = 112;
index_r[1821] = 112;
index_r[1822] = 112;
index_r[1823] = 112;
index_r[1824] = 112;
index_r[1825] = 113;
index_r[1826] = 113;
index_r[1827] = 113;
index_r[1828] = 113;
index_r[1829] = 113;
index_r[1830] = 113;
index_r[1831] = 113;
index_r[1832] = 113;
index_r[1833] = 113;
index_r[1834] = 113;
index_r[1835] = 113;
index_r[1836] = 113;
index_r[1837] = 113;
index_r[1838] = 113;
index_r[1839] = 113;
index_r[1840] = 113;
index_r[1841] = 113;
index_r[1842] = 114;
index_r[1843] = 114;
index_r[1844] = 114;
index_r[1845] = 114;
index_r[1846] = 114;
index_r[1847] = 114;
index_r[1848] = 114;
index_r[1849] = 114;
index_r[1850] = 114;
index_r[1851] = 114;
index_r[1852] = 115;
index_r[1853] = 115;
index_r[1854] = 115;
index_r[1855] = 115;
index_r[1856] = 115;
index_r[1857] = 115;
index_r[1858] = 115;
index_r[1859] = 115;
index_r[1860] = 115;
index_r[1861] = 115;
index_r[1862] = 115;
index_r[1863] = 115;
index_r[1864] = 115;
index_r[1865] = 115;
index_r[1866] = 115;
index_r[1867] = 115;
index_r[1868] = 115;
index_r[1869] = 115;
index_r[1870] = 115;
index_r[1871] = 115;
index_r[1872] = 115;
index_r[1873] = 115;
index_r[1874] = 115;
index_r[1875] = 115;
index_r[1876] = 116;
index_r[1877] = 116;
index_r[1878] = 116;
index_r[1879] = 116;
index_r[1880] = 116;
index_r[1881] = 116;
index_r[1882] = 116;
index_r[1883] = 116;
index_r[1884] = 116;
index_r[1885] = 116;
index_r[1886] = 116;
index_r[1887] = 116;
index_r[1888] = 116;
index_r[1889] = 116;
index_r[1890] = 116;
index_r[1891] = 116;
index_r[1892] = 116;
index_r[1893] = 116;
index_r[1894] = 116;
index_r[1895] = 116;
index_r[1896] = 116;
index_r[1897] = 116;
index_r[1898] = 116;
index_r[1899] = 116;
index_r[1900] = 117;
index_r[1901] = 117;
index_r[1902] = 117;
index_r[1903] = 117;
index_r[1904] = 117;
index_r[1905] = 117;
index_r[1906] = 117;
index_r[1907] = 117;
index_r[1908] = 117;
index_r[1909] = 117;
index_r[1910] = 117;
index_r[1911] = 117;
index_r[1912] = 117;
index_r[1913] = 117;
index_r[1914] = 117;
index_r[1915] = 117;
index_r[1916] = 117;
index_r[1917] = 117;
index_r[1918] = 117;
index_r[1919] = 118;
index_r[1920] = 118;
index_r[1921] = 118;
index_r[1922] = 118;
index_r[1923] = 118;
index_r[1924] = 118;
index_r[1925] = 118;
index_r[1926] = 118;
index_r[1927] = 118;
index_r[1928] = 118;
index_r[1929] = 118;
index_r[1930] = 118;
index_r[1931] = 118;
index_r[1932] = 118;
index_r[1933] = 118;
index_r[1934] = 118;
index_r[1935] = 118;
index_r[1936] = 119;
index_r[1937] = 119;
index_r[1938] = 119;
index_r[1939] = 119;
index_r[1940] = 119;
index_r[1941] = 119;
index_r[1942] = 119;
index_r[1943] = 119;
index_r[1944] = 119;
index_r[1945] = 119;
index_r[1946] = 119;
index_r[1947] = 119;
index_r[1948] = 119;
index_r[1949] = 119;
index_r[1950] = 119;
index_r[1951] = 119;
index_r[1952] = 120;
index_r[1953] = 120;
index_r[1954] = 120;
index_r[1955] = 120;
index_r[1956] = 120;
index_r[1957] = 120;
index_r[1958] = 120;
index_r[1959] = 120;
index_r[1960] = 120;
index_r[1961] = 120;
index_r[1962] = 120;
index_r[1963] = 120;
index_r[1964] = 120;
index_r[1965] = 120;
index_r[1966] = 121;
index_r[1967] = 121;
index_r[1968] = 121;
index_r[1969] = 121;
index_r[1970] = 121;
index_r[1971] = 121;
index_r[1972] = 121;
index_r[1973] = 121;
index_r[1974] = 121;
index_r[1975] = 121;
index_r[1976] = 121;
index_r[1977] = 122;
index_r[1978] = 122;
index_r[1979] = 122;
index_r[1980] = 122;
index_r[1981] = 122;
index_r[1982] = 122;
index_r[1983] = 122;
index_r[1984] = 122;
index_r[1985] = 122;
index_r[1986] = 122;
index_r[1987] = 122;
index_r[1988] = 122;
index_r[1989] = 122;
index_r[1990] = 122;
index_r[1991] = 122;
index_r[1992] = 123;
index_r[1993] = 123;
index_r[1994] = 123;
index_r[1995] = 123;
index_r[1996] = 123;
index_r[1997] = 123;
index_r[1998] = 123;
index_r[1999] = 123;
index_r[2000] = 123;
index_r[2001] = 123;
index_r[2002] = 123;
index_r[2003] = 123;
index_r[2004] = 123;
index_r[2005] = 123;
index_r[2006] = 123;
index_r[2007] = 123;
index_r[2008] = 123;
index_r[2009] = 123;
index_r[2010] = 123;
index_r[2011] = 123;
index_r[2012] = 123;
index_r[2013] = 123;
index_r[2014] = 124;
index_r[2015] = 124;
index_r[2016] = 124;
index_r[2017] = 124;
index_r[2018] = 124;
index_r[2019] = 124;
index_r[2020] = 124;
index_r[2021] = 124;
index_r[2022] = 124;
index_r[2023] = 124;
index_r[2024] = 124;
index_r[2025] = 124;
index_r[2026] = 124;
index_r[2027] = 124;
index_r[2028] = 124;
index_r[2029] = 124;
index_r[2030] = 125;
index_r[2031] = 125;
index_r[2032] = 125;
index_r[2033] = 125;
index_r[2034] = 125;
index_r[2035] = 125;
index_r[2036] = 125;
index_r[2037] = 126;
index_r[2038] = 126;
index_r[2039] = 126;
index_r[2040] = 126;
index_r[2041] = 126;
index_r[2042] = 126;
index_r[2043] = 126;
index_r[2044] = 126;
index_r[2045] = 126;
index_r[2046] = 126;
index_r[2047] = 126;
index_r[2048] = 126;
index_r[2049] = 126;
index_r[2050] = 126;
index_r[2051] = 127;
index_r[2052] = 127;
index_r[2053] = 127;
index_r[2054] = 127;
index_r[2055] = 127;
index_r[2056] = 127;
index_r[2057] = 127;
index_r[2058] = 127;
index_r[2059] = 127;
index_r[2060] = 127;
index_r[2061] = 127;
index_r[2062] = 127;
index_r[2063] = 127;
index_r[2064] = 127;
index_r[2065] = 127;
index_r[2066] = 127;
index_r[2067] = 127;
index_r[2068] = 127;
index_r[2069] = 127;
index_r[2070] = 128;
index_r[2071] = 128;
index_r[2072] = 128;
index_r[2073] = 128;
index_r[2074] = 128;
index_r[2075] = 128;
index_r[2076] = 128;
index_r[2077] = 128;
index_r[2078] = 128;
index_r[2079] = 128;
index_r[2080] = 129;
index_r[2081] = 129;
index_r[2082] = 129;
index_r[2083] = 129;
index_r[2084] = 129;
index_r[2085] = 129;
index_r[2086] = 129;
index_r[2087] = 129;
index_r[2088] = 129;
index_r[2089] = 129;
index_r[2090] = 129;
index_r[2091] = 129;
index_r[2092] = 129;
index_r[2093] = 129;
index_r[2094] = 129;
index_r[2095] = 129;
index_r[2096] = 129;
index_r[2097] = 129;
index_r[2098] = 129;
index_r[2099] = 129;
index_r[2100] = 130;
index_r[2101] = 130;
index_r[2102] = 130;
index_r[2103] = 130;
index_r[2104] = 130;
index_r[2105] = 130;
index_r[2106] = 130;
index_r[2107] = 130;
index_r[2108] = 130;
index_r[2109] = 130;
index_r[2110] = 130;
index_r[2111] = 130;
index_r[2112] = 130;
index_r[2113] = 130;
index_r[2114] = 130;
index_r[2115] = 130;
index_r[2116] = 130;
index_r[2117] = 130;
index_r[2118] = 130;
index_r[2119] = 131;
index_r[2120] = 131;
index_r[2121] = 131;
index_r[2122] = 131;
index_r[2123] = 131;
index_r[2124] = 131;
index_r[2125] = 131;
index_r[2126] = 131;
index_r[2127] = 131;
index_r[2128] = 131;
index_r[2129] = 131;
index_r[2130] = 131;
index_r[2131] = 132;
index_r[2132] = 132;
index_r[2133] = 132;
index_r[2134] = 132;
index_r[2135] = 132;
index_r[2136] = 132;
index_r[2137] = 132;
index_r[2138] = 132;
index_r[2139] = 132;
index_r[2140] = 132;
index_r[2141] = 132;
index_r[2142] = 132;
index_r[2143] = 132;
index_r[2144] = 132;
index_r[2145] = 132;
index_r[2146] = 132;
index_r[2147] = 133;
index_r[2148] = 133;
index_r[2149] = 133;
index_r[2150] = 133;
index_r[2151] = 133;
index_r[2152] = 133;
index_r[2153] = 133;
index_r[2154] = 133;
index_r[2155] = 133;
index_r[2156] = 133;
index_r[2157] = 133;
index_r[2158] = 133;
index_r[2159] = 133;
index_r[2160] = 133;
index_r[2161] = 134;
index_r[2162] = 134;
index_r[2163] = 134;
index_r[2164] = 134;
index_r[2165] = 134;
index_r[2166] = 134;
index_r[2167] = 134;
index_r[2168] = 135;
index_r[2169] = 135;
index_r[2170] = 135;
index_r[2171] = 135;
index_r[2172] = 135;
index_r[2173] = 135;
index_r[2174] = 135;
index_r[2175] = 135;
index_r[2176] = 135;
index_r[2177] = 135;
index_r[2178] = 135;
index_r[2179] = 135;
index_r[2180] = 135;
index_r[2181] = 135;
index_r[2182] = 135;
index_r[2183] = 135;
index_r[2184] = 135;
index_r[2185] = 135;
index_r[2186] = 135;
index_r[2187] = 135;
index_r[2188] = 135;
index_r[2189] = 135;
index_r[2190] = 135;
index_r[2191] = 135;
index_r[2192] = 136;
index_r[2193] = 136;
index_r[2194] = 136;
index_r[2195] = 136;
index_r[2196] = 136;
index_r[2197] = 136;
index_r[2198] = 136;
index_r[2199] = 136;
index_r[2200] = 136;
index_r[2201] = 136;
index_r[2202] = 136;
index_r[2203] = 136;
index_r[2204] = 136;
index_r[2205] = 136;
index_r[2206] = 136;
index_r[2207] = 136;
index_r[2208] = 136;
index_r[2209] = 136;
index_r[2210] = 136;
index_r[2211] = 137;
index_r[2212] = 137;
index_r[2213] = 137;
index_r[2214] = 137;
index_r[2215] = 137;
index_r[2216] = 137;
index_r[2217] = 137;
index_r[2218] = 137;
index_r[2219] = 137;
index_r[2220] = 137;
index_r[2221] = 138;
index_r[2222] = 138;
index_r[2223] = 138;
index_r[2224] = 138;
index_r[2225] = 138;
index_r[2226] = 138;
index_r[2227] = 138;
index_r[2228] = 138;
index_r[2229] = 138;
index_r[2230] = 138;
index_r[2231] = 138;
index_r[2232] = 138;
index_r[2233] = 138;
index_r[2234] = 138;
index_r[2235] = 138;
index_r[2236] = 138;
index_r[2237] = 138;
index_r[2238] = 139;
index_r[2239] = 139;
index_r[2240] = 139;
index_r[2241] = 139;
index_r[2242] = 139;
index_r[2243] = 139;
index_r[2244] = 139;
index_r[2245] = 139;
index_r[2246] = 139;
index_r[2247] = 139;
index_r[2248] = 139;
index_r[2249] = 139;
index_r[2250] = 139;
index_r[2251] = 139;
index_r[2252] = 140;
index_r[2253] = 140;
index_r[2254] = 140;
index_r[2255] = 140;
index_r[2256] = 140;
index_r[2257] = 140;
index_r[2258] = 140;
index_r[2259] = 140;
index_r[2260] = 140;
index_r[2261] = 140;
index_r[2262] = 140;
index_r[2263] = 140;
index_r[2264] = 140;
index_r[2265] = 140;
index_r[2266] = 140;
index_r[2267] = 140;
index_r[2268] = 140;
index_r[2269] = 140;
index_r[2270] = 140;
index_r[2271] = 141;
index_r[2272] = 141;
index_r[2273] = 141;
index_r[2274] = 141;
index_r[2275] = 141;
index_r[2276] = 141;
index_r[2277] = 141;
index_r[2278] = 141;
index_r[2279] = 141;
index_r[2280] = 141;
index_r[2281] = 141;
index_r[2282] = 141;
index_r[2283] = 141;
index_r[2284] = 141;
index_r[2285] = 141;
index_r[2286] = 141;
index_r[2287] = 141;
index_r[2288] = 141;
index_r[2289] = 141;
index_r[2290] = 142;
index_r[2291] = 142;
index_r[2292] = 142;
index_r[2293] = 142;
index_r[2294] = 142;
index_r[2295] = 142;
index_r[2296] = 142;
index_r[2297] = 142;
index_r[2298] = 142;
index_r[2299] = 142;
index_r[2300] = 142;
index_r[2301] = 142;
index_r[2302] = 142;
index_r[2303] = 142;
index_r[2304] = 143;
index_r[2305] = 143;
index_r[2306] = 143;
index_r[2307] = 143;
index_r[2308] = 143;
index_r[2309] = 143;
index_r[2310] = 143;
index_r[2311] = 143;
index_r[2312] = 143;
index_r[2313] = 143;
index_r[2314] = 143;
index_r[2315] = 143;
index_r[2316] = 143;
index_r[2317] = 143;
index_r[2318] = 143;
index_r[2319] = 143;
index_r[2320] = 143;
index_r[2321] = 143;
index_r[2322] = 143;
index_r[2323] = 144;
index_r[2324] = 144;
index_r[2325] = 144;
index_r[2326] = 144;
index_r[2327] = 144;
index_r[2328] = 144;
index_r[2329] = 144;
index_r[2330] = 144;
index_r[2331] = 144;
index_r[2332] = 144;
index_r[2333] = 145;
index_r[2334] = 145;
index_r[2335] = 145;
index_r[2336] = 145;
index_r[2337] = 145;
index_r[2338] = 145;
index_r[2339] = 145;
index_r[2340] = 145;
index_r[2341] = 145;
index_r[2342] = 145;
index_r[2343] = 145;
index_r[2344] = 145;
index_r[2345] = 145;
index_r[2346] = 145;
index_r[2347] = 145;
index_r[2348] = 145;
index_r[2349] = 145;
index_r[2350] = 145;
index_r[2351] = 145;
index_r[2352] = 145;
index_r[2353] = 145;
index_r[2354] = 145;
index_r[2355] = 146;
index_r[2356] = 146;
index_r[2357] = 146;
index_r[2358] = 146;
index_r[2359] = 146;
index_r[2360] = 146;
index_r[2361] = 146;
index_r[2362] = 146;
index_r[2363] = 146;
index_r[2364] = 146;
index_r[2365] = 146;
index_r[2366] = 146;
index_r[2367] = 146;
index_r[2368] = 146;
index_r[2369] = 146;
index_r[2370] = 146;
index_r[2371] = 146;
index_r[2372] = 147;
index_r[2373] = 147;
index_r[2374] = 147;
index_r[2375] = 147;
index_r[2376] = 147;
index_r[2377] = 147;
index_r[2378] = 147;
index_r[2379] = 147;
index_r[2380] = 147;
index_r[2381] = 147;
index_r[2382] = 147;
index_r[2383] = 147;
index_r[2384] = 147;
index_r[2385] = 147;
index_r[2386] = 148;
index_r[2387] = 148;
index_r[2388] = 148;
index_r[2389] = 148;
index_r[2390] = 148;
index_r[2391] = 148;
index_r[2392] = 148;
index_r[2393] = 148;
index_r[2394] = 148;
index_r[2395] = 148;
index_r[2396] = 148;
index_r[2397] = 148;
index_r[2398] = 149;
index_r[2399] = 149;
index_r[2400] = 149;
index_r[2401] = 149;
index_r[2402] = 149;
index_r[2403] = 149;
index_r[2404] = 149;
index_r[2405] = 149;
index_r[2406] = 149;
index_r[2407] = 149;
index_r[2408] = 150;
index_r[2409] = 150;
index_r[2410] = 150;
index_r[2411] = 150;
index_r[2412] = 150;
index_r[2413] = 150;
index_r[2414] = 150;
index_r[2415] = 150;
index_r[2416] = 150;
index_r[2417] = 150;
index_r[2418] = 150;
index_r[2419] = 150;
index_r[2420] = 150;
index_r[2421] = 150;
index_r[2422] = 150;
index_r[2423] = 150;
index_r[2424] = 150;
index_r[2425] = 151;
index_r[2426] = 151;
index_r[2427] = 151;
index_r[2428] = 151;
index_r[2429] = 151;
index_r[2430] = 151;
index_r[2431] = 151;
index_r[2432] = 151;
index_r[2433] = 151;
index_r[2434] = 151;
index_r[2435] = 151;
index_r[2436] = 151;
index_r[2437] = 151;
index_r[2438] = 151;
index_r[2439] = 152;
index_r[2440] = 152;
index_r[2441] = 152;
index_r[2442] = 152;
index_r[2443] = 152;
index_r[2444] = 152;
index_r[2445] = 152;
index_r[2446] = 152;
index_r[2447] = 152;
index_r[2448] = 152;
index_r[2449] = 152;
index_r[2450] = 152;
index_r[2451] = 152;
index_r[2452] = 152;
index_r[2453] = 152;
index_r[2454] = 152;
index_r[2455] = 152;
index_r[2456] = 153;
index_r[2457] = 153;
index_r[2458] = 153;
index_r[2459] = 153;
index_r[2460] = 153;
index_r[2461] = 153;
index_r[2462] = 153;
index_r[2463] = 153;
index_r[2464] = 153;
index_r[2465] = 153;
index_r[2466] = 153;
index_r[2467] = 153;
index_r[2468] = 153;
index_r[2469] = 153;
index_r[2470] = 154;
index_r[2471] = 154;
index_r[2472] = 154;
index_r[2473] = 154;
index_r[2474] = 154;
index_r[2475] = 154;
index_r[2476] = 154;
index_r[2477] = 154;
index_r[2478] = 154;
index_r[2479] = 154;
index_r[2480] = 154;
index_r[2481] = 154;
index_r[2482] = 155;
index_r[2483] = 155;
index_r[2484] = 155;
index_r[2485] = 155;
index_r[2486] = 155;
index_r[2487] = 155;
index_r[2488] = 155;
index_r[2489] = 156;
index_r[2490] = 156;
index_r[2491] = 156;
index_r[2492] = 156;
index_r[2493] = 156;
index_r[2494] = 156;
index_r[2495] = 156;
index_r[2496] = 156;
index_r[2497] = 156;
index_r[2498] = 156;
index_r[2499] = 156;
index_r[2500] = 156;
index_r[2501] = 156;
index_r[2502] = 156;
index_r[2503] = 156;
index_r[2504] = 156;
index_r[2505] = 156;
index_r[2506] = 157;
index_r[2507] = 157;
index_r[2508] = 157;
index_r[2509] = 157;
index_r[2510] = 157;
index_r[2511] = 157;
index_r[2512] = 157;
index_r[2513] = 157;
index_r[2514] = 157;
index_r[2515] = 157;
index_r[2516] = 157;
index_r[2517] = 157;
index_r[2518] = 157;
index_r[2519] = 157;
index_r[2520] = 157;
index_r[2521] = 157;
index_r[2522] = 157;
index_r[2523] = 158;
index_r[2524] = 158;
index_r[2525] = 158;
index_r[2526] = 158;
index_r[2527] = 158;
index_r[2528] = 158;
index_r[2529] = 158;
index_r[2530] = 158;
index_r[2531] = 158;
index_r[2532] = 158;
index_r[2533] = 158;
index_r[2534] = 158;
index_r[2535] = 158;
index_r[2536] = 158;
index_r[2537] = 158;
index_r[2538] = 158;
index_r[2539] = 158;
index_r[2540] = 158;
index_r[2541] = 158;
index_r[2542] = 159;
index_r[2543] = 159;
index_r[2544] = 159;
index_r[2545] = 159;
index_r[2546] = 159;
index_r[2547] = 159;
index_r[2548] = 159;
index_r[2549] = 159;
index_r[2550] = 159;
index_r[2551] = 159;
index_r[2552] = 159;
index_r[2553] = 159;
index_r[2554] = 159;
index_r[2555] = 159;
index_r[2556] = 159;
index_r[2557] = 159;
index_r[2558] = 159;
index_r[2559] = 159;
index_r[2560] = 159;
index_r[2561] = 159;
index_r[2562] = 159;
index_r[2563] = 159;
index_r[2564] = 160;
index_r[2565] = 160;
index_r[2566] = 160;
index_r[2567] = 160;
index_r[2568] = 160;
index_r[2569] = 160;
index_r[2570] = 160;
index_r[2571] = 160;
index_r[2572] = 160;
index_r[2573] = 160;
index_r[2574] = 160;
index_r[2575] = 160;
index_r[2576] = 160;
index_r[2577] = 160;
index_r[2578] = 160;
index_r[2579] = 160;
index_r[2580] = 160;
index_r[2581] = 160;
index_r[2582] = 160;
index_r[2583] = 161;
index_r[2584] = 161;
index_r[2585] = 161;
index_r[2586] = 161;
index_r[2587] = 161;
index_r[2588] = 161;
index_r[2589] = 161;
index_r[2590] = 161;
index_r[2591] = 161;
index_r[2592] = 161;
index_r[2593] = 161;
index_r[2594] = 161;
index_r[2595] = 161;
index_r[2596] = 161;
index_r[2597] = 162;
index_r[2598] = 162;
index_r[2599] = 162;
index_r[2600] = 162;
index_r[2601] = 162;
index_r[2602] = 162;
index_r[2603] = 162;
index_r[2604] = 162;
index_r[2605] = 162;
index_r[2606] = 162;
index_r[2607] = 162;
index_r[2608] = 162;
index_r[2609] = 162;
index_r[2610] = 162;
index_r[2611] = 162;
index_r[2612] = 162;
index_r[2613] = 162;
index_r[2614] = 162;
index_r[2615] = 162;
index_r[2616] = 162;
index_r[2617] = 162;
index_r[2618] = 162;
index_r[2619] = 162;
index_r[2620] = 162;
index_r[2621] = 163;
index_r[2622] = 163;
index_r[2623] = 163;
index_r[2624] = 163;
index_r[2625] = 163;
index_r[2626] = 163;
index_r[2627] = 163;
index_r[2628] = 163;
index_r[2629] = 163;
index_r[2630] = 163;
index_r[2631] = 163;
index_r[2632] = 163;
index_r[2633] = 163;
index_r[2634] = 163;
index_r[2635] = 163;
index_r[2636] = 163;
index_r[2637] = 163;
index_r[2638] = 164;
index_r[2639] = 164;
index_r[2640] = 164;
index_r[2641] = 164;
index_r[2642] = 164;
index_r[2643] = 164;
index_r[2644] = 164;
index_r[2645] = 164;
index_r[2646] = 164;
index_r[2647] = 164;
index_r[2648] = 164;
index_r[2649] = 164;
index_r[2650] = 164;
index_r[2651] = 164;
index_r[2652] = 164;
index_r[2653] = 165;
index_r[2654] = 165;
index_r[2655] = 165;
index_r[2656] = 165;
index_r[2657] = 165;
index_r[2658] = 165;
index_r[2659] = 165;
index_r[2660] = 165;
index_r[2661] = 165;
index_r[2662] = 165;
index_r[2663] = 165;
index_r[2664] = 165;
index_r[2665] = 165;
index_r[2666] = 165;
index_r[2667] = 165;
index_r[2668] = 165;
index_r[2669] = 165;
index_r[2670] = 165;
index_r[2671] = 165;
index_r[2672] = 166;
index_r[2673] = 166;
index_r[2674] = 166;
index_r[2675] = 166;
index_r[2676] = 166;
index_r[2677] = 166;
index_r[2678] = 166;
index_r[2679] = 167;
index_r[2680] = 167;
index_r[2681] = 167;
index_r[2682] = 167;
index_r[2683] = 167;
index_r[2684] = 167;
index_r[2685] = 167;
index_r[2686] = 167;
index_r[2687] = 167;
index_r[2688] = 167;
index_r[2689] = 167;
index_r[2690] = 167;
index_r[2691] = 167;
index_r[2692] = 167;
index_r[2693] = 167;
index_r[2694] = 167;
index_r[2695] = 167;
index_r[2696] = 168;
index_r[2697] = 168;
index_r[2698] = 168;
index_r[2699] = 168;
index_r[2700] = 168;
index_r[2701] = 168;
index_r[2702] = 168;
index_r[2703] = 168;
index_r[2704] = 168;
index_r[2705] = 168;
index_r[2706] = 168;
index_r[2707] = 168;
index_r[2708] = 168;
index_r[2709] = 168;
index_r[2710] = 168;
index_r[2711] = 168;
index_r[2712] = 168;
index_r[2713] = 168;
index_r[2714] = 168;
index_r[2715] = 169;
index_r[2716] = 169;
index_r[2717] = 169;
index_r[2718] = 169;
index_r[2719] = 169;
index_r[2720] = 169;
index_r[2721] = 169;
index_r[2722] = 169;
index_r[2723] = 169;
index_r[2724] = 169;
index_r[2725] = 169;
index_r[2726] = 169;
index_r[2727] = 169;
index_r[2728] = 169;
index_r[2729] = 169;
index_r[2730] = 169;
index_r[2731] = 170;
index_r[2732] = 170;
index_r[2733] = 170;
index_r[2734] = 170;
index_r[2735] = 170;
index_r[2736] = 170;
index_r[2737] = 170;
index_r[2738] = 171;
index_r[2739] = 171;
index_r[2740] = 171;
index_r[2741] = 171;
index_r[2742] = 171;
index_r[2743] = 171;
index_r[2744] = 171;
index_r[2745] = 171;
index_r[2746] = 171;
index_r[2747] = 171;
index_r[2748] = 171;
index_r[2749] = 172;
index_r[2750] = 172;
index_r[2751] = 172;
index_r[2752] = 172;
index_r[2753] = 172;
index_r[2754] = 172;
index_r[2755] = 172;
index_r[2756] = 172;
index_r[2757] = 172;
index_r[2758] = 172;
index_r[2759] = 172;
index_r[2760] = 173;
index_r[2761] = 173;
index_r[2762] = 173;
index_r[2763] = 173;
index_r[2764] = 173;
index_r[2765] = 173;
index_r[2766] = 173;
index_r[2767] = 173;
index_r[2768] = 173;
index_r[2769] = 173;
index_r[2770] = 173;
index_r[2771] = 173;
index_r[2772] = 173;
index_r[2773] = 173;
index_r[2774] = 173;
index_r[2775] = 173;
index_r[2776] = 173;
index_r[2777] = 173;
index_r[2778] = 173;
index_r[2779] = 173;
index_r[2780] = 173;
index_r[2781] = 173;
index_r[2782] = 173;
index_r[2783] = 173;
index_r[2784] = 174;
index_r[2785] = 174;
index_r[2786] = 174;
index_r[2787] = 174;
index_r[2788] = 174;
index_r[2789] = 174;
index_r[2790] = 174;
index_r[2791] = 174;
index_r[2792] = 174;
index_r[2793] = 174;
index_r[2794] = 174;
index_r[2795] = 174;
index_r[2796] = 174;
index_r[2797] = 174;
index_r[2798] = 174;
index_r[2799] = 175;
index_r[2800] = 175;
index_r[2801] = 175;
index_r[2802] = 175;
index_r[2803] = 175;
index_r[2804] = 175;
index_r[2805] = 175;
index_r[2806] = 175;
index_r[2807] = 175;
index_r[2808] = 175;
index_r[2809] = 175;
index_r[2810] = 175;
index_r[2811] = 175;
index_r[2812] = 175;
index_r[2813] = 176;
index_r[2814] = 176;
index_r[2815] = 176;
index_r[2816] = 176;
index_r[2817] = 176;
index_r[2818] = 176;
index_r[2819] = 176;
index_r[2820] = 176;
index_r[2821] = 176;
index_r[2822] = 176;
index_r[2823] = 176;
index_r[2824] = 176;
index_r[2825] = 176;
index_r[2826] = 176;
index_r[2827] = 176;
index_r[2828] = 176;
index_r[2829] = 177;
index_r[2830] = 177;
index_r[2831] = 177;
index_r[2832] = 177;
index_r[2833] = 177;
index_r[2834] = 177;
index_r[2835] = 177;
index_r[2836] = 178;
index_r[2837] = 178;
index_r[2838] = 178;
index_r[2839] = 178;
index_r[2840] = 178;
index_r[2841] = 178;
index_r[2842] = 178;
index_r[2843] = 178;
index_r[2844] = 178;
index_r[2845] = 178;
index_r[2846] = 178;
index_r[2847] = 178;
index_r[2848] = 178;
index_r[2849] = 178;
index_r[2850] = 178;
index_r[2851] = 178;
index_r[2852] = 178;
index_r[2853] = 178;
index_r[2854] = 178;
index_r[2855] = 178;
index_r[2856] = 178;
index_r[2857] = 178;
index_r[2858] = 178;
index_r[2859] = 178;
index_r[2860] = 179;
index_r[2861] = 179;
index_r[2862] = 179;
index_r[2863] = 179;
index_r[2864] = 179;
index_r[2865] = 179;
index_r[2866] = 179;
index_r[2867] = 179;
index_r[2868] = 179;
index_r[2869] = 179;
index_r[2870] = 179;
index_r[2871] = 179;
index_r[2872] = 179;
index_r[2873] = 179;
index_r[2874] = 179;
index_r[2875] = 179;
index_r[2876] = 179;
index_r[2877] = 179;
index_r[2878] = 179;
index_r[2879] = 180;
index_r[2880] = 180;
index_r[2881] = 180;
index_r[2882] = 180;
index_r[2883] = 180;
index_r[2884] = 180;
index_r[2885] = 180;
index_r[2886] = 180;
index_r[2887] = 180;
index_r[2888] = 180;
index_r[2889] = 180;
index_r[2890] = 180;
index_r[2891] = 180;
index_r[2892] = 180;
index_r[2893] = 180;
index_r[2894] = 180;
index_r[2895] = 180;
index_r[2896] = 180;
index_r[2897] = 180;
index_r[2898] = 181;
index_r[2899] = 181;
index_r[2900] = 181;
index_r[2901] = 181;
index_r[2902] = 181;
index_r[2903] = 181;
index_r[2904] = 181;
index_r[2905] = 181;
index_r[2906] = 181;
index_r[2907] = 181;
index_r[2908] = 181;
index_r[2909] = 181;
index_r[2910] = 181;
index_r[2911] = 181;
index_r[2912] = 181;
index_r[2913] = 181;
index_r[2914] = 181;
index_r[2915] = 181;
index_r[2916] = 181;
index_r[2917] = 181;
index_r[2918] = 181;
index_r[2919] = 181;
index_r[2920] = 182;
index_r[2921] = 182;
index_r[2922] = 182;
index_r[2923] = 182;
index_r[2924] = 182;
index_r[2925] = 182;
index_r[2926] = 182;
index_r[2927] = 182;
index_r[2928] = 182;
index_r[2929] = 182;
index_r[2930] = 182;
index_r[2931] = 182;
index_r[2932] = 182;
index_r[2933] = 182;
index_r[2934] = 182;
index_r[2935] = 182;
index_r[2936] = 182;
index_r[2937] = 183;
index_r[2938] = 183;
index_r[2939] = 183;
index_r[2940] = 183;
index_r[2941] = 183;
index_r[2942] = 183;
index_r[2943] = 183;
index_r[2944] = 183;
index_r[2945] = 183;
index_r[2946] = 183;
index_r[2947] = 183;
index_r[2948] = 183;
index_r[2949] = 183;
index_r[2950] = 183;
index_r[2951] = 183;
index_r[2952] = 183;
index_r[2953] = 183;
index_r[2954] = 183;
index_r[2955] = 183;
index_r[2956] = 184;
index_r[2957] = 184;
index_r[2958] = 184;
index_r[2959] = 184;
index_r[2960] = 184;
index_r[2961] = 184;
index_r[2962] = 184;
index_r[2963] = 184;
index_r[2964] = 184;
index_r[2965] = 184;
index_r[2966] = 184;
index_r[2967] = 184;
index_r[2968] = 184;
index_r[2969] = 184;
index_r[2970] = 184;
index_r[2971] = 185;
index_r[2972] = 185;
index_r[2973] = 185;
index_r[2974] = 185;
index_r[2975] = 185;
index_r[2976] = 185;
index_r[2977] = 185;
index_r[2978] = 185;
index_r[2979] = 185;
index_r[2980] = 185;
index_r[2981] = 185;
index_r[2982] = 185;
index_r[2983] = 186;
index_r[2984] = 186;
index_r[2985] = 186;
index_r[2986] = 186;
index_r[2987] = 186;
index_r[2988] = 186;
index_r[2989] = 186;
index_r[2990] = 186;
index_r[2991] = 186;
index_r[2992] = 186;
index_r[2993] = 186;
index_r[2994] = 186;
index_r[2995] = 186;
index_r[2996] = 186;
index_r[2997] = 186;
index_r[2998] = 186;
index_r[2999] = 186;
index_r[3000] = 187;
index_r[3001] = 187;
index_r[3002] = 187;
index_r[3003] = 187;
index_r[3004] = 187;
index_r[3005] = 187;
index_r[3006] = 187;
index_r[3007] = 187;
index_r[3008] = 187;
index_r[3009] = 187;
index_r[3010] = 187;
index_r[3011] = 187;
index_r[3012] = 187;
index_r[3013] = 187;
index_r[3014] = 188;
index_r[3015] = 188;
index_r[3016] = 188;
index_r[3017] = 188;
index_r[3018] = 188;
index_r[3019] = 188;
index_r[3020] = 188;
index_r[3021] = 188;
index_r[3022] = 188;
index_r[3023] = 188;
index_r[3024] = 188;
index_r[3025] = 188;
index_r[3026] = 188;
index_r[3027] = 188;
index_r[3028] = 188;
index_r[3029] = 188;
index_r[3030] = 188;
index_r[3031] = 188;
index_r[3032] = 188;
index_r[3033] = 189;
index_r[3034] = 189;
index_r[3035] = 189;
index_r[3036] = 189;
index_r[3037] = 189;
index_r[3038] = 189;
index_r[3039] = 189;
index_r[3040] = 189;
index_r[3041] = 189;
index_r[3042] = 189;
index_r[3043] = 189;
index_r[3044] = 189;
index_r[3045] = 189;
index_r[3046] = 189;
index_r[3047] = 189;
index_r[3048] = 189;
index_r[3049] = 189;
index_r[3050] = 189;
index_r[3051] = 189;
index_r[3052] = 190;
index_r[3053] = 190;
index_r[3054] = 190;
index_r[3055] = 190;
index_r[3056] = 190;
index_r[3057] = 190;
index_r[3058] = 190;
index_r[3059] = 190;
index_r[3060] = 190;
index_r[3061] = 190;
index_r[3062] = 190;
index_r[3063] = 191;
index_r[3064] = 191;
index_r[3065] = 191;
index_r[3066] = 191;
index_r[3067] = 191;
index_r[3068] = 191;
index_r[3069] = 191;
index_r[3070] = 191;
index_r[3071] = 191;
index_r[3072] = 191;
index_r[3073] = 192;
index_r[3074] = 192;
index_r[3075] = 192;
index_r[3076] = 192;
index_r[3077] = 192;
index_r[3078] = 192;
index_r[3079] = 192;
index_r[3080] = 192;
index_r[3081] = 192;
index_r[3082] = 192;
index_r[3083] = 192;
index_r[3084] = 192;
index_r[3085] = 192;
index_r[3086] = 192;
index_r[3087] = 192;
index_r[3088] = 192;
index_r[3089] = 192;
index_r[3090] = 193;
index_r[3091] = 193;
index_r[3092] = 193;
index_r[3093] = 193;
index_r[3094] = 193;
index_r[3095] = 193;
index_r[3096] = 193;
index_r[3097] = 194;
index_r[3098] = 194;
index_r[3099] = 194;
index_r[3100] = 194;
index_r[3101] = 194;
index_r[3102] = 194;
index_r[3103] = 194;
index_r[3104] = 194;
index_r[3105] = 194;
index_r[3106] = 194;
index_r[3107] = 194;
index_r[3108] = 194;
index_r[3109] = 194;
index_r[3110] = 194;
index_r[3111] = 194;
index_r[3112] = 194;
index_r[3113] = 194;
index_r[3114] = 194;
index_r[3115] = 194;
index_r[3116] = 194;
index_r[3117] = 194;
index_r[3118] = 194;
index_r[3119] = 195;
index_r[3120] = 195;
index_r[3121] = 195;
index_r[3122] = 195;
index_r[3123] = 195;
index_r[3124] = 195;
index_r[3125] = 195;
index_r[3126] = 195;
index_r[3127] = 195;
index_r[3128] = 195;
index_r[3129] = 195;
index_r[3130] = 195;
index_r[3131] = 195;
index_r[3132] = 195;
index_r[3133] = 196;
index_r[3134] = 196;
index_r[3135] = 196;
index_r[3136] = 196;
index_r[3137] = 196;
index_r[3138] = 196;
index_r[3139] = 196;
index_r[3140] = 196;
index_r[3141] = 196;
index_r[3142] = 196;
index_r[3143] = 196;
index_r[3144] = 196;
index_r[3145] = 196;
index_r[3146] = 196;
index_r[3147] = 196;
index_r[3148] = 196;
index_r[3149] = 196;
index_r[3150] = 196;
index_r[3151] = 196;
index_r[3152] = 197;
index_r[3153] = 197;
index_r[3154] = 197;
index_r[3155] = 197;
index_r[3156] = 197;
index_r[3157] = 197;
index_r[3158] = 197;
index_r[3159] = 197;
index_r[3160] = 197;
index_r[3161] = 197;
index_r[3162] = 197;
index_r[3163] = 197;
index_r[3164] = 197;
index_r[3165] = 197;
index_r[3166] = 197;
index_r[3167] = 197;
index_r[3168] = 198;
index_r[3169] = 198;
index_r[3170] = 198;
index_r[3171] = 198;
index_r[3172] = 198;
index_r[3173] = 198;
index_r[3174] = 198;
index_r[3175] = 198;
index_r[3176] = 198;
index_r[3177] = 198;
index_r[3178] = 198;
index_r[3179] = 198;
index_r[3180] = 198;
index_r[3181] = 198;
index_r[3182] = 198;
index_r[3183] = 198;
index_r[3184] = 199;
index_r[3185] = 199;
index_r[3186] = 199;
index_r[3187] = 199;
index_r[3188] = 199;
index_r[3189] = 199;
index_r[3190] = 199;
index_r[3191] = 199;
index_r[3192] = 199;
index_r[3193] = 199;
index_r[3194] = 199;
index_r[3195] = 199;
index_r[3196] = 199;
index_r[3197] = 199;
index_r[3198] = 199;
index_r[3199] = 199;
index_r[3200] = 199;
index_r[3201] = 199;
index_r[3202] = 199;
index_r[3203] = 199;
index_r[3204] = 199;
index_r[3205] = 199;
index_r[3206] = 200;
index_r[3207] = 200;
index_r[3208] = 200;
index_r[3209] = 200;
index_r[3210] = 200;
index_r[3211] = 200;
index_r[3212] = 200;
index_r[3213] = 200;
index_r[3214] = 200;
index_r[3215] = 200;
index_r[3216] = 200;
index_r[3217] = 200;
index_r[3218] = 200;
index_r[3219] = 200;
index_r[3220] = 200;
index_r[3221] = 200;
index_r[3222] = 201;
index_r[3223] = 201;
index_r[3224] = 201;
index_r[3225] = 201;
index_r[3226] = 201;
index_r[3227] = 201;
index_r[3228] = 201;
index_r[3229] = 201;
index_r[3230] = 201;
index_r[3231] = 201;
index_r[3232] = 201;
index_r[3233] = 201;
index_r[3234] = 202;
index_r[3235] = 202;
index_r[3236] = 202;
index_r[3237] = 202;
index_r[3238] = 202;
index_r[3239] = 202;
index_r[3240] = 202;
index_r[3241] = 202;
index_r[3242] = 202;
index_r[3243] = 202;
index_r[3244] = 202;
index_r[3245] = 202;
index_r[3246] = 202;
index_r[3247] = 202;
index_r[3248] = 203;
index_r[3249] = 203;
index_r[3250] = 203;
index_r[3251] = 203;
index_r[3252] = 203;
index_r[3253] = 203;
index_r[3254] = 203;
index_r[3255] = 203;
index_r[3256] = 203;
index_r[3257] = 203;
index_r[3258] = 203;
index_r[3259] = 203;
index_r[3260] = 203;
index_r[3261] = 203;
index_r[3262] = 204;
index_r[3263] = 204;
index_r[3264] = 204;
index_r[3265] = 204;
index_r[3266] = 204;
index_r[3267] = 204;
index_r[3268] = 204;
index_r[3269] = 204;
index_r[3270] = 204;
index_r[3271] = 204;
index_r[3272] = 204;
index_r[3273] = 204;
index_r[3274] = 204;
index_r[3275] = 204;
index_r[3276] = 204;
index_r[3277] = 204;
index_r[3278] = 204;
index_r[3279] = 204;
index_r[3280] = 204;
index_r[3281] = 205;
index_r[3282] = 205;
index_r[3283] = 205;
index_r[3284] = 205;
index_r[3285] = 205;
index_r[3286] = 205;
index_r[3287] = 205;
index_r[3288] = 205;
index_r[3289] = 205;
index_r[3290] = 205;
index_r[3291] = 205;
index_r[3292] = 205;
index_r[3293] = 205;
index_r[3294] = 205;
index_r[3295] = 205;
index_r[3296] = 206;
index_r[3297] = 206;
index_r[3298] = 206;
index_r[3299] = 206;
index_r[3300] = 206;
index_r[3301] = 206;
index_r[3302] = 206;
index_r[3303] = 206;
index_r[3304] = 206;
index_r[3305] = 206;
index_r[3306] = 206;
index_r[3307] = 206;
index_r[3308] = 206;
index_r[3309] = 206;
index_r[3310] = 206;
index_r[3311] = 206;
index_r[3312] = 206;
index_r[3313] = 206;
index_r[3314] = 206;
index_r[3315] = 206;
index_r[3316] = 206;
index_r[3317] = 206;
index_r[3318] = 206;
index_r[3319] = 206;
index_r[3320] = 207;
index_r[3321] = 207;
index_r[3322] = 207;
index_r[3323] = 207;
index_r[3324] = 207;
index_r[3325] = 207;
index_r[3326] = 207;
index_r[3327] = 207;
index_r[3328] = 207;
index_r[3329] = 207;
index_r[3330] = 207;
index_r[3331] = 207;
index_r[3332] = 207;
index_r[3333] = 207;
index_r[3334] = 207;
index_r[3335] = 207;
index_r[3336] = 207;
index_r[3337] = 207;
index_r[3338] = 207;
index_r[3339] = 207;
index_r[3340] = 208;
index_r[3341] = 208;
index_r[3342] = 208;
index_r[3343] = 208;
index_r[3344] = 208;
index_r[3345] = 208;
index_r[3346] = 208;
index_r[3347] = 208;
index_r[3348] = 208;
index_r[3349] = 208;
index_r[3350] = 208;
index_r[3351] = 208;
index_r[3352] = 208;
index_r[3353] = 208;
index_r[3354] = 208;
index_r[3355] = 208;
index_r[3356] = 208;
index_r[3357] = 208;
index_r[3358] = 208;
index_r[3359] = 209;
index_r[3360] = 209;
index_r[3361] = 209;
index_r[3362] = 209;
index_r[3363] = 209;
index_r[3364] = 209;
index_r[3365] = 209;
index_r[3366] = 209;
index_r[3367] = 209;
index_r[3368] = 209;
index_r[3369] = 209;
index_r[3370] = 210;
index_r[3371] = 210;
index_r[3372] = 210;
index_r[3373] = 210;
index_r[3374] = 210;
index_r[3375] = 210;
index_r[3376] = 210;
index_r[3377] = 210;
index_r[3378] = 210;
index_r[3379] = 210;
index_r[3380] = 210;
index_r[3381] = 210;
index_r[3382] = 210;
index_r[3383] = 210;
index_r[3384] = 210;
index_r[3385] = 210;
index_r[3386] = 210;
index_r[3387] = 211;
index_r[3388] = 211;
index_r[3389] = 211;
index_r[3390] = 211;
index_r[3391] = 211;
index_r[3392] = 211;
index_r[3393] = 211;
index_r[3394] = 211;
index_r[3395] = 211;
index_r[3396] = 211;
index_r[3397] = 211;
index_r[3398] = 212;
index_r[3399] = 212;
index_r[3400] = 212;
index_r[3401] = 212;
index_r[3402] = 212;
index_r[3403] = 212;
index_r[3404] = 212;
index_r[3405] = 212;
index_r[3406] = 212;
index_r[3407] = 212;
index_r[3408] = 212;
index_r[3409] = 212;
index_r[3410] = 212;
index_r[3411] = 212;
index_r[3412] = 212;
index_r[3413] = 212;
index_r[3414] = 212;
index_r[3415] = 213;
index_r[3416] = 213;
index_r[3417] = 213;
index_r[3418] = 213;
index_r[3419] = 213;
index_r[3420] = 213;
index_r[3421] = 213;
index_r[3422] = 213;
index_r[3423] = 213;
index_r[3424] = 213;
index_r[3425] = 213;
index_r[3426] = 213;
index_r[3427] = 213;
index_r[3428] = 213;
index_r[3429] = 213;
index_r[3430] = 213;
index_r[3431] = 213;
index_r[3432] = 213;
index_r[3433] = 213;
index_r[3434] = 214;
index_r[3435] = 214;
index_r[3436] = 214;
index_r[3437] = 214;
index_r[3438] = 214;
index_r[3439] = 214;
index_r[3440] = 214;
index_r[3441] = 214;
index_r[3442] = 214;
index_r[3443] = 214;
index_r[3444] = 214;
index_r[3445] = 214;
index_r[3446] = 214;
index_r[3447] = 214;
index_r[3448] = 214;
index_r[3449] = 214;
index_r[3450] = 214;
index_r[3451] = 215;
index_r[3452] = 215;
index_r[3453] = 215;
index_r[3454] = 215;
index_r[3455] = 215;
index_r[3456] = 215;
index_r[3457] = 215;
index_r[3458] = 215;
index_r[3459] = 215;
index_r[3460] = 215;
index_r[3461] = 215;
index_r[3462] = 215;
index_r[3463] = 215;
index_r[3464] = 215;
index_r[3465] = 215;
index_r[3466] = 215;
index_r[3467] = 215;
index_r[3468] = 215;
index_r[3469] = 215;
index_r[3470] = 215;
index_r[3471] = 215;
index_r[3472] = 215;
index_r[3473] = 216;
index_r[3474] = 216;
index_r[3475] = 216;
index_r[3476] = 216;
index_r[3477] = 216;
index_r[3478] = 216;
index_r[3479] = 216;
index_r[3480] = 216;
index_r[3481] = 216;
index_r[3482] = 216;
index_r[3483] = 216;
index_r[3484] = 216;
index_r[3485] = 216;
index_r[3486] = 216;
index_r[3487] = 216;
index_r[3488] = 216;
index_r[3489] = 216;
index_r[3490] = 216;
index_r[3491] = 216;
index_r[3492] = 216;
index_r[3493] = 216;
index_r[3494] = 217;
index_r[3495] = 217;
index_r[3496] = 217;
index_r[3497] = 217;
index_r[3498] = 217;
index_r[3499] = 217;
index_r[3500] = 217;
index_r[3501] = 217;
index_r[3502] = 217;
index_r[3503] = 217;
index_r[3504] = 217;
index_r[3505] = 217;
index_r[3506] = 217;
index_r[3507] = 217;
index_r[3508] = 218;
index_r[3509] = 218;
index_r[3510] = 218;
index_r[3511] = 218;
index_r[3512] = 218;
index_r[3513] = 218;
index_r[3514] = 218;
index_r[3515] = 218;
index_r[3516] = 218;
index_r[3517] = 218;
index_r[3518] = 218;
index_r[3519] = 219;
index_r[3520] = 219;
index_r[3521] = 219;
index_r[3522] = 219;
index_r[3523] = 219;
index_r[3524] = 219;
index_r[3525] = 219;
index_r[3526] = 219;
index_r[3527] = 219;
index_r[3528] = 219;
index_r[3529] = 219;
index_r[3530] = 219;
index_r[3531] = 219;
index_r[3532] = 219;
index_r[3533] = 219;
index_r[3534] = 219;
index_r[3535] = 219;
index_r[3536] = 219;
index_r[3537] = 219;
index_r[3538] = 219;
index_r[3539] = 219;
index_r[3540] = 219;
index_r[3541] = 220;
index_r[3542] = 220;
index_r[3543] = 220;
index_r[3544] = 220;
index_r[3545] = 220;
index_r[3546] = 220;
index_r[3547] = 220;
index_r[3548] = 220;
index_r[3549] = 220;
index_r[3550] = 220;
index_r[3551] = 220;
index_r[3552] = 221;
index_r[3553] = 221;
index_r[3554] = 221;
index_r[3555] = 221;
index_r[3556] = 221;
index_r[3557] = 221;
index_r[3558] = 221;
index_r[3559] = 221;
index_r[3560] = 221;
index_r[3561] = 221;
index_r[3562] = 221;
index_r[3563] = 221;
index_r[3564] = 221;
index_r[3565] = 221;
index_r[3566] = 222;
index_r[3567] = 222;
index_r[3568] = 222;
index_r[3569] = 222;
index_r[3570] = 222;
index_r[3571] = 222;
index_r[3572] = 222;
index_r[3573] = 222;
index_r[3574] = 222;
index_r[3575] = 222;
index_r[3576] = 222;
index_r[3577] = 222;
index_r[3578] = 222;
index_r[3579] = 222;
index_r[3580] = 222;
index_r[3581] = 222;
index_r[3582] = 222;
index_r[3583] = 222;
index_r[3584] = 222;
index_r[3585] = 223;
index_r[3586] = 223;
index_r[3587] = 223;
index_r[3588] = 223;
index_r[3589] = 223;
index_r[3590] = 223;
index_r[3591] = 223;
index_r[3592] = 223;
index_r[3593] = 223;
index_r[3594] = 223;
index_r[3595] = 223;
index_r[3596] = 223;
index_r[3597] = 223;
index_r[3598] = 223;
index_r[3599] = 223;
index_r[3600] = 223;
index_r[3601] = 223;
index_r[3602] = 223;
index_r[3603] = 223;
index_r[3604] = 224;
index_r[3605] = 224;
index_r[3606] = 224;
index_r[3607] = 224;
index_r[3608] = 224;
index_r[3609] = 224;
index_r[3610] = 224;
index_r[3611] = 224;
index_r[3612] = 224;
index_r[3613] = 224;
index_r[3614] = 224;
index_r[3615] = 224;
index_r[3616] = 224;
index_r[3617] = 224;
index_r[3618] = 224;
index_r[3619] = 224;
index_r[3620] = 224;
index_r[3621] = 225;
index_r[3622] = 225;
index_r[3623] = 225;
index_r[3624] = 225;
index_r[3625] = 225;
index_r[3626] = 225;
index_r[3627] = 225;
index_r[3628] = 225;
index_r[3629] = 225;
index_r[3630] = 225;
index_r[3631] = 225;
index_r[3632] = 225;
index_r[3633] = 225;
index_r[3634] = 225;
index_r[3635] = 225;
index_r[3636] = 225;
index_r[3637] = 225;
index_r[3638] = 226;
index_r[3639] = 226;
index_r[3640] = 226;
index_r[3641] = 226;
index_r[3642] = 226;
index_r[3643] = 226;
index_r[3644] = 226;
index_r[3645] = 227;
index_r[3646] = 227;
index_r[3647] = 227;
index_r[3648] = 227;
index_r[3649] = 227;
index_r[3650] = 227;
index_r[3651] = 227;
index_r[3652] = 227;
index_r[3653] = 227;
index_r[3654] = 227;
index_r[3655] = 227;
index_r[3656] = 227;
index_r[3657] = 227;
index_r[3658] = 227;
index_r[3659] = 227;
index_r[3660] = 228;
index_r[3661] = 228;
index_r[3662] = 228;
index_r[3663] = 228;
index_r[3664] = 228;
index_r[3665] = 228;
index_r[3666] = 228;
index_r[3667] = 228;
index_r[3668] = 228;
index_r[3669] = 228;
index_r[3670] = 228;
index_r[3671] = 228;
index_r[3672] = 228;
index_r[3673] = 228;
index_r[3674] = 228;
index_r[3675] = 228;
index_r[3676] = 228;
index_r[3677] = 228;
index_r[3678] = 228;
index_r[3679] = 228;
index_r[3680] = 228;
index_r[3681] = 228;
index_r[3682] = 229;
index_r[3683] = 229;
index_r[3684] = 229;
index_r[3685] = 229;
index_r[3686] = 229;
index_r[3687] = 229;
index_r[3688] = 229;
index_r[3689] = 229;
index_r[3690] = 229;
index_r[3691] = 229;
index_r[3692] = 230;
index_r[3693] = 230;
index_r[3694] = 230;
index_r[3695] = 230;
index_r[3696] = 230;
index_r[3697] = 230;
index_r[3698] = 230;
index_r[3699] = 230;
index_r[3700] = 230;
index_r[3701] = 230;
index_r[3702] = 230;
index_r[3703] = 230;
index_r[3704] = 230;
index_r[3705] = 230;
index_r[3706] = 230;
index_r[3707] = 231;
index_r[3708] = 231;
index_r[3709] = 231;
index_r[3710] = 231;
index_r[3711] = 231;
index_r[3712] = 231;
index_r[3713] = 231;
index_r[3714] = 231;
index_r[3715] = 231;
index_r[3716] = 231;
index_r[3717] = 231;
index_r[3718] = 231;
index_r[3719] = 231;
index_r[3720] = 231;
index_r[3721] = 232;
index_r[3722] = 232;
index_r[3723] = 232;
index_r[3724] = 232;
index_r[3725] = 232;
index_r[3726] = 232;
index_r[3727] = 232;
index_r[3728] = 232;
index_r[3729] = 232;
index_r[3730] = 232;
index_r[3731] = 232;
index_r[3732] = 232;
index_r[3733] = 232;
index_r[3734] = 232;
index_r[3735] = 232;
index_r[3736] = 232;
index_r[3737] = 232;
index_r[3738] = 232;
index_r[3739] = 232;
index_r[3740] = 233;
index_r[3741] = 233;
index_r[3742] = 233;
index_r[3743] = 233;
index_r[3744] = 233;
index_r[3745] = 233;
index_r[3746] = 233;
index_r[3747] = 233;
index_r[3748] = 233;
index_r[3749] = 233;
index_r[3750] = 233;
index_r[3751] = 233;
index_r[3752] = 233;
index_r[3753] = 233;
index_r[3754] = 233;
index_r[3755] = 233;
index_r[3756] = 233;
index_r[3757] = 233;
index_r[3758] = 233;
index_r[3759] = 233;
index_r[3760] = 233;
index_r[3761] = 234;
index_r[3762] = 234;
index_r[3763] = 234;
index_r[3764] = 234;
index_r[3765] = 234;
index_r[3766] = 234;
index_r[3767] = 234;
index_r[3768] = 234;
index_r[3769] = 234;
index_r[3770] = 234;
index_r[3771] = 234;
index_r[3772] = 234;
index_r[3773] = 234;
index_r[3774] = 234;
index_r[3775] = 234;
index_r[3776] = 234;
index_r[3777] = 234;
index_r[3778] = 234;
index_r[3779] = 234;
index_r[3780] = 234;
index_r[3781] = 234;
index_r[3782] = 235;
index_r[3783] = 235;
index_r[3784] = 235;
index_r[3785] = 235;
index_r[3786] = 235;
index_r[3787] = 235;
index_r[3788] = 235;
index_r[3789] = 235;
index_r[3790] = 235;
index_r[3791] = 235;
index_r[3792] = 235;
index_r[3793] = 235;
index_r[3794] = 235;
index_r[3795] = 235;
index_r[3796] = 235;
index_r[3797] = 235;
index_r[3798] = 235;
index_r[3799] = 235;
index_r[3800] = 235;
index_r[3801] = 236;
index_r[3802] = 236;
index_r[3803] = 236;
index_r[3804] = 236;
index_r[3805] = 236;
index_r[3806] = 236;
index_r[3807] = 236;
index_r[3808] = 236;
index_r[3809] = 236;
index_r[3810] = 236;
index_r[3811] = 236;
index_r[3812] = 236;
index_r[3813] = 236;
index_r[3814] = 236;
index_r[3815] = 236;
index_r[3816] = 236;
index_r[3817] = 237;
index_r[3818] = 237;
index_r[3819] = 237;
index_r[3820] = 237;
index_r[3821] = 237;
index_r[3822] = 237;
index_r[3823] = 237;
index_r[3824] = 237;
index_r[3825] = 237;
index_r[3826] = 237;
index_r[3827] = 237;
index_r[3828] = 237;
index_r[3829] = 237;
index_r[3830] = 237;
index_r[3831] = 237;
index_r[3832] = 237;
index_r[3833] = 237;
index_r[3834] = 237;
index_r[3835] = 237;
index_r[3836] = 237;
index_r[3837] = 237;
index_r[3838] = 237;
index_r[3839] = 238;
index_r[3840] = 238;
index_r[3841] = 238;
index_r[3842] = 238;
index_r[3843] = 238;
index_r[3844] = 238;
index_r[3845] = 238;
index_r[3846] = 239;
index_r[3847] = 239;
index_r[3848] = 239;
index_r[3849] = 239;
index_r[3850] = 239;
index_r[3851] = 239;
index_r[3852] = 239;
index_r[3853] = 239;
index_r[3854] = 239;
index_r[3855] = 239;
index_r[3856] = 239;
index_r[3857] = 240;
index_r[3858] = 240;
index_r[3859] = 240;
index_r[3860] = 240;
index_r[3861] = 240;
index_r[3862] = 240;
index_r[3863] = 240;
index_r[3864] = 240;
index_r[3865] = 240;
index_r[3866] = 240;
index_r[3867] = 240;
index_r[3868] = 240;
index_r[3869] = 240;
index_r[3870] = 240;
index_r[3871] = 240;
index_r[3872] = 240;
index_r[3873] = 241;
index_r[3874] = 241;
index_r[3875] = 241;
index_r[3876] = 241;
index_r[3877] = 241;
index_r[3878] = 241;
index_r[3879] = 241;
index_r[3880] = 241;
index_r[3881] = 241;
index_r[3882] = 241;
index_r[3883] = 242;
index_r[3884] = 242;
index_r[3885] = 242;
index_r[3886] = 242;
index_r[3887] = 242;
index_r[3888] = 242;
index_r[3889] = 242;
index_r[3890] = 242;
index_r[3891] = 242;
index_r[3892] = 242;
index_r[3893] = 242;
index_r[3894] = 242;
index_r[3895] = 242;
index_r[3896] = 242;
index_r[3897] = 242;
index_r[3898] = 242;
index_r[3899] = 243;
index_r[3900] = 243;
index_r[3901] = 243;
index_r[3902] = 243;
index_r[3903] = 243;
index_r[3904] = 243;
index_r[3905] = 243;
index_r[3906] = 243;
index_r[3907] = 243;
index_r[3908] = 243;
index_r[3909] = 243;
index_r[3910] = 243;
index_r[3911] = 243;
index_r[3912] = 243;
index_r[3913] = 243;
index_r[3914] = 243;
index_r[3915] = 243;
index_r[3916] = 243;
index_r[3917] = 243;
index_r[3918] = 244;
index_r[3919] = 244;
index_r[3920] = 244;
index_r[3921] = 244;
index_r[3922] = 244;
index_r[3923] = 244;
index_r[3924] = 244;
index_r[3925] = 244;
index_r[3926] = 244;
index_r[3927] = 244;
index_r[3928] = 244;
index_r[3929] = 244;
index_r[3930] = 244;
index_r[3931] = 244;
index_r[3932] = 244;
index_r[3933] = 244;
index_r[3934] = 244;
index_r[3935] = 244;
index_r[3936] = 244;
index_r[3937] = 245;
index_r[3938] = 245;
index_r[3939] = 245;
index_r[3940] = 245;
index_r[3941] = 245;
index_r[3942] = 245;
index_r[3943] = 245;
index_r[3944] = 245;
index_r[3945] = 245;
index_r[3946] = 245;
index_r[3947] = 245;
index_r[3948] = 245;
index_r[3949] = 245;
index_r[3950] = 245;
index_r[3951] = 245;
index_r[3952] = 245;
index_r[3953] = 245;
index_r[3954] = 245;
index_r[3955] = 245;
index_r[3956] = 245;
index_r[3957] = 245;
index_r[3958] = 245;
index_r[3959] = 246;
index_r[3960] = 246;
index_r[3961] = 246;
index_r[3962] = 246;
index_r[3963] = 246;
index_r[3964] = 246;
index_r[3965] = 246;
index_r[3966] = 246;
index_r[3967] = 246;
index_r[3968] = 246;
index_r[3969] = 246;
index_r[3970] = 246;
index_r[3971] = 247;
index_r[3972] = 247;
index_r[3973] = 247;
index_r[3974] = 247;
index_r[3975] = 247;
index_r[3976] = 247;
index_r[3977] = 247;
index_r[3978] = 247;
index_r[3979] = 247;
index_r[3980] = 247;
index_r[3981] = 247;
index_r[3982] = 247;
index_r[3983] = 247;
index_r[3984] = 247;
index_r[3985] = 247;
index_r[3986] = 248;
index_r[3987] = 248;
index_r[3988] = 248;
index_r[3989] = 248;
index_r[3990] = 248;
index_r[3991] = 248;
index_r[3992] = 248;
index_r[3993] = 248;
index_r[3994] = 248;
index_r[3995] = 248;
index_r[3996] = 248;
index_r[3997] = 248;
index_r[3998] = 248;
index_r[3999] = 248;
index_r[4000] = 248;
index_r[4001] = 249;
index_r[4002] = 249;
index_r[4003] = 249;
index_r[4004] = 249;
index_r[4005] = 249;
index_r[4006] = 249;
index_r[4007] = 249;
index_r[4008] = 250;
index_r[4009] = 250;
index_r[4010] = 250;
index_r[4011] = 250;
index_r[4012] = 250;
index_r[4013] = 250;
index_r[4014] = 250;
index_r[4015] = 250;
index_r[4016] = 250;
index_r[4017] = 250;
index_r[4018] = 250;
index_r[4019] = 250;
index_r[4020] = 250;
index_r[4021] = 250;
index_r[4022] = 250;
index_r[4023] = 250;
index_r[4024] = 250;
index_r[4025] = 250;
index_r[4026] = 250;
index_r[4027] = 250;
index_r[4028] = 250;
index_r[4029] = 250;
index_r[4030] = 251;
index_r[4031] = 251;
index_r[4032] = 251;
index_r[4033] = 251;
index_r[4034] = 251;
index_r[4035] = 251;
index_r[4036] = 251;
index_r[4037] = 251;
index_r[4038] = 251;
index_r[4039] = 251;
index_r[4040] = 251;
index_r[4041] = 251;
index_r[4042] = 251;
index_r[4043] = 251;
index_r[4044] = 251;
index_r[4045] = 252;
index_r[4046] = 252;
index_r[4047] = 252;
index_r[4048] = 252;
index_r[4049] = 252;
index_r[4050] = 252;
index_r[4051] = 252;
index_r[4052] = 252;
index_r[4053] = 252;
index_r[4054] = 252;
index_r[4055] = 252;
index_r[4056] = 252;
index_r[4057] = 252;
index_r[4058] = 252;
index_r[4059] = 252;
index_r[4060] = 252;
index_r[4061] = 252;
index_r[4062] = 253;
index_r[4063] = 253;
index_r[4064] = 253;
index_r[4065] = 253;
index_r[4066] = 253;
index_r[4067] = 253;
index_r[4068] = 253;
index_r[4069] = 253;
index_r[4070] = 253;
index_r[4071] = 253;
index_r[4072] = 253;
index_r[4073] = 253;
index_r[4074] = 253;
index_r[4075] = 253;
index_r[4076] = 253;
index_r[4077] = 253;
index_r[4078] = 253;
index_r[4079] = 253;
index_r[4080] = 253;
index_r[4081] = 254;
index_r[4082] = 254;
index_r[4083] = 254;
index_r[4084] = 254;
index_r[4085] = 254;
index_r[4086] = 254;
index_r[4087] = 254;
index_r[4088] = 254;
index_r[4089] = 254;
index_r[4090] = 254;
index_r[4091] = 254;
index_r[4092] = 254;
index_r[4093] = 254;
index_r[4094] = 254;
index_r[4095] = 254;
index_r[4096] = 254;
index_r[4097] = 254;
index_r[4098] = 254;
index_r[4099] = 254;
index_r[4100] = 255;
index_r[4101] = 255;
index_r[4102] = 255;
index_r[4103] = 255;
index_r[4104] = 255;
index_r[4105] = 255;
index_r[4106] = 255;
index_r[4107] = 255;
index_r[4108] = 255;
index_r[4109] = 255;
index_r[4110] = 255;
index_r[4111] = 256;
index_r[4112] = 256;
index_r[4113] = 256;
index_r[4114] = 256;
index_r[4115] = 256;
index_r[4116] = 256;
index_r[4117] = 256;
index_r[4118] = 256;
index_r[4119] = 256;
index_r[4120] = 256;
index_r[4121] = 256;
index_r[4122] = 256;
index_r[4123] = 256;
index_r[4124] = 256;
index_r[4125] = 256;
index_r[4126] = 256;
index_r[4127] = 256;
index_r[4128] = 256;
index_r[4129] = 256;
index_r[4130] = 256;
index_r[4131] = 256;
index_r[4132] = 257;
index_r[4133] = 257;
index_r[4134] = 257;
index_r[4135] = 257;
index_r[4136] = 257;
index_r[4137] = 257;
index_r[4138] = 257;
index_r[4139] = 257;
index_r[4140] = 257;
index_r[4141] = 257;
index_r[4142] = 257;
index_r[4143] = 257;
index_r[4144] = 257;
index_r[4145] = 257;
index_r[4146] = 257;
index_r[4147] = 257;
index_r[4148] = 257;
index_r[4149] = 257;
index_r[4150] = 257;
index_r[4151] = 258;
index_r[4152] = 258;
index_r[4153] = 258;
index_r[4154] = 258;
index_r[4155] = 258;
index_r[4156] = 258;
index_r[4157] = 258;
index_r[4158] = 258;
index_r[4159] = 258;
index_r[4160] = 258;
index_r[4161] = 258;
index_r[4162] = 258;
index_r[4163] = 258;
index_r[4164] = 258;
index_r[4165] = 259;
index_r[4166] = 259;
index_r[4167] = 259;
index_r[4168] = 259;
index_r[4169] = 259;
index_r[4170] = 259;
index_r[4171] = 259;
index_r[4172] = 259;
index_r[4173] = 259;
index_r[4174] = 259;
index_r[4175] = 259;
index_r[4176] = 259;
index_r[4177] = 259;
index_r[4178] = 259;
index_r[4179] = 259;
index_r[4180] = 259;
index_r[4181] = 259;
index_r[4182] = 260;
index_r[4183] = 260;
index_r[4184] = 260;
index_r[4185] = 260;
index_r[4186] = 260;
index_r[4187] = 260;
index_r[4188] = 260;
index_r[4189] = 261;
index_r[4190] = 261;
index_r[4191] = 261;
index_r[4192] = 261;
index_r[4193] = 261;
index_r[4194] = 261;
index_r[4195] = 261;
index_r[4196] = 261;
index_r[4197] = 261;
index_r[4198] = 261;
index_r[4199] = 261;
index_r[4200] = 261;
index_r[4201] = 262;
index_r[4202] = 262;
index_r[4203] = 262;
index_r[4204] = 262;
index_r[4205] = 262;
index_r[4206] = 262;
index_r[4207] = 262;
index_r[4208] = 262;
index_r[4209] = 262;
index_r[4210] = 262;
index_r[4211] = 262;
index_r[4212] = 262;
index_r[4213] = 262;
index_r[4214] = 262;
index_r[4215] = 262;
index_r[4216] = 262;
index_r[4217] = 262;
index_r[4218] = 262;
index_r[4219] = 262;
index_r[4220] = 262;
index_r[4221] = 263;
index_r[4222] = 263;
index_r[4223] = 263;
index_r[4224] = 263;
index_r[4225] = 263;
index_r[4226] = 263;
index_r[4227] = 263;
index_r[4228] = 263;
index_r[4229] = 263;
index_r[4230] = 263;
index_r[4231] = 263;
index_r[4232] = 263;
index_r[4233] = 263;
index_r[4234] = 263;
index_r[4235] = 263;
index_r[4236] = 263;
index_r[4237] = 263;
index_r[4238] = 263;
index_r[4239] = 263;
index_r[4240] = 264;
index_r[4241] = 264;
index_r[4242] = 264;
index_r[4243] = 264;
index_r[4244] = 264;
index_r[4245] = 264;
index_r[4246] = 264;
index_r[4247] = 265;
index_r[4248] = 265;
index_r[4249] = 265;
index_r[4250] = 265;
index_r[4251] = 265;
index_r[4252] = 265;
index_r[4253] = 265;
index_r[4254] = 265;
index_r[4255] = 265;
index_r[4256] = 265;
index_r[4257] = 265;
index_r[4258] = 265;
index_r[4259] = 265;
index_r[4260] = 265;
index_r[4261] = 265;
index_r[4262] = 266;
index_r[4263] = 266;
index_r[4264] = 266;
index_r[4265] = 266;
index_r[4266] = 266;
index_r[4267] = 266;
index_r[4268] = 266;
index_r[4269] = 266;
index_r[4270] = 266;
index_r[4271] = 266;
index_r[4272] = 266;
index_r[4273] = 266;
index_r[4274] = 266;
index_r[4275] = 266;
index_r[4276] = 266;
index_r[4277] = 266;
index_r[4278] = 266;
index_r[4279] = 266;
index_r[4280] = 266;
index_r[4281] = 267;
index_r[4282] = 267;
index_r[4283] = 267;
index_r[4284] = 267;
index_r[4285] = 267;
index_r[4286] = 267;
index_r[4287] = 267;
index_r[4288] = 268;
index_r[4289] = 268;
index_r[4290] = 268;
index_r[4291] = 268;
index_r[4292] = 268;
index_r[4293] = 268;
index_r[4294] = 268;
index_r[4295] = 268;
index_r[4296] = 268;
index_r[4297] = 268;
index_r[4298] = 268;
index_r[4299] = 268;
index_r[4300] = 268;
index_r[4301] = 268;
index_r[4302] = 268;
index_r[4303] = 268;
index_r[4304] = 268;
index_r[4305] = 268;
index_r[4306] = 268;
index_r[4307] = 269;
index_r[4308] = 269;
index_r[4309] = 269;
index_r[4310] = 269;
index_r[4311] = 269;
index_r[4312] = 269;
index_r[4313] = 269;
index_r[4314] = 269;
index_r[4315] = 269;
index_r[4316] = 269;
index_r[4317] = 269;
index_r[4318] = 269;
index_r[4319] = 269;
index_r[4320] = 269;
index_r[4321] = 269;
index_r[4322] = 269;
index_r[4323] = 269;
index_r[4324] = 269;
index_r[4325] = 269;
index_r[4326] = 269;
index_r[4327] = 270;
index_r[4328] = 270;
index_r[4329] = 270;
index_r[4330] = 270;
index_r[4331] = 270;
index_r[4332] = 270;
index_r[4333] = 270;
index_r[4334] = 270;
index_r[4335] = 270;
index_r[4336] = 270;
index_r[4337] = 270;
index_r[4338] = 270;
index_r[4339] = 270;
index_r[4340] = 270;
index_r[4341] = 270;
index_r[4342] = 271;
index_r[4343] = 271;
index_r[4344] = 271;
index_r[4345] = 271;
index_r[4346] = 271;
index_r[4347] = 271;
index_r[4348] = 271;
index_r[4349] = 271;
index_r[4350] = 271;
index_r[4351] = 271;
index_r[4352] = 271;
index_r[4353] = 271;
index_r[4354] = 271;
index_r[4355] = 271;
index_r[4356] = 271;
index_r[4357] = 272;
index_r[4358] = 272;
index_r[4359] = 272;
index_r[4360] = 272;
index_r[4361] = 272;
index_r[4362] = 272;
index_r[4363] = 272;
index_r[4364] = 273;
index_r[4365] = 273;
index_r[4366] = 273;
index_r[4367] = 273;
index_r[4368] = 273;
index_r[4369] = 273;
index_r[4370] = 273;
index_r[4371] = 273;
index_r[4372] = 273;
index_r[4373] = 273;
index_r[4374] = 273;
index_r[4375] = 273;
index_r[4376] = 273;
index_r[4377] = 273;
index_r[4378] = 273;
index_r[4379] = 273;
index_r[4380] = 273;
index_r[4381] = 274;
index_r[4382] = 274;
index_r[4383] = 274;
index_r[4384] = 274;
index_r[4385] = 274;
index_r[4386] = 274;
index_r[4387] = 274;
index_r[4388] = 274;
index_r[4389] = 274;
index_r[4390] = 274;
index_r[4391] = 274;
index_r[4392] = 274;
index_r[4393] = 274;
index_r[4394] = 274;
index_r[4395] = 274;
index_r[4396] = 275;
index_r[4397] = 275;
index_r[4398] = 275;
index_r[4399] = 275;
index_r[4400] = 275;
index_r[4401] = 275;
index_r[4402] = 275;
index_r[4403] = 275;
index_r[4404] = 275;
index_r[4405] = 275;
index_r[4406] = 275;
index_r[4407] = 275;
index_r[4408] = 275;
index_r[4409] = 275;
index_r[4410] = 275;
index_r[4411] = 275;
index_r[4412] = 275;
index_r[4413] = 275;
index_r[4414] = 275;
index_r[4415] = 275;
index_r[4416] = 275;
index_r[4417] = 275;
index_r[4418] = 275;
index_r[4419] = 275;
index_r[4420] = 276;
index_r[4421] = 276;
index_r[4422] = 276;
index_r[4423] = 276;
index_r[4424] = 276;
index_r[4425] = 276;
index_r[4426] = 276;
index_r[4427] = 276;
index_r[4428] = 276;
index_r[4429] = 276;
index_r[4430] = 276;
index_r[4431] = 277;
index_r[4432] = 277;
index_r[4433] = 277;
index_r[4434] = 277;
index_r[4435] = 277;
index_r[4436] = 277;
index_r[4437] = 277;
index_r[4438] = 277;
index_r[4439] = 277;
index_r[4440] = 277;
index_r[4441] = 278;
index_r[4442] = 278;
index_r[4443] = 278;
index_r[4444] = 278;
index_r[4445] = 278;
index_r[4446] = 278;
index_r[4447] = 278;
index_r[4448] = 278;
index_r[4449] = 278;
index_r[4450] = 278;
index_r[4451] = 278;
index_r[4452] = 278;
index_r[4453] = 278;
index_r[4454] = 278;
index_r[4455] = 278;
index_r[4456] = 278;
index_r[4457] = 278;
index_r[4458] = 278;
index_r[4459] = 278;
index_r[4460] = 278;
index_r[4461] = 278;
index_r[4462] = 278;
index_r[4463] = 278;
index_r[4464] = 278;
index_r[4465] = 279;
index_r[4466] = 279;
index_r[4467] = 279;
index_r[4468] = 279;
index_r[4469] = 279;
index_r[4470] = 279;
index_r[4471] = 279;
index_r[4472] = 279;
index_r[4473] = 279;
index_r[4474] = 279;
index_r[4475] = 279;
index_r[4476] = 279;
index_r[4477] = 279;
index_r[4478] = 279;
index_r[4479] = 279;
index_r[4480] = 279;
index_r[4481] = 280;
index_r[4482] = 280;
index_r[4483] = 280;
index_r[4484] = 280;
index_r[4485] = 280;
index_r[4486] = 280;
index_r[4487] = 280;
index_r[4488] = 280;
index_r[4489] = 280;
index_r[4490] = 280;
index_r[4491] = 280;
index_r[4492] = 280;
index_r[4493] = 280;
index_r[4494] = 280;
index_r[4495] = 280;
index_r[4496] = 280;
index_r[4497] = 280;
index_r[4498] = 280;
index_r[4499] = 280;
index_r[4500] = 280;
index_r[4501] = 280;
index_r[4502] = 280;
index_r[4503] = 280;
index_r[4504] = 280;
index_r[4505] = 281;
index_r[4506] = 281;
index_r[4507] = 281;
index_r[4508] = 281;
index_r[4509] = 281;
index_r[4510] = 281;
index_r[4511] = 281;
index_r[4512] = 281;
index_r[4513] = 281;
index_r[4514] = 281;
index_r[4515] = 282;
index_r[4516] = 282;
index_r[4517] = 282;
index_r[4518] = 282;
index_r[4519] = 282;
index_r[4520] = 282;
index_r[4521] = 282;
index_r[4522] = 282;
index_r[4523] = 282;
index_r[4524] = 282;
index_r[4525] = 282;
index_r[4526] = 282;
index_r[4527] = 282;
index_r[4528] = 282;
index_r[4529] = 282;
index_r[4530] = 282;
index_r[4531] = 282;
index_r[4532] = 282;
index_r[4533] = 282;
index_r[4534] = 282;
index_r[4535] = 282;
index_r[4536] = 282;
index_r[4537] = 283;
index_r[4538] = 283;
index_r[4539] = 283;
index_r[4540] = 283;
index_r[4541] = 283;
index_r[4542] = 283;
index_r[4543] = 283;
index_r[4544] = 283;
index_r[4545] = 283;
index_r[4546] = 283;
index_r[4547] = 283;
index_r[4548] = 283;
index_r[4549] = 283;
index_r[4550] = 283;
index_r[4551] = 284;
index_r[4552] = 284;
index_r[4553] = 284;
index_r[4554] = 284;
index_r[4555] = 284;
index_r[4556] = 284;
index_r[4557] = 284;
index_r[4558] = 284;
index_r[4559] = 284;
index_r[4560] = 284;
index_r[4561] = 285;
index_r[4562] = 285;
index_r[4563] = 285;
index_r[4564] = 285;
index_r[4565] = 285;
index_r[4566] = 285;
index_r[4567] = 285;
index_r[4568] = 285;
index_r[4569] = 285;
index_r[4570] = 285;
index_r[4571] = 285;
index_r[4572] = 286;
index_r[4573] = 286;
index_r[4574] = 286;
index_r[4575] = 286;
index_r[4576] = 286;
index_r[4577] = 286;
index_r[4578] = 286;
index_r[4579] = 286;
index_r[4580] = 286;
index_r[4581] = 286;
index_r[4582] = 286;
index_r[4583] = 286;
index_r[4584] = 286;
index_r[4585] = 286;
index_r[4586] = 286;
index_r[4587] = 287;
index_r[4588] = 287;
index_r[4589] = 287;
index_r[4590] = 287;
index_r[4591] = 287;
index_r[4592] = 287;
index_r[4593] = 287;
index_r[4594] = 287;
index_r[4595] = 287;
index_r[4596] = 287;
index_r[4597] = 287;
index_r[4598] = 287;
index_r[4599] = 287;
index_r[4600] = 287;
index_r[4601] = 287;
index_r[4602] = 287;
index_r[4603] = 288;
index_r[4604] = 288;
index_r[4605] = 288;
index_r[4606] = 288;
index_r[4607] = 288;
index_r[4608] = 288;
index_r[4609] = 288;
index_r[4610] = 288;
index_r[4611] = 288;
index_r[4612] = 288;
index_r[4613] = 289;
index_r[4614] = 289;
index_r[4615] = 289;
index_r[4616] = 289;
index_r[4617] = 289;
index_r[4618] = 289;
index_r[4619] = 289;
index_r[4620] = 289;
index_r[4621] = 289;
index_r[4622] = 289;
index_r[4623] = 289;
index_r[4624] = 289;
index_r[4625] = 289;
index_r[4626] = 289;
index_r[4627] = 289;
index_r[4628] = 290;
index_r[4629] = 290;
index_r[4630] = 290;
index_r[4631] = 290;
index_r[4632] = 290;
index_r[4633] = 290;
index_r[4634] = 290;
index_r[4635] = 290;
index_r[4636] = 290;
index_r[4637] = 290;
index_r[4638] = 290;
index_r[4639] = 290;
index_r[4640] = 290;
index_r[4641] = 290;
index_r[4642] = 290;
index_r[4643] = 290;
index_r[4644] = 290;
index_r[4645] = 290;
index_r[4646] = 290;
index_r[4647] = 291;
index_r[4648] = 291;
index_r[4649] = 291;
index_r[4650] = 291;
index_r[4651] = 291;
index_r[4652] = 291;
index_r[4653] = 291;
index_r[4654] = 291;
index_r[4655] = 291;
index_r[4656] = 291;
index_r[4657] = 291;
index_r[4658] = 292;
index_r[4659] = 292;
index_r[4660] = 292;
index_r[4661] = 292;
index_r[4662] = 292;
index_r[4663] = 292;
index_r[4664] = 292;
index_r[4665] = 292;
index_r[4666] = 292;
index_r[4667] = 292;
index_r[4668] = 292;
index_r[4669] = 292;
index_r[4670] = 292;
index_r[4671] = 292;
index_r[4672] = 292;
index_r[4673] = 292;
index_r[4674] = 292;
index_r[4675] = 292;
index_r[4676] = 292;
index_r[4677] = 292;
index_r[4678] = 292;
index_r[4679] = 293;
index_r[4680] = 293;
index_r[4681] = 293;
index_r[4682] = 293;
index_r[4683] = 293;
index_r[4684] = 293;
index_r[4685] = 293;
index_r[4686] = 293;
index_r[4687] = 293;
index_r[4688] = 293;
index_r[4689] = 293;
index_r[4690] = 293;
index_r[4691] = 293;
index_r[4692] = 293;
index_r[4693] = 293;
index_r[4694] = 293;
index_r[4695] = 293;
index_r[4696] = 293;
index_r[4697] = 293;
index_r[4698] = 293;
index_r[4699] = 293;
index_r[4700] = 293;
index_r[4701] = 294;
index_r[4702] = 294;
index_r[4703] = 294;
index_r[4704] = 294;
index_r[4705] = 294;
index_r[4706] = 294;
index_r[4707] = 294;
index_r[4708] = 294;
index_r[4709] = 294;
index_r[4710] = 294;
index_r[4711] = 294;
index_r[4712] = 294;
index_r[4713] = 294;
index_r[4714] = 294;
index_r[4715] = 294;
index_r[4716] = 294;
index_r[4717] = 294;
index_r[4718] = 294;
index_r[4719] = 294;
index_r[4720] = 294;
index_r[4721] = 294;
index_r[4722] = 294;
index_r[4723] = 295;
index_r[4724] = 295;
index_r[4725] = 295;
index_r[4726] = 295;
index_r[4727] = 295;
index_r[4728] = 295;
index_r[4729] = 295;
index_r[4730] = 295;
index_r[4731] = 295;
index_r[4732] = 295;
index_r[4733] = 295;
index_r[4734] = 295;
index_r[4735] = 295;
index_r[4736] = 295;
index_r[4737] = 295;
index_r[4738] = 295;
index_r[4739] = 295;
index_r[4740] = 295;
index_r[4741] = 295;
index_r[4742] = 295;
index_r[4743] = 296;
index_r[4744] = 296;
index_r[4745] = 296;
index_r[4746] = 296;
index_r[4747] = 296;
index_r[4748] = 296;
index_r[4749] = 296;
index_r[4750] = 296;
index_r[4751] = 296;
index_r[4752] = 296;
index_r[4753] = 296;
index_r[4754] = 296;
index_r[4755] = 296;
index_r[4756] = 296;
index_r[4757] = 296;
index_r[4758] = 296;
index_r[4759] = 296;
index_r[4760] = 296;
index_r[4761] = 296;
index_r[4762] = 296;
index_r[4763] = 296;
index_r[4764] = 296;
index_r[4765] = 296;
index_r[4766] = 296;
index_r[4767] = 297;
index_r[4768] = 297;
index_r[4769] = 297;
index_r[4770] = 297;
index_r[4771] = 297;
index_r[4772] = 297;
index_r[4773] = 297;
index_r[4774] = 297;
index_r[4775] = 297;
index_r[4776] = 297;
index_r[4777] = 297;
index_r[4778] = 297;
index_r[4779] = 297;
index_r[4780] = 297;
index_r[4781] = 297;
index_r[4782] = 297;
index_r[4783] = 297;
index_r[4784] = 298;
index_r[4785] = 298;
index_r[4786] = 298;
index_r[4787] = 298;
index_r[4788] = 298;
index_r[4789] = 298;
index_r[4790] = 298;
index_r[4791] = 298;
index_r[4792] = 298;
index_r[4793] = 298;
index_r[4794] = 298;
index_r[4795] = 298;
index_r[4796] = 298;
index_r[4797] = 298;
index_r[4798] = 298;
index_r[4799] = 298;
index_r[4800] = 298;
index_r[4801] = 298;
index_r[4802] = 298;
index_r[4803] = 299;
index_r[4804] = 299;
index_r[4805] = 299;
index_r[4806] = 299;
index_r[4807] = 299;
index_r[4808] = 299;
index_r[4809] = 299;
index_r[4810] = 299;
index_r[4811] = 299;
index_r[4812] = 299;
index_r[4813] = 299;
index_r[4814] = 299;
index_r[4815] = 299;
index_r[4816] = 299;
index_r[4817] = 299;
index_r[4818] = 299;
index_r[4819] = 299;
index_r[4820] = 299;
index_r[4821] = 299;
index_r[4822] = 300;
index_r[4823] = 300;
index_r[4824] = 300;
index_r[4825] = 300;
index_r[4826] = 300;
index_r[4827] = 300;
index_r[4828] = 300;
index_r[4829] = 300;
index_r[4830] = 300;
index_r[4831] = 300;
index_r[4832] = 300;
index_r[4833] = 300;
index_r[4834] = 300;
index_r[4835] = 300;
index_r[4836] = 300;
index_r[4837] = 300;
index_r[4838] = 300;
index_r[4839] = 301;
index_r[4840] = 301;
index_r[4841] = 301;
index_r[4842] = 301;
index_r[4843] = 301;
index_r[4844] = 301;
index_r[4845] = 301;
index_r[4846] = 301;
index_r[4847] = 301;
index_r[4848] = 301;
index_r[4849] = 301;
index_r[4850] = 301;
index_r[4851] = 301;
index_r[4852] = 301;
index_r[4853] = 301;
index_r[4854] = 301;
index_r[4855] = 302;
index_r[4856] = 302;
index_r[4857] = 302;
index_r[4858] = 302;
index_r[4859] = 302;
index_r[4860] = 302;
index_r[4861] = 302;
index_r[4862] = 302;
index_r[4863] = 302;
index_r[4864] = 302;
index_r[4865] = 302;
index_r[4866] = 302;
index_r[4867] = 302;
index_r[4868] = 302;
index_r[4869] = 303;
index_r[4870] = 303;
index_r[4871] = 303;
index_r[4872] = 303;
index_r[4873] = 303;
index_r[4874] = 303;
index_r[4875] = 303;
index_r[4876] = 303;
index_r[4877] = 303;
index_r[4878] = 303;
index_r[4879] = 303;
index_r[4880] = 303;
index_r[4881] = 303;
index_r[4882] = 303;
index_r[4883] = 304;
index_r[4884] = 304;
index_r[4885] = 304;
index_r[4886] = 304;
index_r[4887] = 304;
index_r[4888] = 304;
index_r[4889] = 304;
index_r[4890] = 304;
index_r[4891] = 304;
index_r[4892] = 304;
index_r[4893] = 304;
index_r[4894] = 304;
index_r[4895] = 305;
index_r[4896] = 305;
index_r[4897] = 305;
index_r[4898] = 305;
index_r[4899] = 305;
index_r[4900] = 305;
index_r[4901] = 305;
index_r[4902] = 305;
index_r[4903] = 305;
index_r[4904] = 305;
index_r[4905] = 305;
index_r[4906] = 305;
index_r[4907] = 305;
index_r[4908] = 305;
index_r[4909] = 305;
index_r[4910] = 305;
index_r[4911] = 305;
index_r[4912] = 305;
index_r[4913] = 305;
index_r[4914] = 306;
index_r[4915] = 306;
index_r[4916] = 306;
index_r[4917] = 306;
index_r[4918] = 306;
index_r[4919] = 306;
index_r[4920] = 306;
index_r[4921] = 306;
index_r[4922] = 306;
index_r[4923] = 306;
index_r[4924] = 306;
index_r[4925] = 306;
index_r[4926] = 306;
index_r[4927] = 306;
index_r[4928] = 306;
index_r[4929] = 306;
index_r[4930] = 306;
index_r[4931] = 306;
index_r[4932] = 306;
index_r[4933] = 307;
index_r[4934] = 307;
index_r[4935] = 307;
index_r[4936] = 307;
index_r[4937] = 307;
index_r[4938] = 307;
index_r[4939] = 307;
index_r[4940] = 307;
index_r[4941] = 307;
index_r[4942] = 307;
index_r[4943] = 307;
index_r[4944] = 307;
index_r[4945] = 307;
index_r[4946] = 307;
index_r[4947] = 307;
index_r[4948] = 307;
index_r[4949] = 307;
index_r[4950] = 308;
index_r[4951] = 308;
index_r[4952] = 308;
index_r[4953] = 308;
index_r[4954] = 308;
index_r[4955] = 308;
index_r[4956] = 308;
index_r[4957] = 308;
index_r[4958] = 308;
index_r[4959] = 308;
index_r[4960] = 308;
index_r[4961] = 308;
index_r[4962] = 308;
index_r[4963] = 308;
index_r[4964] = 308;
index_r[4965] = 308;
index_r[4966] = 308;
index_r[4967] = 308;
index_r[4968] = 308;
index_r[4969] = 308;
index_r[4970] = 308;
index_r[4971] = 308;
index_r[4972] = 308;
index_r[4973] = 308;
index_r[4974] = 309;
index_r[4975] = 309;
index_r[4976] = 309;
index_r[4977] = 309;
index_r[4978] = 309;
index_r[4979] = 309;
index_r[4980] = 309;
index_r[4981] = 309;
index_r[4982] = 309;
index_r[4983] = 309;
index_r[4984] = 309;
index_r[4985] = 309;
index_r[4986] = 309;
index_r[4987] = 309;
index_r[4988] = 309;
index_r[4989] = 309;
index_r[4990] = 309;
index_r[4991] = 309;
index_r[4992] = 309;
index_r[4993] = 310;
index_r[4994] = 310;
index_r[4995] = 310;
index_r[4996] = 310;
index_r[4997] = 310;
index_r[4998] = 310;
index_r[4999] = 310;
index_r[5000] = 310;
index_r[5001] = 310;
index_r[5002] = 310;
index_r[5003] = 310;
index_r[5004] = 311;
index_r[5005] = 311;
index_r[5006] = 311;
index_r[5007] = 311;
index_r[5008] = 311;
index_r[5009] = 311;
index_r[5010] = 311;
index_r[5011] = 311;
index_r[5012] = 311;
index_r[5013] = 311;
index_r[5014] = 312;
index_r[5015] = 312;
index_r[5016] = 312;
index_r[5017] = 312;
index_r[5018] = 312;
index_r[5019] = 312;
index_r[5020] = 312;
index_r[5021] = 312;
index_r[5022] = 312;
index_r[5023] = 312;
index_r[5024] = 312;
index_r[5025] = 312;
index_r[5026] = 312;
index_r[5027] = 312;
index_r[5028] = 312;
index_r[5029] = 312;
index_r[5030] = 312;
index_r[5031] = 313;
index_r[5032] = 313;
index_r[5033] = 313;
index_r[5034] = 313;
index_r[5035] = 313;
index_r[5036] = 313;
index_r[5037] = 313;
index_r[5038] = 313;
index_r[5039] = 313;
index_r[5040] = 313;
index_r[5041] = 313;
index_r[5042] = 313;
index_r[5043] = 313;
index_r[5044] = 313;
index_r[5045] = 313;
index_r[5046] = 313;
index_r[5047] = 313;
index_r[5048] = 314;
index_r[5049] = 314;
index_r[5050] = 314;
index_r[5051] = 314;
index_r[5052] = 314;
index_r[5053] = 314;
index_r[5054] = 314;
index_r[5055] = 314;
index_r[5056] = 314;
index_r[5057] = 314;
index_r[5058] = 315;
index_r[5059] = 315;
index_r[5060] = 315;
index_r[5061] = 315;
index_r[5062] = 315;
index_r[5063] = 315;
index_r[5064] = 315;
index_r[5065] = 315;
index_r[5066] = 315;
index_r[5067] = 315;
index_r[5068] = 315;
index_r[5069] = 315;
index_r[5070] = 315;
index_r[5071] = 315;
index_r[5072] = 315;
index_r[5073] = 315;
index_r[5074] = 315;
index_r[5075] = 315;
index_r[5076] = 315;
index_r[5077] = 315;
index_r[5078] = 315;
index_r[5079] = 315;
index_r[5080] = 315;
index_r[5081] = 315;
index_r[5082] = 316;
index_r[5083] = 316;
index_r[5084] = 316;
index_r[5085] = 316;
index_r[5086] = 316;
index_r[5087] = 316;
index_r[5088] = 316;
index_r[5089] = 316;
index_r[5090] = 316;
index_r[5091] = 316;
index_r[5092] = 316;
index_r[5093] = 316;
index_r[5094] = 316;
index_r[5095] = 316;
index_r[5096] = 316;
index_r[5097] = 316;
index_r[5098] = 316;
index_r[5099] = 316;
index_r[5100] = 316;
index_r[5101] = 316;
index_r[5102] = 316;
index_r[5103] = 316;
index_r[5104] = 316;
index_r[5105] = 316;
index_r[5106] = 317;
index_r[5107] = 317;
index_r[5108] = 317;
index_r[5109] = 317;
index_r[5110] = 317;
index_r[5111] = 317;
index_r[5112] = 317;
index_r[5113] = 317;
index_r[5114] = 317;
index_r[5115] = 317;
index_r[5116] = 317;
index_r[5117] = 317;
index_r[5118] = 317;
index_r[5119] = 317;
index_r[5120] = 317;
index_r[5121] = 317;
index_r[5122] = 317;
index_r[5123] = 317;
index_r[5124] = 317;
index_r[5125] = 318;
index_r[5126] = 318;
index_r[5127] = 318;
index_r[5128] = 318;
index_r[5129] = 318;
index_r[5130] = 318;
index_r[5131] = 318;
index_r[5132] = 318;
index_r[5133] = 318;
index_r[5134] = 318;
index_r[5135] = 318;
index_r[5136] = 318;
index_r[5137] = 318;
index_r[5138] = 318;
index_r[5139] = 318;
index_r[5140] = 318;
index_r[5141] = 318;
index_r[5142] = 319;
index_r[5143] = 319;
index_r[5144] = 319;
index_r[5145] = 319;
index_r[5146] = 319;
index_r[5147] = 319;
index_r[5148] = 319;
index_r[5149] = 319;
index_r[5150] = 319;
index_r[5151] = 319;
index_r[5152] = 319;
index_r[5153] = 319;
index_r[5154] = 319;
index_r[5155] = 319;
index_r[5156] = 319;
index_r[5157] = 319;
index_r[5158] = 320;
index_r[5159] = 320;
index_r[5160] = 320;
index_r[5161] = 320;
index_r[5162] = 320;
index_r[5163] = 320;
index_r[5164] = 320;
index_r[5165] = 320;
index_r[5166] = 320;
index_r[5167] = 320;
index_r[5168] = 320;
index_r[5169] = 320;
index_r[5170] = 320;
index_r[5171] = 320;
index_r[5172] = 321;
index_r[5173] = 321;
index_r[5174] = 321;
index_r[5175] = 321;
index_r[5176] = 321;
index_r[5177] = 321;
index_r[5178] = 321;
index_r[5179] = 321;
index_r[5180] = 321;
index_r[5181] = 321;
index_r[5182] = 321;
index_r[5183] = 322;
index_r[5184] = 322;
index_r[5185] = 322;
index_r[5186] = 322;
index_r[5187] = 322;
index_r[5188] = 322;
index_r[5189] = 322;
index_r[5190] = 322;
index_r[5191] = 322;
index_r[5192] = 322;
index_r[5193] = 322;
index_r[5194] = 322;
index_r[5195] = 322;
index_r[5196] = 322;
index_r[5197] = 322;
index_r[5198] = 323;
index_r[5199] = 323;
index_r[5200] = 323;
index_r[5201] = 323;
index_r[5202] = 323;
index_r[5203] = 323;
index_r[5204] = 323;
index_r[5205] = 323;
index_r[5206] = 323;
index_r[5207] = 323;
index_r[5208] = 323;
index_r[5209] = 323;
index_r[5210] = 323;
index_r[5211] = 323;
index_r[5212] = 323;
index_r[5213] = 323;
index_r[5214] = 323;
index_r[5215] = 323;
index_r[5216] = 323;
index_r[5217] = 323;
index_r[5218] = 323;
index_r[5219] = 323;
index_r[5220] = 324;
index_r[5221] = 324;
index_r[5222] = 324;
index_r[5223] = 324;
index_r[5224] = 324;
index_r[5225] = 324;
index_r[5226] = 324;
index_r[5227] = 324;
index_r[5228] = 324;
index_r[5229] = 324;
index_r[5230] = 324;
index_r[5231] = 324;
index_r[5232] = 324;
index_r[5233] = 324;
index_r[5234] = 324;
index_r[5235] = 324;
index_r[5236] = 325;
index_r[5237] = 325;
index_r[5238] = 325;
index_r[5239] = 325;
index_r[5240] = 325;
index_r[5241] = 325;
index_r[5242] = 325;
index_r[5243] = 326;
index_r[5244] = 326;
index_r[5245] = 326;
index_r[5246] = 326;
index_r[5247] = 326;
index_r[5248] = 326;
index_r[5249] = 326;
index_r[5250] = 326;
index_r[5251] = 326;
index_r[5252] = 326;
index_r[5253] = 326;
index_r[5254] = 326;
index_r[5255] = 326;
index_r[5256] = 326;
index_r[5257] = 327;
index_r[5258] = 327;
index_r[5259] = 327;
index_r[5260] = 327;
index_r[5261] = 327;
index_r[5262] = 327;
index_r[5263] = 327;
index_r[5264] = 327;
index_r[5265] = 327;
index_r[5266] = 327;
index_r[5267] = 327;
index_r[5268] = 327;
index_r[5269] = 327;
index_r[5270] = 327;
index_r[5271] = 327;
index_r[5272] = 327;
index_r[5273] = 327;
index_r[5274] = 327;
index_r[5275] = 327;
index_r[5276] = 328;
index_r[5277] = 328;
index_r[5278] = 328;
index_r[5279] = 328;
index_r[5280] = 328;
index_r[5281] = 328;
index_r[5282] = 328;
index_r[5283] = 328;
index_r[5284] = 328;
index_r[5285] = 328;
index_r[5286] = 329;
index_r[5287] = 329;
index_r[5288] = 329;
index_r[5289] = 329;
index_r[5290] = 329;
index_r[5291] = 329;
index_r[5292] = 329;
index_r[5293] = 329;
index_r[5294] = 329;
index_r[5295] = 329;
index_r[5296] = 329;
index_r[5297] = 329;
index_r[5298] = 329;
index_r[5299] = 329;
index_r[5300] = 329;
index_r[5301] = 329;
index_r[5302] = 329;
index_r[5303] = 329;
index_r[5304] = 329;
index_r[5305] = 329;
index_r[5306] = 330;
index_r[5307] = 330;
index_r[5308] = 330;
index_r[5309] = 330;
index_r[5310] = 330;
index_r[5311] = 330;
index_r[5312] = 330;
index_r[5313] = 330;
index_r[5314] = 330;
index_r[5315] = 330;
index_r[5316] = 330;
index_r[5317] = 330;
index_r[5318] = 330;
index_r[5319] = 330;
index_r[5320] = 330;
index_r[5321] = 330;
index_r[5322] = 330;
index_r[5323] = 330;
index_r[5324] = 330;
index_r[5325] = 331;
index_r[5326] = 331;
index_r[5327] = 331;
index_r[5328] = 331;
index_r[5329] = 331;
index_r[5330] = 331;
index_r[5331] = 331;
index_r[5332] = 331;
index_r[5333] = 331;
index_r[5334] = 331;
index_r[5335] = 331;
index_r[5336] = 331;
index_r[5337] = 332;
index_r[5338] = 332;
index_r[5339] = 332;
index_r[5340] = 332;
index_r[5341] = 332;
index_r[5342] = 332;
index_r[5343] = 332;
index_r[5344] = 332;
index_r[5345] = 332;
index_r[5346] = 332;
index_r[5347] = 332;
index_r[5348] = 332;
index_r[5349] = 332;
index_r[5350] = 332;
index_r[5351] = 332;
index_r[5352] = 332;
index_r[5353] = 333;
index_r[5354] = 333;
index_r[5355] = 333;
index_r[5356] = 333;
index_r[5357] = 333;
index_r[5358] = 333;
index_r[5359] = 333;
index_r[5360] = 333;
index_r[5361] = 333;
index_r[5362] = 333;
index_r[5363] = 333;
index_r[5364] = 333;
index_r[5365] = 333;
index_r[5366] = 333;
index_r[5367] = 334;
index_r[5368] = 334;
index_r[5369] = 334;
index_r[5370] = 334;
index_r[5371] = 334;
index_r[5372] = 334;
index_r[5373] = 334;
index_r[5374] = 335;
index_r[5375] = 335;
index_r[5376] = 335;
index_r[5377] = 335;
index_r[5378] = 335;
index_r[5379] = 335;
index_r[5380] = 335;
index_r[5381] = 335;
index_r[5382] = 335;
index_r[5383] = 335;
index_r[5384] = 335;
index_r[5385] = 335;
index_r[5386] = 335;
index_r[5387] = 335;
index_r[5388] = 335;
index_r[5389] = 335;
index_r[5390] = 335;
index_r[5391] = 335;
index_r[5392] = 335;
index_r[5393] = 335;
index_r[5394] = 335;
index_r[5395] = 335;
index_r[5396] = 335;
index_r[5397] = 335;
index_r[5398] = 336;
index_r[5399] = 336;
index_r[5400] = 336;
index_r[5401] = 336;
index_r[5402] = 336;
index_r[5403] = 336;
index_r[5404] = 336;
index_r[5405] = 336;
index_r[5406] = 336;
index_r[5407] = 336;
index_r[5408] = 336;
index_r[5409] = 336;
index_r[5410] = 336;
index_r[5411] = 336;
index_r[5412] = 336;
index_r[5413] = 336;
index_r[5414] = 336;
index_r[5415] = 336;
index_r[5416] = 336;
index_r[5417] = 337;
index_r[5418] = 337;
index_r[5419] = 337;
index_r[5420] = 337;
index_r[5421] = 337;
index_r[5422] = 337;
index_r[5423] = 337;
index_r[5424] = 337;
index_r[5425] = 337;
index_r[5426] = 337;
index_r[5427] = 338;
index_r[5428] = 338;
index_r[5429] = 338;
index_r[5430] = 338;
index_r[5431] = 338;
index_r[5432] = 338;
index_r[5433] = 338;
index_r[5434] = 338;
index_r[5435] = 338;
index_r[5436] = 338;
index_r[5437] = 338;
index_r[5438] = 338;
index_r[5439] = 338;
index_r[5440] = 338;
index_r[5441] = 338;
index_r[5442] = 338;
index_r[5443] = 338;
index_r[5444] = 339;
index_r[5445] = 339;
index_r[5446] = 339;
index_r[5447] = 339;
index_r[5448] = 339;
index_r[5449] = 339;
index_r[5450] = 339;
index_r[5451] = 339;
index_r[5452] = 339;
index_r[5453] = 339;
index_r[5454] = 339;
index_r[5455] = 339;
index_r[5456] = 339;
index_r[5457] = 339;
index_r[5458] = 340;
index_r[5459] = 340;
index_r[5460] = 340;
index_r[5461] = 340;
index_r[5462] = 340;
index_r[5463] = 340;
index_r[5464] = 340;
index_r[5465] = 340;
index_r[5466] = 340;
index_r[5467] = 340;
index_r[5468] = 340;
index_r[5469] = 340;
index_r[5470] = 340;
index_r[5471] = 340;
index_r[5472] = 340;
index_r[5473] = 340;
index_r[5474] = 340;
index_r[5475] = 340;
index_r[5476] = 340;
index_r[5477] = 341;
index_r[5478] = 341;
index_r[5479] = 341;
index_r[5480] = 341;
index_r[5481] = 341;
index_r[5482] = 341;
index_r[5483] = 341;
index_r[5484] = 341;
index_r[5485] = 341;
index_r[5486] = 341;
index_r[5487] = 341;
index_r[5488] = 341;
index_r[5489] = 341;
index_r[5490] = 341;
index_r[5491] = 341;
index_r[5492] = 341;
index_r[5493] = 341;
index_r[5494] = 341;
index_r[5495] = 341;
index_r[5496] = 342;
index_r[5497] = 342;
index_r[5498] = 342;
index_r[5499] = 342;
index_r[5500] = 342;
index_r[5501] = 342;
index_r[5502] = 342;
index_r[5503] = 342;
index_r[5504] = 342;
index_r[5505] = 342;
index_r[5506] = 342;
index_r[5507] = 342;
index_r[5508] = 342;
index_r[5509] = 342;
index_r[5510] = 343;
index_r[5511] = 343;
index_r[5512] = 343;
index_r[5513] = 343;
index_r[5514] = 343;
index_r[5515] = 343;
index_r[5516] = 343;
index_r[5517] = 343;
index_r[5518] = 343;
index_r[5519] = 343;
index_r[5520] = 343;
index_r[5521] = 343;
index_r[5522] = 343;
index_r[5523] = 343;
index_r[5524] = 343;
index_r[5525] = 343;
index_r[5526] = 343;
index_r[5527] = 343;
index_r[5528] = 343;
index_r[5529] = 344;
index_r[5530] = 344;
index_r[5531] = 344;
index_r[5532] = 344;
index_r[5533] = 344;
index_r[5534] = 344;
index_r[5535] = 344;
index_r[5536] = 344;
index_r[5537] = 344;
index_r[5538] = 344;
index_r[5539] = 345;
index_r[5540] = 345;
index_r[5541] = 345;
index_r[5542] = 345;
index_r[5543] = 345;
index_r[5544] = 345;
index_r[5545] = 345;
index_r[5546] = 345;
index_r[5547] = 345;
index_r[5548] = 345;
index_r[5549] = 345;
index_r[5550] = 345;
index_r[5551] = 345;
index_r[5552] = 345;
index_r[5553] = 345;
index_r[5554] = 345;
index_r[5555] = 345;
index_r[5556] = 345;
index_r[5557] = 345;
index_r[5558] = 345;
index_r[5559] = 345;
index_r[5560] = 345;
index_r[5561] = 346;
index_r[5562] = 346;
index_r[5563] = 346;
index_r[5564] = 346;
index_r[5565] = 346;
index_r[5566] = 346;
index_r[5567] = 346;
index_r[5568] = 346;
index_r[5569] = 346;
index_r[5570] = 346;
index_r[5571] = 346;
index_r[5572] = 346;
index_r[5573] = 346;
index_r[5574] = 346;
index_r[5575] = 346;
index_r[5576] = 346;
index_r[5577] = 346;
index_r[5578] = 347;
index_r[5579] = 347;
index_r[5580] = 347;
index_r[5581] = 347;
index_r[5582] = 347;
index_r[5583] = 347;
index_r[5584] = 347;
index_r[5585] = 347;
index_r[5586] = 347;
index_r[5587] = 347;
index_r[5588] = 347;
index_r[5589] = 347;
index_r[5590] = 347;
index_r[5591] = 347;
index_r[5592] = 348;
index_r[5593] = 348;
index_r[5594] = 348;
index_r[5595] = 348;
index_r[5596] = 348;
index_r[5597] = 348;
index_r[5598] = 348;
index_r[5599] = 348;
index_r[5600] = 348;
index_r[5601] = 348;
index_r[5602] = 348;
index_r[5603] = 348;
index_r[5604] = 349;
index_r[5605] = 349;
index_r[5606] = 349;
index_r[5607] = 349;
index_r[5608] = 349;
index_r[5609] = 349;
index_r[5610] = 349;
index_r[5611] = 349;
index_r[5612] = 349;
index_r[5613] = 349;
index_r[5614] = 350;
index_r[5615] = 350;
index_r[5616] = 350;
index_r[5617] = 350;
index_r[5618] = 350;
index_r[5619] = 350;
index_r[5620] = 350;
index_r[5621] = 350;
index_r[5622] = 350;
index_r[5623] = 350;
index_r[5624] = 350;
index_r[5625] = 350;
index_r[5626] = 350;
index_r[5627] = 350;
index_r[5628] = 350;
index_r[5629] = 350;
index_r[5630] = 350;
index_r[5631] = 351;
index_r[5632] = 351;
index_r[5633] = 351;
index_r[5634] = 351;
index_r[5635] = 351;
index_r[5636] = 351;
index_r[5637] = 351;
index_r[5638] = 351;
index_r[5639] = 351;
index_r[5640] = 351;
index_r[5641] = 351;
index_r[5642] = 351;
index_r[5643] = 351;
index_r[5644] = 351;
index_r[5645] = 352;
index_r[5646] = 352;
index_r[5647] = 352;
index_r[5648] = 352;
index_r[5649] = 352;
index_r[5650] = 352;
index_r[5651] = 352;
index_r[5652] = 352;
index_r[5653] = 352;
index_r[5654] = 352;
index_r[5655] = 352;
index_r[5656] = 352;
index_r[5657] = 352;
index_r[5658] = 352;
index_r[5659] = 352;
index_r[5660] = 352;
index_r[5661] = 352;
index_r[5662] = 353;
index_r[5663] = 353;
index_r[5664] = 353;
index_r[5665] = 353;
index_r[5666] = 353;
index_r[5667] = 353;
index_r[5668] = 353;
index_r[5669] = 353;
index_r[5670] = 353;
index_r[5671] = 353;
index_r[5672] = 353;
index_r[5673] = 353;
index_r[5674] = 353;
index_r[5675] = 353;
index_r[5676] = 354;
index_r[5677] = 354;
index_r[5678] = 354;
index_r[5679] = 354;
index_r[5680] = 354;
index_r[5681] = 354;
index_r[5682] = 354;
index_r[5683] = 354;
index_r[5684] = 354;
index_r[5685] = 354;
index_r[5686] = 354;
index_r[5687] = 354;
index_r[5688] = 355;
index_r[5689] = 355;
index_r[5690] = 355;
index_r[5691] = 355;
index_r[5692] = 355;
index_r[5693] = 355;
index_r[5694] = 355;
index_r[5695] = 356;
index_r[5696] = 356;
index_r[5697] = 356;
index_r[5698] = 356;
index_r[5699] = 356;
index_r[5700] = 356;
index_r[5701] = 356;
index_r[5702] = 356;
index_r[5703] = 356;
index_r[5704] = 356;
index_r[5705] = 356;
index_r[5706] = 356;
index_r[5707] = 356;
index_r[5708] = 356;
index_r[5709] = 356;
index_r[5710] = 356;
index_r[5711] = 356;
index_r[5712] = 357;
index_r[5713] = 357;
index_r[5714] = 357;
index_r[5715] = 357;
index_r[5716] = 357;
index_r[5717] = 357;
index_r[5718] = 357;
index_r[5719] = 357;
index_r[5720] = 357;
index_r[5721] = 357;
index_r[5722] = 357;
index_r[5723] = 357;
index_r[5724] = 357;
index_r[5725] = 357;
index_r[5726] = 357;
index_r[5727] = 357;
index_r[5728] = 357;
index_r[5729] = 358;
index_r[5730] = 358;
index_r[5731] = 358;
index_r[5732] = 358;
index_r[5733] = 358;
index_r[5734] = 358;
index_r[5735] = 358;
index_r[5736] = 358;
index_r[5737] = 358;
index_r[5738] = 358;
index_r[5739] = 358;
index_r[5740] = 358;
index_r[5741] = 358;
index_r[5742] = 358;
index_r[5743] = 358;
index_r[5744] = 358;
index_r[5745] = 358;
index_r[5746] = 358;
index_r[5747] = 358;
index_r[5748] = 359;
index_r[5749] = 359;
index_r[5750] = 359;
index_r[5751] = 359;
index_r[5752] = 359;
index_r[5753] = 359;
index_r[5754] = 359;
index_r[5755] = 359;
index_r[5756] = 359;
index_r[5757] = 359;
index_r[5758] = 359;
index_r[5759] = 359;
index_r[5760] = 359;
index_r[5761] = 359;
index_r[5762] = 359;
index_r[5763] = 359;
index_r[5764] = 359;
index_r[5765] = 359;
index_r[5766] = 359;
index_r[5767] = 359;
index_r[5768] = 359;
index_r[5769] = 359;
index_r[5770] = 360;
index_r[5771] = 360;
index_r[5772] = 360;
index_r[5773] = 360;
index_r[5774] = 360;
index_r[5775] = 360;
index_r[5776] = 360;
index_r[5777] = 360;
index_r[5778] = 360;
index_r[5779] = 360;
index_r[5780] = 360;
index_r[5781] = 360;
index_r[5782] = 360;
index_r[5783] = 360;
index_r[5784] = 360;
index_r[5785] = 360;
index_r[5786] = 360;
index_r[5787] = 360;
index_r[5788] = 360;
index_r[5789] = 361;
index_r[5790] = 361;
index_r[5791] = 361;
index_r[5792] = 361;
index_r[5793] = 361;
index_r[5794] = 361;
index_r[5795] = 361;
index_r[5796] = 361;
index_r[5797] = 361;
index_r[5798] = 361;
index_r[5799] = 361;
index_r[5800] = 361;
index_r[5801] = 361;
index_r[5802] = 361;
index_r[5803] = 362;
index_r[5804] = 362;
index_r[5805] = 362;
index_r[5806] = 362;
index_r[5807] = 362;
index_r[5808] = 362;
index_r[5809] = 362;
index_r[5810] = 362;
index_r[5811] = 362;
index_r[5812] = 362;
index_r[5813] = 362;
index_r[5814] = 362;
index_r[5815] = 362;
index_r[5816] = 362;
index_r[5817] = 362;
index_r[5818] = 362;
index_r[5819] = 362;
index_r[5820] = 362;
index_r[5821] = 362;
index_r[5822] = 362;
index_r[5823] = 362;
index_r[5824] = 362;
index_r[5825] = 362;
index_r[5826] = 362;
index_r[5827] = 363;
index_r[5828] = 363;
index_r[5829] = 363;
index_r[5830] = 363;
index_r[5831] = 363;
index_r[5832] = 363;
index_r[5833] = 363;
index_r[5834] = 363;
index_r[5835] = 363;
index_r[5836] = 363;
index_r[5837] = 363;
index_r[5838] = 363;
index_r[5839] = 363;
index_r[5840] = 363;
index_r[5841] = 363;
index_r[5842] = 363;
index_r[5843] = 363;
index_r[5844] = 364;
index_r[5845] = 364;
index_r[5846] = 364;
index_r[5847] = 364;
index_r[5848] = 364;
index_r[5849] = 364;
index_r[5850] = 364;
index_r[5851] = 364;
index_r[5852] = 364;
index_r[5853] = 364;
index_r[5854] = 364;
index_r[5855] = 364;
index_r[5856] = 364;
index_r[5857] = 364;
index_r[5858] = 364;
index_r[5859] = 365;
index_r[5860] = 365;
index_r[5861] = 365;
index_r[5862] = 365;
index_r[5863] = 365;
index_r[5864] = 365;
index_r[5865] = 365;
index_r[5866] = 365;
index_r[5867] = 365;
index_r[5868] = 365;
index_r[5869] = 365;
index_r[5870] = 365;
index_r[5871] = 365;
index_r[5872] = 365;
index_r[5873] = 365;
index_r[5874] = 365;
index_r[5875] = 365;
index_r[5876] = 365;
index_r[5877] = 365;
index_r[5878] = 366;
index_r[5879] = 366;
index_r[5880] = 366;
index_r[5881] = 366;
index_r[5882] = 366;
index_r[5883] = 366;
index_r[5884] = 366;
index_r[5885] = 367;
index_r[5886] = 367;
index_r[5887] = 367;
index_r[5888] = 367;
index_r[5889] = 367;
index_r[5890] = 367;
index_r[5891] = 367;
index_r[5892] = 367;
index_r[5893] = 367;
index_r[5894] = 367;
index_r[5895] = 367;
index_r[5896] = 367;
index_r[5897] = 367;
index_r[5898] = 367;
index_r[5899] = 367;
index_r[5900] = 367;
index_r[5901] = 367;
index_r[5902] = 368;
index_r[5903] = 368;
index_r[5904] = 368;
index_r[5905] = 368;
index_r[5906] = 368;
index_r[5907] = 368;
index_r[5908] = 368;
index_r[5909] = 368;
index_r[5910] = 368;
index_r[5911] = 368;
index_r[5912] = 368;
index_r[5913] = 368;
index_r[5914] = 368;
index_r[5915] = 368;
index_r[5916] = 368;
index_r[5917] = 368;
index_r[5918] = 368;
index_r[5919] = 368;
index_r[5920] = 368;
index_r[5921] = 369;
index_r[5922] = 369;
index_r[5923] = 369;
index_r[5924] = 369;
index_r[5925] = 369;
index_r[5926] = 369;
index_r[5927] = 369;
index_r[5928] = 369;
index_r[5929] = 369;
index_r[5930] = 369;
index_r[5931] = 369;
index_r[5932] = 369;
index_r[5933] = 369;
index_r[5934] = 369;
index_r[5935] = 369;
index_r[5936] = 369;
index_r[5937] = 370;
index_r[5938] = 370;
index_r[5939] = 370;
index_r[5940] = 370;
index_r[5941] = 370;
index_r[5942] = 370;
index_r[5943] = 370;
index_r[5944] = 371;
index_r[5945] = 371;
index_r[5946] = 371;
index_r[5947] = 371;
index_r[5948] = 371;
index_r[5949] = 371;
index_r[5950] = 371;
index_r[5951] = 371;
index_r[5952] = 371;
index_r[5953] = 371;
index_r[5954] = 371;
index_r[5955] = 372;
index_r[5956] = 372;
index_r[5957] = 372;
index_r[5958] = 372;
index_r[5959] = 372;
index_r[5960] = 372;
index_r[5961] = 372;
index_r[5962] = 372;
index_r[5963] = 372;
index_r[5964] = 372;
index_r[5965] = 372;
index_r[5966] = 373;
index_r[5967] = 373;
index_r[5968] = 373;
index_r[5969] = 373;
index_r[5970] = 373;
index_r[5971] = 373;
index_r[5972] = 373;
index_r[5973] = 373;
index_r[5974] = 373;
index_r[5975] = 373;
index_r[5976] = 373;
index_r[5977] = 373;
index_r[5978] = 373;
index_r[5979] = 373;
index_r[5980] = 373;
index_r[5981] = 373;
index_r[5982] = 373;
index_r[5983] = 373;
index_r[5984] = 373;
index_r[5985] = 373;
index_r[5986] = 373;
index_r[5987] = 373;
index_r[5988] = 373;
index_r[5989] = 373;
index_r[5990] = 374;
index_r[5991] = 374;
index_r[5992] = 374;
index_r[5993] = 374;
index_r[5994] = 374;
index_r[5995] = 374;
index_r[5996] = 374;
index_r[5997] = 374;
index_r[5998] = 374;
index_r[5999] = 374;
index_r[6000] = 374;
index_r[6001] = 374;
index_r[6002] = 374;
index_r[6003] = 374;
index_r[6004] = 374;
index_r[6005] = 375;
index_r[6006] = 375;
index_r[6007] = 375;
index_r[6008] = 375;
index_r[6009] = 375;
index_r[6010] = 375;
index_r[6011] = 375;
index_r[6012] = 375;
index_r[6013] = 375;
index_r[6014] = 375;
index_r[6015] = 375;
index_r[6016] = 375;
index_r[6017] = 375;
index_r[6018] = 375;
index_r[6019] = 376;
index_r[6020] = 376;
index_r[6021] = 376;
index_r[6022] = 376;
index_r[6023] = 376;
index_r[6024] = 376;
index_r[6025] = 376;
index_r[6026] = 376;
index_r[6027] = 376;
index_r[6028] = 376;
index_r[6029] = 376;
index_r[6030] = 376;
index_r[6031] = 376;
index_r[6032] = 376;
index_r[6033] = 376;
index_r[6034] = 376;
index_r[6035] = 377;
index_r[6036] = 377;
index_r[6037] = 377;
index_r[6038] = 377;
index_r[6039] = 377;
index_r[6040] = 377;
index_r[6041] = 377;
index_r[6042] = 378;
index_r[6043] = 378;
index_r[6044] = 378;
index_r[6045] = 378;
index_r[6046] = 378;
index_r[6047] = 378;
index_r[6048] = 378;
index_r[6049] = 378;
index_r[6050] = 378;
index_r[6051] = 378;
index_r[6052] = 378;
index_r[6053] = 378;
index_r[6054] = 378;
index_r[6055] = 378;
index_r[6056] = 378;
index_r[6057] = 378;
index_r[6058] = 378;
index_r[6059] = 378;
index_r[6060] = 378;
index_r[6061] = 378;
index_r[6062] = 378;
index_r[6063] = 378;
index_r[6064] = 378;
index_r[6065] = 378;
index_r[6066] = 379;
index_r[6067] = 379;
index_r[6068] = 379;
index_r[6069] = 379;
index_r[6070] = 379;
index_r[6071] = 379;
index_r[6072] = 379;
index_r[6073] = 379;
index_r[6074] = 379;
index_r[6075] = 379;
index_r[6076] = 379;
index_r[6077] = 379;
index_r[6078] = 379;
index_r[6079] = 379;
index_r[6080] = 379;
index_r[6081] = 379;
index_r[6082] = 379;
index_r[6083] = 379;
index_r[6084] = 379;
index_r[6085] = 380;
index_r[6086] = 380;
index_r[6087] = 380;
index_r[6088] = 380;
index_r[6089] = 380;
index_r[6090] = 380;
index_r[6091] = 380;
index_r[6092] = 380;
index_r[6093] = 380;
index_r[6094] = 380;
index_r[6095] = 380;
index_r[6096] = 380;
index_r[6097] = 380;
index_r[6098] = 380;
index_r[6099] = 380;
index_r[6100] = 380;
index_r[6101] = 380;
index_r[6102] = 380;
index_r[6103] = 380;
index_r[6104] = 381;
index_r[6105] = 381;
index_r[6106] = 381;
index_r[6107] = 381;
index_r[6108] = 381;
index_r[6109] = 381;
index_r[6110] = 381;
index_r[6111] = 381;
index_r[6112] = 381;
index_r[6113] = 381;
index_r[6114] = 381;
index_r[6115] = 381;
index_r[6116] = 381;
index_r[6117] = 381;
index_r[6118] = 381;
index_r[6119] = 381;
index_r[6120] = 381;
index_r[6121] = 381;
index_r[6122] = 381;
index_r[6123] = 381;
index_r[6124] = 381;
index_r[6125] = 381;
index_r[6126] = 382;
index_r[6127] = 382;
index_r[6128] = 382;
index_r[6129] = 382;
index_r[6130] = 382;
index_r[6131] = 382;
index_r[6132] = 382;
index_r[6133] = 382;
index_r[6134] = 382;
index_r[6135] = 382;
index_r[6136] = 382;
index_r[6137] = 382;
index_r[6138] = 382;
index_r[6139] = 382;
index_r[6140] = 382;
index_r[6141] = 382;
index_r[6142] = 382;
index_r[6143] = 383;
index_r[6144] = 383;
index_r[6145] = 383;
index_r[6146] = 383;
index_r[6147] = 383;
index_r[6148] = 383;
index_r[6149] = 383;
index_r[6150] = 383;
index_r[6151] = 383;
index_r[6152] = 383;
index_r[6153] = 383;
index_r[6154] = 383;
index_r[6155] = 383;
index_r[6156] = 383;
index_r[6157] = 383;
index_r[6158] = 383;
index_r[6159] = 383;
index_r[6160] = 383;
index_r[6161] = 383;
index_r[6162] = 384;
index_r[6163] = 384;
index_r[6164] = 384;
index_r[6165] = 384;
index_r[6166] = 384;
index_r[6167] = 384;
index_r[6168] = 384;
index_r[6169] = 384;
index_r[6170] = 384;
index_r[6171] = 384;
index_r[6172] = 384;
index_r[6173] = 384;
index_r[6174] = 384;
index_r[6175] = 384;
index_r[6176] = 384;
index_r[6177] = 385;
index_r[6178] = 385;
index_r[6179] = 385;
index_r[6180] = 385;
index_r[6181] = 385;
index_r[6182] = 385;
index_r[6183] = 385;
index_r[6184] = 385;
index_r[6185] = 385;
index_r[6186] = 385;
index_r[6187] = 385;
index_r[6188] = 385;
index_r[6189] = 386;
index_r[6190] = 386;
index_r[6191] = 386;
index_r[6192] = 386;
index_r[6193] = 386;
index_r[6194] = 386;
index_r[6195] = 386;
index_r[6196] = 386;
index_r[6197] = 386;
index_r[6198] = 386;
index_r[6199] = 386;
index_r[6200] = 386;
index_r[6201] = 386;
index_r[6202] = 386;
index_r[6203] = 386;
index_r[6204] = 386;
index_r[6205] = 386;
index_r[6206] = 387;
index_r[6207] = 387;
index_r[6208] = 387;
index_r[6209] = 387;
index_r[6210] = 387;
index_r[6211] = 387;
index_r[6212] = 387;
index_r[6213] = 387;
index_r[6214] = 387;
index_r[6215] = 387;
index_r[6216] = 387;
index_r[6217] = 387;
index_r[6218] = 387;
index_r[6219] = 387;
index_r[6220] = 388;
index_r[6221] = 388;
index_r[6222] = 388;
index_r[6223] = 388;
index_r[6224] = 388;
index_r[6225] = 388;
index_r[6226] = 388;
index_r[6227] = 388;
index_r[6228] = 388;
index_r[6229] = 388;
index_r[6230] = 388;
index_r[6231] = 388;
index_r[6232] = 388;
index_r[6233] = 388;
index_r[6234] = 388;
index_r[6235] = 388;
index_r[6236] = 388;
index_r[6237] = 388;
index_r[6238] = 388;
index_r[6239] = 389;
index_r[6240] = 389;
index_r[6241] = 389;
index_r[6242] = 389;
index_r[6243] = 389;
index_r[6244] = 389;
index_r[6245] = 389;
index_r[6246] = 389;
index_r[6247] = 389;
index_r[6248] = 389;
index_r[6249] = 389;
index_r[6250] = 389;
index_r[6251] = 389;
index_r[6252] = 389;
index_r[6253] = 389;
index_r[6254] = 389;
index_r[6255] = 389;
index_r[6256] = 389;
index_r[6257] = 389;
index_r[6258] = 390;
index_r[6259] = 390;
index_r[6260] = 390;
index_r[6261] = 390;
index_r[6262] = 390;
index_r[6263] = 390;
index_r[6264] = 390;
index_r[6265] = 390;
index_r[6266] = 390;
index_r[6267] = 390;
index_r[6268] = 390;
index_r[6269] = 391;
index_r[6270] = 391;
index_r[6271] = 391;
index_r[6272] = 391;
index_r[6273] = 391;
index_r[6274] = 391;
index_r[6275] = 391;
index_r[6276] = 391;
index_r[6277] = 391;
index_r[6278] = 391;
index_r[6279] = 392;
index_r[6280] = 392;
index_r[6281] = 392;
index_r[6282] = 392;
index_r[6283] = 392;
index_r[6284] = 392;
index_r[6285] = 392;
index_r[6286] = 392;
index_r[6287] = 392;
index_r[6288] = 392;
index_r[6289] = 392;
index_r[6290] = 392;
index_r[6291] = 392;
index_r[6292] = 392;
index_r[6293] = 392;
index_r[6294] = 392;
index_r[6295] = 392;
index_r[6296] = 393;
index_r[6297] = 393;
index_r[6298] = 393;
index_r[6299] = 393;
index_r[6300] = 393;
index_r[6301] = 393;
index_r[6302] = 393;
index_r[6303] = 394;
index_r[6304] = 394;
index_r[6305] = 394;
index_r[6306] = 394;
index_r[6307] = 394;
index_r[6308] = 394;
index_r[6309] = 394;
index_r[6310] = 394;
index_r[6311] = 394;
index_r[6312] = 394;
index_r[6313] = 394;
index_r[6314] = 394;
index_r[6315] = 394;
index_r[6316] = 394;
index_r[6317] = 394;
index_r[6318] = 394;
index_r[6319] = 394;
index_r[6320] = 394;
index_r[6321] = 394;
index_r[6322] = 394;
index_r[6323] = 394;
index_r[6324] = 394;
index_r[6325] = 395;
index_r[6326] = 395;
index_r[6327] = 395;
index_r[6328] = 395;
index_r[6329] = 395;
index_r[6330] = 395;
index_r[6331] = 395;
index_r[6332] = 395;
index_r[6333] = 395;
index_r[6334] = 395;
index_r[6335] = 395;
index_r[6336] = 395;
index_r[6337] = 395;
index_r[6338] = 395;
index_r[6339] = 396;
index_r[6340] = 396;
index_r[6341] = 396;
index_r[6342] = 396;
index_r[6343] = 396;
index_r[6344] = 396;
index_r[6345] = 396;
index_r[6346] = 396;
index_r[6347] = 396;
index_r[6348] = 396;
index_r[6349] = 396;
index_r[6350] = 396;
index_r[6351] = 396;
index_r[6352] = 396;
index_r[6353] = 396;
index_r[6354] = 396;
index_r[6355] = 396;
index_r[6356] = 396;
index_r[6357] = 396;
index_r[6358] = 397;
index_r[6359] = 397;
index_r[6360] = 397;
index_r[6361] = 397;
index_r[6362] = 397;
index_r[6363] = 397;
index_r[6364] = 397;
index_r[6365] = 397;
index_r[6366] = 397;
index_r[6367] = 397;
index_r[6368] = 397;
index_r[6369] = 397;
index_r[6370] = 397;
index_r[6371] = 397;
index_r[6372] = 397;
index_r[6373] = 397;
index_r[6374] = 398;
index_r[6375] = 398;
index_r[6376] = 398;
index_r[6377] = 398;
index_r[6378] = 398;
index_r[6379] = 398;
index_r[6380] = 398;
index_r[6381] = 398;
index_r[6382] = 398;
index_r[6383] = 398;
index_r[6384] = 398;
index_r[6385] = 398;
index_r[6386] = 398;
index_r[6387] = 398;
index_r[6388] = 398;
index_r[6389] = 398;
index_r[6390] = 399;
index_r[6391] = 399;
index_r[6392] = 399;
index_r[6393] = 399;
index_r[6394] = 399;
index_r[6395] = 399;
index_r[6396] = 399;
index_r[6397] = 399;
index_r[6398] = 399;
index_r[6399] = 399;
index_r[6400] = 399;
index_r[6401] = 399;
index_r[6402] = 399;
index_r[6403] = 399;
index_r[6404] = 399;
index_r[6405] = 399;
index_r[6406] = 399;
index_r[6407] = 399;
index_r[6408] = 399;
index_r[6409] = 399;
index_r[6410] = 399;
index_r[6411] = 400;
index_r[6412] = 400;
index_r[6413] = 400;
index_r[6414] = 400;
index_r[6415] = 400;
index_r[6416] = 400;
index_r[6417] = 400;
index_r[6418] = 400;


int nres = 401;*/

 t_bin *rb;
 int isv,*isp;
 real ene0,pot0,diff0;
 rb = mk_bin();
 real tmp;

 snew(isp,nres);


#ifdef DO_FLOW
    snew(mdatoms->node_ener,nres);
    snew(mdatoms->edge_ener,nres);
    mdatoms->sol_res = nres-1;
    for(ii=0;ii<nres;ii++)
    {
     snew(mdatoms->edge_ener[ii],nres);
    }

    real *node_ener;
    real **edge_ener;
    node_ener = mdatoms->node_ener;
    edge_ener = mdatoms->edge_ener;

    real node_ener0[MNRES];
    real node_flow[MNRES];
#ifdef DO_EDGE
    real edge_ener0[MNRES][MNRES];
    real flow[MNRES][MNRES];
#endif

            int jj;
             for(ii=0;ii<MNRES;ii++)
             {
              node_flow[ii] = 0.0;
#ifdef DO_EDGE
              for(jj=0;jj<MNRES;jj++)
              {
               flow[ii][jj] = 0.0;
               edge_ener0[ii][jj] = 0.0;
              }
#endif
             }
#endif


    /* Check for special mdrun options */
    bRerunMD = (Flags & MD_RERUN);
    bIonize  = (Flags & MD_IONIZE);
    bFFscan  = (Flags & MD_FFSCAN);
    bAppend  = (Flags & MD_APPENDFILES);

#ifdef DO_FLOW
    ir->nsteps = 400000;
#endif

    if (Flags & MD_RESETCOUNTERSHALFWAY)
    {
        if (ir->nsteps > 0)
        {
            /* Signal to reset the counters half the simulation steps. */
            wcycle_set_reset_counters(wcycle, ir->nsteps/2);
        }
        /* Signal to reset the counters halfway the simulation time. */
        bResetCountersHalfMaxH = (max_hours > 0);
    }

    /* md-vv uses averaged full step velocities for T-control
       md-vv-avek uses averaged half step velocities for T-control (but full step ekin for P control)
       md uses averaged half step kinetic energies to determine temperature unless defined otherwise by GMX_EKIN_AVE_VEL; */
    bVV = EI_VV(ir->eI);
    if (bVV) /* to store the initial velocities while computing virial */
    {
        snew(cbuf, top_global->natoms);
    }
    /* all the iteratative cases - only if there are constraints */
    bIterativeCase = ((IR_NPH_TROTTER(ir) || IR_NPT_TROTTER(ir)) && (constr) && (!bRerunMD));
    gmx_iterate_init(&iterate, FALSE); /* The default value of iterate->bIterationActive is set to
                                          false in this step.  The correct value, true or false,
                                          is set at each step, as it depends on the frequency of temperature
                                          and pressure control.*/
    bTrotter = (bVV && (IR_NPT_TROTTER(ir) || IR_NPH_TROTTER(ir) || IR_NVT_TROTTER(ir)));

    if (bRerunMD)
    {
        /* Since we don't know if the frames read are related in any way,
         * rebuild the neighborlist at every step.
         */
        ir->nstlist       = 1;
        ir->nstcalcenergy = 1;
        nstglobalcomm     = 1;
    }

    check_ir_old_tpx_versions(cr, fplog, ir, top_global);

    nstglobalcomm   = check_nstglobalcomm(fplog, cr, nstglobalcomm, ir);
    bGStatEveryStep = (nstglobalcomm == 1);

    if (!bGStatEveryStep && ir->nstlist == -1 && fplog != NULL)
    {
        fprintf(fplog,
                "To reduce the energy communication with nstlist = -1\n"
                "the neighbor list validity should not be checked at every step,\n"
                "this means that exact integration is not guaranteed.\n"
                "The neighbor list validity is checked after:\n"
                "  <n.list life time> - 2*std.dev.(n.list life time)  steps.\n"
                "In most cases this will result in exact integration.\n"
                "This reduces the energy communication by a factor of 2 to 3.\n"
                "If you want less energy communication, set nstlist > 3.\n\n");
    }

    if (bRerunMD || bFFscan)
    {
        ir->nstxtcout = 0;
    }
    groups = &top_global->groups;

    /* Initial values */
    init_md(fplog, cr, ir, oenv, &t, &t0, state_global->lambda,
            &(state_global->fep_state), lam0,
            nrnb, top_global, &upd,
            nfile, fnm, &outf, &mdebin,
            force_vir, shake_vir, mu_tot, &bSimAnn, &vcm, state_global, Flags);

    int quitframe = 10000;   
    mdebin->doAvEner = TRUE;
    //mdebin->doAvEner = FALSE;

    clear_mat(total_vir);
    clear_mat(pres);
    /* Energy terms and groups */
    snew(enerd, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, ir->fepvals->n_lambda,
                  enerd);
    if (DOMAINDECOMP(cr))
    {
        f = NULL;
    }
    else
    {
        snew(f, top_global->natoms);
    }

    /* lambda Monte carlo random number generator  */
    if (ir->bExpanded)
    {
        mcrng = gmx_rng_init(ir->expandedvals->lmc_seed);
    }
    /* copy the state into df_history */
    copy_df_history(&df_history, &state_global->dfhist);

    /* Kinetic energy data */
    snew(ekind, 1);
    init_ekindata(fplog, top_global, &(ir->opts), ekind);
    /* needed for iteration of constraints */
    snew(ekind_save, 1);
    init_ekindata(fplog, top_global, &(ir->opts), ekind_save);
    /* Copy the cos acceleration to the groups struct */
    ekind->cosacc.cos_accel = ir->cos_accel;

    gstat = global_stat_init(ir);
    debug_gmx();

    /* Check for polarizable models and flexible constraints */
    shellfc = init_shell_flexcon(fplog,
                                 top_global, n_flexible_constraints(constr),
                                 (ir->bContinuation ||
                                  (DOMAINDECOMP(cr) && !MASTER(cr))) ?
                                 NULL : state_global->x);

    if (DEFORM(*ir))
    {
#ifdef GMX_THREAD_MPI
        tMPI_Thread_mutex_lock(&deform_init_box_mutex);
#endif
        set_deform_reference_box(upd,
                                 deform_init_init_step_tpx,
                                 deform_init_box_tpx);
#ifdef GMX_THREAD_MPI
        tMPI_Thread_mutex_unlock(&deform_init_box_mutex);
#endif
    }

    {
        double io = compute_io(ir, top_global->natoms, groups, mdebin->ebin->nener, 1);
        if ((io > 2000) && MASTER(cr))
        {
            fprintf(stderr,
                    "\nWARNING: This run will generate roughly %.0f Mb of data\n\n",
                    io);
        }
    }

    if (DOMAINDECOMP(cr))
    {
        top = dd_init_local_top(top_global);

        snew(state, 1);
        dd_init_local_state(cr->dd, state_global, state);

        if (DDMASTER(cr->dd) && ir->nstfout)
        {
            snew(f_global, state_global->natoms);
        }
    }
    else
    {
        if (PAR(cr))
        {
            /* Initialize the particle decomposition and split the topology */
            top = split_system(fplog, top_global, ir, cr);

            pd_cg_range(cr, &fr->cg0, &fr->hcg);
            pd_at_range(cr, &a0, &a1);
        }
        else
        {
            top = gmx_mtop_generate_local_top(top_global, ir);

            a0 = 0;
            a1 = top_global->natoms;
        }

        forcerec_set_excl_load(fr, top, cr);

        state    = partdec_init_local_state(cr, state_global);
        f_global = f;

        atoms2md(top_global, ir, 0, NULL, a0, a1-a0, mdatoms);

        if (vsite)
        {
            set_vsite_top(vsite, top, mdatoms, cr);
        }

        if (ir->ePBC != epbcNONE && !fr->bMolPBC)
        {
            graph = mk_graph(fplog, &(top->idef), 0, top_global->natoms, FALSE, FALSE);
        }

        if (shellfc)
        {
            make_local_shells(cr, mdatoms, shellfc);
        }

        init_bonded_thread_force_reduction(fr, &top->idef);

        if (ir->pull && PAR(cr))
        {
            dd_make_local_pull_groups(NULL, ir->pull, mdatoms);
        }
    }

    if (DOMAINDECOMP(cr))
    {
        /* Distribute the charge groups over the nodes from the master node */
        dd_partition_system(fplog, ir->init_step, cr, TRUE, 1,
                            state_global, top_global, ir,
                            state, &f, mdatoms, top, fr,
                            vsite, shellfc, constr,
                            nrnb, wcycle, FALSE);

    }

    update_mdatoms(mdatoms, state->lambda[efptMASS]);

    if (opt2bSet("-cpi", nfile, fnm))
    {
        bStateFromCP = gmx_fexist_master(opt2fn_master("-cpi", nfile, fnm, cr), cr);
    }
    else
    {
        bStateFromCP = FALSE;
    }

    if (MASTER(cr))
    {
        if (bStateFromCP)
        {
            /* Update mdebin with energy history if appending to output files */
            if (Flags & MD_APPENDFILES)
            {
                restore_energyhistory_from_state(mdebin, &state_global->enerhist);
            }
            else
            {
                /* We might have read an energy history from checkpoint,
                 * free the allocated memory and reset the counts.
                 */
                done_energyhistory(&state_global->enerhist);
                init_energyhistory(&state_global->enerhist);
            }
        }
        /* Set the initial energy history in state by updating once */
        update_energyhistory(&state_global->enerhist, mdebin);
    }

    if ((state->flags & (1<<estLD_RNG)) && (Flags & MD_READ_RNG))
    {
        /* Set the random state if we read a checkpoint file */
        set_stochd_state(upd, state);
    }

    if (state->flags & (1<<estMC_RNG))
    {
        set_mc_state(mcrng, state);
    }

    /* Initialize constraints */
    if (constr)
    {
        if (!DOMAINDECOMP(cr))
        {
            set_constraints(constr, top, ir, mdatoms, cr);
        }
    }

    /* Check whether we have to GCT stuff */
    bTCR = ftp2bSet(efGCT, nfile, fnm);
    if (bTCR)
    {
        if (MASTER(cr))
        {
            fprintf(stderr, "Will do General Coupling Theory!\n");
        }
        gnx = top_global->mols.nr;
        snew(grpindex, gnx);
        for (i = 0; (i < gnx); i++)
        {
            grpindex[i] = i;
        }
    }

    if (repl_ex_nst > 0)
    {
        /* We need to be sure replica exchange can only occur
         * when the energies are current */
        check_nst_param(fplog, cr, "nstcalcenergy", ir->nstcalcenergy,
                        "repl_ex_nst", &repl_ex_nst);
        /* This check needs to happen before inter-simulation
         * signals are initialized, too */
    }
    if (repl_ex_nst > 0 && MASTER(cr))
    {
        repl_ex = init_replica_exchange(fplog, cr->ms, state_global, ir,
                                        repl_ex_nst, repl_ex_nex, repl_ex_seed);
    }

    /* PME tuning is only supported with GPUs or PME nodes and not with rerun.
     * With perturbed charges with soft-core we should not change the cut-off.
     */
    if ((Flags & MD_TUNEPME) &&
        EEL_PME(fr->eeltype) &&
        ( (fr->cutoff_scheme == ecutsVERLET && fr->nbv->bUseGPU) || !(cr->duty & DUTY_PME)) &&
        !(ir->efep != efepNO && mdatoms->nChargePerturbed > 0 && ir->fepvals->bScCoul) &&
        !bRerunMD)
    {
        pme_loadbal_init(&pme_loadbal, ir, state->box, fr->ic, fr->pmedata);
        cycles_pmes = 0;
        if (cr->duty & DUTY_PME)
        {
            /* Start tuning right away, as we can't measure the load */
            bPMETuneRunning = TRUE;
        }
        else
        {
            /* Separate PME nodes, we can measure the PP/PME load balance */
            bPMETuneTry = TRUE;
        }
    }

    if (!ir->bContinuation && !bRerunMD)
    {
        if (mdatoms->cFREEZE && (state->flags & (1<<estV)))
        {
            /* Set the velocities of frozen particles to zero */
            for (i = mdatoms->start; i < mdatoms->start+mdatoms->homenr; i++)
            {
                for (m = 0; m < DIM; m++)
                {
                    if (ir->opts.nFreeze[mdatoms->cFREEZE[i]][m])
                    {
                        state->v[i][m] = 0;
                    }
                }
            }
        }

        if (constr)
        {
            /* Constrain the initial coordinates and velocities */
            do_constrain_first(fplog, constr, ir, mdatoms, state, f,
                               graph, cr, nrnb, fr, top, shake_vir);
        }
        if (vsite)
        {
            /* Construct the virtual sites for the initial configuration */
            construct_vsites(fplog, vsite, state->x, nrnb, ir->delta_t, NULL,
                             top->idef.iparams, top->idef.il,
                             fr->ePBC, fr->bMolPBC, graph, cr, state->box);
        }
    }

    debug_gmx();

    /* set free energy calculation frequency as the minimum
       greatest common denominator of nstdhdl, nstexpanded, and repl_ex_nst*/
    nstfep = ir->fepvals->nstdhdl;
    if (ir->bExpanded)
    {
        nstfep = gmx_greatest_common_divisor(ir->fepvals->nstdhdl,nstfep);
    }
    if (repl_ex_nst > 0)
    {
        nstfep = gmx_greatest_common_divisor(repl_ex_nst,nstfep);
    }

    /* I'm assuming we need global communication the first time! MRS */
    cglo_flags = (CGLO_TEMPERATURE | CGLO_GSTAT
                  | ((ir->comm_mode != ecmNO) ? CGLO_STOPCM : 0)
                  | (bVV ? CGLO_PRESSURE : 0)
                  | (bVV ? CGLO_CONSTRAINT : 0)
                  | (bRerunMD ? CGLO_RERUNMD : 0)
                  | ((Flags & MD_READ_EKIN) ? CGLO_READEKIN : 0));

    bSumEkinhOld = FALSE;
    compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                    NULL, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                    constr, NULL, FALSE, state->box,
                    top_global, &pcurr, top_global->natoms, &bSumEkinhOld, cglo_flags);
    if (ir->eI == eiVVAK)
    {
        /* a second call to get the half step temperature initialized as well */
        /* we do the same call as above, but turn the pressure off -- internally to
           compute_globals, this is recognized as a velocity verlet half-step
           kinetic energy calculation.  This minimized excess variables, but
           perhaps loses some logic?*/

        compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                        NULL, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                        constr, NULL, FALSE, state->box,
                        top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                        cglo_flags &~(CGLO_STOPCM | CGLO_PRESSURE));
    }

    /* Calculate the initial half step temperature, and save the ekinh_old */
    if (!(Flags & MD_STARTFROMCPT))
    {
        for (i = 0; (i < ir->opts.ngtc); i++)
        {
            copy_mat(ekind->tcstat[i].ekinh, ekind->tcstat[i].ekinh_old);
        }
    }
    if (ir->eI != eiVV)
    {
        enerd->term[F_TEMP] *= 2; /* result of averages being done over previous and current step,
                                     and there is no previous step */
    }

    /* if using an iterative algorithm, we need to create a working directory for the state. */
    if (bIterativeCase)
    {
        bufstate = init_bufstate(state);
    }
    if (bFFscan)
    {
        snew(xcopy, state->natoms);
        snew(vcopy, state->natoms);
        copy_rvecn(state->x, xcopy, 0, state->natoms);
        copy_rvecn(state->v, vcopy, 0, state->natoms);
        copy_mat(state->box, boxcopy);
    }

    /* need to make an initiation call to get the Trotter variables set, as well as other constants for non-trotter
       temperature control */
    trotter_seq = init_npt_vars(ir, state, &MassQ, bTrotter);

    if (MASTER(cr))
    {
        if (constr && !ir->bContinuation && ir->eConstrAlg == econtLINCS)
        {
            fprintf(fplog,
                    "RMS relative constraint deviation after constraining: %.2e\n",
                    constr_rmsd(constr, FALSE));
        }
        if (EI_STATE_VELOCITY(ir->eI))
        {
            fprintf(fplog, "Initial temperature: %g K\n", enerd->term[F_TEMP]);
        }
        if (bRerunMD)
        {
            fprintf(stderr, "starting md rerun '%s', reading coordinates from"
                    " input trajectory '%s'\n\n",
                    *(top_global->name), opt2fn("-rerun", nfile, fnm));
            if (bVerbose)
            {
                fprintf(stderr, "Calculated time to finish depends on nsteps from "
                        "run input file,\nwhich may not correspond to the time "
                        "needed to process input trajectory.\n\n");
            }
        }
        else
        {
            char tbuf[20];
            fprintf(stderr, "starting mdrun '%s'\n",
                    *(top_global->name));
            if (ir->nsteps >= 0)
            {
                sprintf(tbuf, "%8.1f", (ir->init_step+ir->nsteps)*ir->delta_t);
            }
            else
            {
                sprintf(tbuf, "%s", "infinite");
            }
            if (ir->init_step > 0)
            {
                fprintf(stderr, "%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                        gmx_step_str(ir->init_step+ir->nsteps, sbuf), tbuf,
                        gmx_step_str(ir->init_step, sbuf2),
                        ir->init_step*ir->delta_t);
            }
            else
            {
                fprintf(stderr, "%s steps, %s ps.\n",
                        gmx_step_str(ir->nsteps, sbuf), tbuf);
            }
        }
        fprintf(fplog, "\n");
    }

    /* Set and write start time */
    runtime_start(runtime);
    print_date_and_time(fplog, cr->nodeid, "Started mdrun", runtime);
    wallcycle_start(wcycle, ewcRUN);
    if (fplog)
    {
        fprintf(fplog, "\n");
    }

    /* safest point to do file checkpointing is here.  More general point would be immediately before integrator call */
#ifdef GMX_FAHCORE
    chkpt_ret = fcCheckPointParallel( cr->nodeid,
                                      NULL, 0);
    if (chkpt_ret == 0)
    {
        gmx_fatal( 3, __FILE__, __LINE__, "Checkpoint error on step %d\n", 0 );
    }
#endif

    debug_gmx();
    /***********************************************************
     *
     *             Loop over MD steps
     *
     ************************************************************/

    /* if rerunMD then read coordinates and velocities from input trajectory */
    if (bRerunMD)
    {
        if (getenv("GMX_FORCE_UPDATE"))
        {
            bForceUpdate = TRUE;
        }

        rerun_fr.natoms = 0;
        if (MASTER(cr))
        {
            bNotLastFrame = read_first_frame(oenv, &status,
                                             opt2fn("-rerun", nfile, fnm),
                                             &rerun_fr, TRX_NEED_X | TRX_READ_V);
            if (rerun_fr.natoms != top_global->natoms)
            {
                gmx_fatal(FARGS,
                          "Number of atoms in trajectory (%d) does not match the "
                          "run input file (%d)\n",
                          rerun_fr.natoms, top_global->natoms);
            }
            if (ir->ePBC != epbcNONE)
            {
                if (!rerun_fr.bBox)
                {
                    gmx_fatal(FARGS, "Rerun trajectory frame step %d time %f does not contain a box, while pbc is used", rerun_fr.step, rerun_fr.time);
                }
                if (max_cutoff2(ir->ePBC, rerun_fr.box) < sqr(fr->rlistlong))
                {
                    gmx_fatal(FARGS, "Rerun trajectory frame step %d time %f has too small box dimensions", rerun_fr.step, rerun_fr.time);
                }
            }
        }

        if (PAR(cr))
        {
            rerun_parallel_comm(cr, &rerun_fr, &bNotLastFrame);
        }

        if (ir->ePBC != epbcNONE)
        {
            /* Set the shift vectors.
             * Necessary here when have a static box different from the tpr box.
             */
            calc_shifts(rerun_fr.box, fr->shift_vec);
        }
    }

    /* loop over MD steps or if rerunMD to end of input trajectory */
    bFirstStep = TRUE;
    /* Skip the first Nose-Hoover integration when we get the state from tpx */
    bStateFromTPX    = !bStateFromCP;
    bInitStep        = bFirstStep && (bStateFromTPX || bVV);
    bStartingFromCpt = (Flags & MD_STARTFROMCPT) && bInitStep;
    bLastStep        = FALSE;
    bSumEkinhOld     = FALSE;
    bExchanged       = FALSE;

    init_global_signals(&gs, cr, ir, repl_ex_nst);

    step     = ir->init_step;
    step_rel = 0;

    if (ir->nstlist == -1)
    {
        init_nlistheuristics(&nlh, bGStatEveryStep, step);
    }

    if (MULTISIM(cr) && (repl_ex_nst <= 0 ))
    {
        /* check how many steps are left in other sims */
        multisim_nsteps = get_multisim_nsteps(cr, ir->nsteps);
    }


    /* and stop now if we should */
    bLastStep = (bRerunMD || (ir->nsteps >= 0 && step_rel > ir->nsteps) ||
                 ((multisim_nsteps >= 0) && (step_rel >= multisim_nsteps )));
    while (!bLastStep || (bRerunMD && bNotLastFrame))
    {

        wallcycle_start(wcycle, ewcSTEP);

        GMX_MPE_LOG(ev_timestep1);

        if (bRerunMD)
        {
            if (rerun_fr.bStep)
            {
                step     = rerun_fr.step;
                step_rel = step - ir->init_step;
            }
            if (rerun_fr.bTime)
            {
                t = rerun_fr.time;
            }
            else
            {
                t = step;
            }
        }
        else
        {
            bLastStep = (step_rel == ir->nsteps);
            t         = t0 + step*ir->delta_t;
        }

        if (ir->efep != efepNO || ir->bSimTemp)
        {
            /* find and set the current lambdas.  If rerunning, we either read in a state, or a lambda value,
               requiring different logic. */

            set_current_lambdas(step, ir->fepvals, bRerunMD, &rerun_fr, state_global, state, lam0);
            bDoDHDL      = do_per_step(step, ir->fepvals->nstdhdl);
            bDoFEP       = (do_per_step(step, nstfep) && (ir->efep != efepNO));
            bDoExpanded  = (do_per_step(step, ir->expandedvals->nstexpanded) && (ir->bExpanded) && (step > 0));
        }

        if (bSimAnn)
        {
            update_annealing_target_temp(&(ir->opts), t);
        }

        if (bRerunMD)
        {
            if (!(DOMAINDECOMP(cr) && !MASTER(cr)))
            {
                for (i = 0; i < state_global->natoms; i++)
                {
                    copy_rvec(rerun_fr.x[i], state_global->x[i]);
                }
                if (rerun_fr.bV)
                {
                    for (i = 0; i < state_global->natoms; i++)
                    {
                        copy_rvec(rerun_fr.v[i], state_global->v[i]);
                    }
                }
                else
                {
                    for (i = 0; i < state_global->natoms; i++)
                    {
                        clear_rvec(state_global->v[i]);
                    }
                    if (bRerunWarnNoV)
                    {
                        fprintf(stderr, "\nWARNING: Some frames do not contain velocities.\n"
                                "         Ekin, temperature and pressure are incorrect,\n"
                                "         the virial will be incorrect when constraints are present.\n"
                                "\n");
                        bRerunWarnNoV = FALSE;
                    }
                }
            }
            copy_mat(rerun_fr.box, state_global->box);
            copy_mat(state_global->box, state->box);

            if (vsite && (Flags & MD_RERUN_VSITE))
            {
                if (DOMAINDECOMP(cr))
                {
                    gmx_fatal(FARGS, "Vsite recalculation with -rerun is not implemented for domain decomposition, use particle decomposition");
                }
                if (graph)
                {
                    /* Following is necessary because the graph may get out of sync
                     * with the coordinates if we only have every N'th coordinate set
                     */
                    mk_mshift(fplog, graph, fr->ePBC, state->box, state->x);
                    shift_self(graph, state->box, state->x);
                }
                construct_vsites(fplog, vsite, state->x, nrnb, ir->delta_t, state->v,
                                 top->idef.iparams, top->idef.il,
                                 fr->ePBC, fr->bMolPBC, graph, cr, state->box);
                if (graph)
                {
                    unshift_self(graph, state->box, state->x);
                }
            }
        }

        /* Stop Center of Mass motion */
        bStopCM = (ir->comm_mode != ecmNO && do_per_step(step, ir->nstcomm));

        /* Copy back starting coordinates in case we're doing a forcefield scan */
        if (bFFscan)
        {
            for (ii = 0; (ii < state->natoms); ii++)
            {
                copy_rvec(xcopy[ii], state->x[ii]);
                copy_rvec(vcopy[ii], state->v[ii]);
            }
            copy_mat(boxcopy, state->box);
        }

        if (bRerunMD)
        {
            /* for rerun MD always do Neighbour Searching */
            bNS      = (bFirstStep || ir->nstlist != 0);
            bNStList = bNS;
        }
        else
        {
            /* Determine whether or not to do Neighbour Searching and LR */
            bNStList = (ir->nstlist > 0  && step % ir->nstlist == 0);

            bNS = (bFirstStep || bExchanged || bNStList || bDoFEP ||
                   (ir->nstlist == -1 && nlh.nabnsb > 0));

            if (bNS && ir->nstlist == -1)
            {
                set_nlistheuristics(&nlh, bFirstStep || bExchanged || bDoFEP, step);
            }
        }

        /* check whether we should stop because another simulation has
           stopped. */
        if (MULTISIM(cr))
        {
            if ( (multisim_nsteps >= 0) &&  (step_rel >= multisim_nsteps)  &&
                 (multisim_nsteps != ir->nsteps) )
            {
                if (bNS)
                {
                    if (MASTER(cr))
                    {
                        fprintf(stderr,
                                "Stopping simulation %d because another one has finished\n",
                                cr->ms->sim);
                    }
                    bLastStep         = TRUE;
                    gs.sig[eglsCHKPT] = 1;
                }
            }
        }

        /* < 0 means stop at next step, > 0 means stop at next NS step */
        if ( (gs.set[eglsSTOPCOND] < 0) ||
             ( (gs.set[eglsSTOPCOND] > 0) && (bNStList || ir->nstlist == 0) ) )
        {
            bLastStep = TRUE;
        }

        /* Determine whether or not to update the Born radii if doing GB */
        bBornRadii = bFirstStep;
        if (ir->implicit_solvent && (step % ir->nstgbradii == 0))
        {
            bBornRadii = TRUE;
        }

        do_log     = do_per_step(step, ir->nstlog) || bFirstStep || bLastStep;
        do_verbose = bVerbose &&
            (step % stepout == 0 || bFirstStep || bLastStep);

        if (bNS && !(bFirstStep && ir->bContinuation && !bRerunMD))
        {
            if (bRerunMD)
            {
                bMasterState = TRUE;
            }
            else
            {
                bMasterState = FALSE;
                /* Correct the new box if it is too skewed */
                if (DYNAMIC_BOX(*ir))
                {
                    if (correct_box(fplog, step, state->box, graph))
                    {
                        bMasterState = TRUE;
                    }
                }
                if (DOMAINDECOMP(cr) && bMasterState)
                {
                    dd_collect_state(cr->dd, state, state_global);
                }
            }

            if (DOMAINDECOMP(cr))
            {
                /* Repartition the domain decomposition */
                wallcycle_start(wcycle, ewcDOMDEC);
                dd_partition_system(fplog, step, cr,
                                    bMasterState, nstglobalcomm,
                                    state_global, top_global, ir,
                                    state, &f, mdatoms, top, fr,
                                    vsite, shellfc, constr,
                                    nrnb, wcycle,
                                    do_verbose && !bPMETuneRunning);
                wallcycle_stop(wcycle, ewcDOMDEC);
                /* If using an iterative integrator, reallocate space to match the decomposition */
            }
        }

        if (MASTER(cr) && do_log && !bFFscan)
        {
            print_ebin_header(fplog, step, t, state->lambda[efptFEP]); /* can we improve the information printed here? */
        }

        if (ir->efep != efepNO)
        {
            update_mdatoms(mdatoms, state->lambda[efptMASS]);
        }

        if ((bRerunMD && rerun_fr.bV) || bExchanged)
        {

            /* We need the kinetic energy at minus the half step for determining
             * the full step kinetic energy and possibly for T-coupling.*/
            /* This may not be quite working correctly yet . . . . */
            compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                            wcycle, enerd, NULL, NULL, NULL, NULL, mu_tot,
                            constr, NULL, FALSE, state->box,
                            top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                            CGLO_RERUNMD | CGLO_GSTAT | CGLO_TEMPERATURE);
        }
        clear_mat(force_vir);

        /* Ionize the atoms if necessary */
        if (bIonize)
        {
            ionize(fplog, oenv, mdatoms, top_global, t, ir, state->x, state->v,
                   mdatoms->start, mdatoms->start+mdatoms->homenr, state->box, cr);
        }

        /* Update force field in ffscan program */
        if (bFFscan)
        {
            if (update_forcefield(fplog,
                                  nfile, fnm, fr,
                                  mdatoms->nr, state->x, state->box))
            {
                gmx_finalize_par();

                exit(0);
            }
        }

        GMX_MPE_LOG(ev_timestep2);

        /* We write a checkpoint at this MD step when:
         * either at an NS step when we signalled through gs,
         * or at the last step (but not when we do not want confout),
         * but never at the first step or with rerun.
         */
        bCPT = (((gs.set[eglsCHKPT] && (bNS || ir->nstlist == 0)) ||
                 (bLastStep && (Flags & MD_CONFOUT))) &&
                step > ir->init_step && !bRerunMD);
        if (bCPT)
        {
            gs.set[eglsCHKPT] = 0;
        }

        /* Determine the energy and pressure:
         * at nstcalcenergy steps and at energy output steps (set below).
         */
        if (EI_VV(ir->eI) && (!bInitStep))
        {
            /* for vv, the first half of the integration actually corresponds
               to the previous step.  bCalcEner is only required to be evaluated on the 'next' step,
               but the virial needs to be calculated on both the current step and the 'next' step. Future
               reorganization may be able to get rid of one of the bCalcVir=TRUE steps. */

            bCalcEner = do_per_step(step-1, ir->nstcalcenergy);
            bCalcVir  = bCalcEner ||
                (ir->epc != epcNO && (do_per_step(step, ir->nstpcouple) || do_per_step(step-1, ir->nstpcouple)));
        }
        else
        {
            bCalcEner = do_per_step(step, ir->nstcalcenergy);
            bCalcVir  = bCalcEner ||
                (ir->epc != epcNO && do_per_step(step, ir->nstpcouple));
        }

        /* Do we need global communication ? */
        bGStat = (bCalcVir || bCalcEner || bStopCM ||
                  do_per_step(step, nstglobalcomm) || (bVV && IR_NVT_TROTTER(ir) && do_per_step(step-1, nstglobalcomm)) ||
                  (ir->nstlist == -1 && !bRerunMD && step >= nlh.step_nscheck));

        do_ene = (do_per_step(step, ir->nstenergy) || bLastStep);

        if (do_ene || do_log)
        {
            bCalcVir  = TRUE;
            bCalcEner = TRUE;
            bGStat    = TRUE;
        }

        /* these CGLO_ options remain the same throughout the iteration */
        cglo_flags = ((bRerunMD ? CGLO_RERUNMD : 0) |
                      (bGStat ? CGLO_GSTAT : 0)
                      );

        force_flags = (GMX_FORCE_STATECHANGED |
                       ((DYNAMIC_BOX(*ir) || bRerunMD) ? GMX_FORCE_DYNAMICBOX : 0) |
                       GMX_FORCE_ALLFORCES |
                       GMX_FORCE_SEPLRF |
                       (bCalcVir ? GMX_FORCE_VIRIAL : 0) |
                       (bCalcEner ? GMX_FORCE_ENERGY : 0) |
                       (bDoFEP ? GMX_FORCE_DHDL : 0)
                       );

        if (fr->bTwinRange)
        {
            if (do_per_step(step, ir->nstcalclr))
            {
                force_flags |= GMX_FORCE_DO_LR;
            }
        }

        if (shellfc)
        {
            /* Now is the time to relax the shells */
            count = relax_shell_flexcon(fplog, cr, bVerbose, bFFscan ? step+1 : step,
                                        ir, bNS, force_flags,
                                        bStopCM, top, top_global,
                                        constr, enerd, fcd,
                                        state, f, force_vir, mdatoms,
                                        nrnb, wcycle, graph, groups,
                                        shellfc, fr, bBornRadii, t, mu_tot,
                                        state->natoms, &bConverged, vsite,
                                        outf->fp_field);
            tcount += count;

            if (bConverged)
            {
                nconverged++;
            }
        }
        else
        {
            /* The coordinates (x) are shifted (to get whole molecules)
             * in do_force.
             * This is parallellized as well, and does communication too.
             * Check comments in sim_util.c
             */
/*            for(ii=0;ii<nres;ii++)
            {
             node_ener[ii] = 0.0;
            }*/
#ifdef DO_FLOW
            if(step_rel > 0)
            {
             for(ii=0;ii<nres;ii++)
             {
             // node_ener0[ii] = node_ener[ii];
#ifdef DO_EDGE
              for(jj=0;jj<nres;jj++)
              {
               edge_ener0[ii][jj] = edge_ener[ii][jj];
              }
#endif
             }
            }
            if(PAR(cr))
            {
             //gmx_tx_wait(cr, GMX_LEFT);
             //gmx_rx_wait(cr, GMX_RIGHT);
            //MPI_Waitall(nreq, req, stat);
             MPI_Barrier(MPI_COMM_WORLD);
            }
            for(ii=0;ii<nres;ii++)
            {
             node_ener[ii] = 0.0;
#ifdef DO_EDGE
             for(jj=0;jj<nres;jj++)
             {
              edge_ener[ii][jj] = 0.0;
             }
#endif
            }
            if(PAR(cr))
            {
             //MPI_Barrier(MPI_COMM_WORLD);
            }
#endif

            do_force(fplog, cr, ir, step, nrnb, wcycle, top, top_global, groups,
                     state->box, state->x, &state->hist,
                     f, force_vir, mdatoms, enerd, fcd,
                     state->lambda, graph,
                     fr, vsite, mu_tot, t, outf->fp_field, ed, bBornRadii,
                     (bNS ? GMX_FORCE_NS : 0) | force_flags);
#ifdef DO_FLOW
            if(PAR(cr))
            {
             reset_bin(rb);
             isv = add_binr(rb, nres, node_ener);
             for(ii=0;ii<nres;ii++)
             {
              isp[ii] = add_binr(rb, nres, edge_ener[ii]);
             }
             sum_bin(rb, cr);
             extract_binr(rb, isv, nres, node_ener);
             for(ii=0;ii<nres;ii++)
             {
              extract_binr(rb, isp[ii], nres, edge_ener[ii]);
             }
             //gmx_tx_wait(cr, GMX_LEFT);
             //gmx_rx_wait(cr, GMX_RIGHT);
             //MPI_Barrier(MPI_COMM_WORLD);
            }
            int chosen = 399000;
            //int chosen = 80000;
            for(ii=mdatoms->start;ii<mdatoms->start+mdatoms->homenr;ii++)
            {
             if(ii >= LAST_ATOM)
             {
              break;
             }
              if(ii == 71)
               pot0 = node_ener[ii];
             //node_ener[index_r[ii]] += sqr(norm(state->v[ii]))/(2.0*mdatoms->invmass[ii]);
             //fprintf(stderr,"index %d %f step %d\n",index_r[ii],norm(state->v[ii]),(int)step_rel);
            }
            if(PAR(cr))
            {
             //gmx_tx_wait(cr, GMX_LEFT);
             //gmx_rx_wait(cr, GMX_RIGHT);
            MPI_Waitall(nreq, req, stat);
             //MPI_Barrier(MPI_COMM_WORLD);
            }

            for(ii=0;ii<nres;ii++)
            {
             if(step_rel == 0)
             {
              node_ener0[ii] = node_ener[ii];
              if(ii == 71)
              {
               ene0 = node_ener[ii];
               diff0 = ene0-pot0;
              }
             }
             else
             {
              real tmp = node_ener[ii]-node_ener0[ii];
/*              if(fabs(tmp) > 1 && node_ener0[ii] != 1e9)
              {
               //fprintf(stdout,"node %d time %f\n",ii+301,node_ener[ii]-node_ener0[ii]);
               fprintf(stdout,"node %d step %d\n",ii+301,(int)step_rel);
               node_ener0[ii] = 1e9;
              }*/
              if(step_rel == 95000)
              {
              // fprintf(stdout,"node %d delta %f\n",ii+301,node_ener[ii]-node_ener0[ii]);
              }
             }
            }
            if(PAR(cr))
            {
            // MPI_Barrier(MPI_COMM_WORLD);
            }
            if(MASTER(cr) && step_rel > 0)
            {
             for(ii=0;ii<nres;ii++)
             {
              tmp = node_ener[ii]-node_ener0[ii];
              //if(tmp > 0)
              node_flow[ii] += fabs(tmp);
              //if((step_rel % 20 == 0) && fabs(node_flow[ii]) > 5e-2)
              if((step_rel == 300000))
              {
               //fprintf(stdout,"node %d %d %f %f %f\n",ii,(int)step_rel,node_flow[ii],node_ener[ii],node_ener0[ii]);
               //fprintf(stdout,"node %d %d %f\n",ii,(int)step_rel,node_flow[ii]);
               fprintf(stdout,"node %d %d %f\n",ii,(int)step_rel,tmp);
               //if(ii == 71)
               //fprintf(stdout,"ene0 %d %d %f\n",ii,(int)step_rel,(node_ener[ii]-pot0)/diff0);
              }
#ifdef DO_EDGE
              for(jj=0;jj<nres;jj++)
              {
               tmp = edge_ener[ii][jj]-edge_ener0[ii][jj];
               //if(tmp > 0)
               flow[ii][jj] += fabs(tmp);
             
               if(step_rel == chosen && jj > ii && fabs(flow[ii][jj]) > 5e-2)
               {
                int kk;
                real sum = 0;
                /*for (kk=0;kk<nres;kk++)
                {
                 sum += edge_ener[ii][kk] - edge_ener0[ii][kk];
                }*/
                fprintf(stdout,"fafa %d %d %f %f\n",ii,jj,flow[ii][jj],sum);
               }
                //fprintf(stdout,"%d %d %f\n",ii+1,jj+1,flab[0]);
              }
#endif
             }
            }
            if(step_rel == chosen)
             bLastStep = TRUE;
#endif

            //gmx_grppairener_t *grpp;
            //grpp = enerd->grpp;
            if(step_rel > 0 || 1 == 1)
            {
             nf++;
/*             enerd->grpp.avener[egLJSR][200000] = 10;
             if(MASTER(cr))
             {
              for(ii=0;ii<enerd->grpp.nener;ii++)
              {
               if(enerd->grpp.avener[egLJSR][ii] != 0)
               printf("lala %f %d %d\n",enerd->grpp.avener[egLJSR][ii]/nf,ii,enerd->grpp.nener);
              }
             }*/
            }
        }

        GMX_BARRIER(cr->mpi_comm_mygroup);

        if (bTCR)
        {
            mu_aver = calc_mu_aver(cr, state->x, mdatoms->chargeA,
                                   mu_tot, &top_global->mols, mdatoms, gnx, grpindex);
        }

        if (bTCR && bFirstStep)
        {
            tcr = init_coupling(fplog, nfile, fnm, cr, fr, mdatoms, &(top->idef));
            fprintf(fplog, "Done init_coupling\n");
            fflush(fplog);
        }

        if (bVV && !bStartingFromCpt && !bRerunMD)
        /*  ############### START FIRST UPDATE HALF-STEP FOR VV METHODS############### */
        {
            if (ir->eI == eiVV && bInitStep)
            {
                /* if using velocity verlet with full time step Ekin,
                 * take the first half step only to compute the
                 * virial for the first step. From there,
                 * revert back to the initial coordinates
                 * so that the input is actually the initial step.
                 */
                copy_rvecn(state->v, cbuf, 0, state->natoms); /* should make this better for parallelizing? */
            }
            else
            {
                /* this is for NHC in the Ekin(t+dt/2) version of vv */
                trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ1);
            }

            /* If we are using twin-range interactions where the long-range component
             * is only evaluated every nstcalclr>1 steps, we should do a special update
             * step to combine the long-range forces on these steps.
             * For nstcalclr=1 this is not done, since the forces would have been added
             * directly to the short-range forces already.
             */
            bUpdateDoLR = (fr->bTwinRange && do_per_step(step, ir->nstcalclr));

            update_coords(fplog, step, ir, mdatoms, state, fr->bMolPBC,
                          f, bUpdateDoLR, fr->f_twin, fcd,
                          ekind, M, wcycle, upd, bInitStep, etrtVELOCITY1,
                          cr, nrnb, constr, &top->idef);

            if (bIterativeCase && do_per_step(step-1, ir->nstpcouple) && !bInitStep)
            {
                gmx_iterate_init(&iterate, TRUE);
            }
            /* for iterations, we save these vectors, as we will be self-consistently iterating
               the calculations */

            /*#### UPDATE EXTENDED VARIABLES IN TROTTER FORMULATION */

            /* save the state */
            if (iterate.bIterationActive)
            {
                copy_coupling_state(state, bufstate, ekind, ekind_save, &(ir->opts));
            }

            bFirstIterate = TRUE;
            while (bFirstIterate || iterate.bIterationActive)
            {
                if (iterate.bIterationActive)
                {
                    copy_coupling_state(bufstate, state, ekind_save, ekind, &(ir->opts));
                    if (bFirstIterate && bTrotter)
                    {
                        /* The first time through, we need a decent first estimate
                           of veta(t+dt) to compute the constraints.  Do
                           this by computing the box volume part of the
                           trotter integration at this time. Nothing else
                           should be changed by this routine here.  If
                           !(first time), we start with the previous value
                           of veta.  */

                        veta_save = state->veta;
                        trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ0);
                        vetanew     = state->veta;
                        state->veta = veta_save;
                    }
                }

                bOK = TRUE;
                if (!bRerunMD || rerun_fr.bV || bForceUpdate)     /* Why is rerun_fr.bV here?  Unclear. */
                {
                    update_constraints(fplog, step, NULL, ir, ekind, mdatoms,
                                       state, fr->bMolPBC, graph, f,
                                       &top->idef, shake_vir, NULL,
                                       cr, nrnb, wcycle, upd, constr,
                                       bInitStep, TRUE, bCalcVir, vetanew);

                    if (!bOK && !bFFscan)
                    {
                        gmx_fatal(FARGS, "Constraint error: Shake, Lincs or Settle could not solve the constrains");
                    }

                }
                else if (graph)
                {
                    /* Need to unshift here if a do_force has been
                       called in the previous step */
                    unshift_self(graph, state->box, state->x);
                }

                /* if VV, compute the pressure and constraints */
                /* For VV2, we strictly only need this if using pressure
                 * control, but we really would like to have accurate pressures
                 * printed out.
                 * Think about ways around this in the future?
                 * For now, keep this choice in comments.
                 */
                /*bPres = (ir->eI==eiVV || IR_NPT_TROTTER(ir)); */
                /*bTemp = ((ir->eI==eiVV &&(!bInitStep)) || (ir->eI==eiVVAK && IR_NPT_TROTTER(ir)));*/
                bPres = TRUE;
                bTemp = ((ir->eI == eiVV && (!bInitStep)) || (ir->eI == eiVVAK));
                if (bCalcEner && ir->eI == eiVVAK)  /*MRS:  7/9/2010 -- this still doesn't fix it?*/
                {
                    bSumEkinhOld = TRUE;
                }
                /* for vv, the first half of the integration actually corresponds to the previous step.
                   So we need information from the last step in the first half of the integration */
                if (bGStat || do_per_step(step-1, nstglobalcomm))
                {
                    compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                                    wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                    constr, NULL, FALSE, state->box,
                                    top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                                    cglo_flags
                                    | CGLO_ENERGY
                                    | (bTemp ? CGLO_TEMPERATURE : 0)
                                    | (bPres ? CGLO_PRESSURE : 0)
                                    | (bPres ? CGLO_CONSTRAINT : 0)
                                    | ((iterate.bIterationActive) ? CGLO_ITERATE : 0)
                                    | (bFirstIterate ? CGLO_FIRSTITERATE : 0)
                                    | CGLO_SCALEEKIN
                                    );
                    /* explanation of above:
                       a) We compute Ekin at the full time step
                       if 1) we are using the AveVel Ekin, and it's not the
                       initial step, or 2) if we are using AveEkin, but need the full
                       time step kinetic energy for the pressure (always true now, since we want accurate statistics).
                       b) If we are using EkinAveEkin for the kinetic energy for the temperature control, we still feed in
                       EkinAveVel because it's needed for the pressure */
                }
                /* temperature scaling and pressure scaling to produce the extended variables at t+dt */
                if (!bInitStep)
                {
                    if (bTrotter)
                    {
                        m_add(force_vir, shake_vir, total_vir); /* we need the un-dispersion corrected total vir here */
                        trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ2);
                    }
                    else
                    {
                        if (bExchanged)
                        {

                            /* We need the kinetic energy at minus the half step for determining
                             * the full step kinetic energy and possibly for T-coupling.*/
                            /* This may not be quite working correctly yet . . . . */
                            compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                                            wcycle, enerd, NULL, NULL, NULL, NULL, mu_tot,
                                            constr, NULL, FALSE, state->box,
                                            top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                                            CGLO_RERUNMD | CGLO_GSTAT | CGLO_TEMPERATURE);
                        }
                    }
                }

                if (iterate.bIterationActive &&
                    done_iterating(cr, fplog, step, &iterate, bFirstIterate,
                                   state->veta, &vetanew))
                {
                    break;
                }
                bFirstIterate = FALSE;
            }

            if (bTrotter && !bInitStep)
            {
                copy_mat(shake_vir, state->svir_prev);
                copy_mat(force_vir, state->fvir_prev);
                if (IR_NVT_TROTTER(ir) && ir->eI == eiVV)
                {
                    /* update temperature and kinetic energy now that step is over - this is the v(t+dt) point */
                    enerd->term[F_TEMP] = sum_ekin(&(ir->opts), ekind, NULL, (ir->eI == eiVV), FALSE, FALSE);
                    enerd->term[F_EKIN] = trace(ekind->ekin);
                }
            }
            /* if it's the initial step, we performed this first step just to get the constraint virial */
            if (bInitStep && ir->eI == eiVV)
            {
                copy_rvecn(cbuf, state->v, 0, state->natoms);
            }

            GMX_MPE_LOG(ev_timestep1);
        }

        /* MRS -- now done iterating -- compute the conserved quantity */
        if (bVV)
        {
            saved_conserved_quantity = compute_conserved_from_auxiliary(ir, state, &MassQ);
            if (ir->eI == eiVV)
            {
                last_ekin = enerd->term[F_EKIN];
            }
            if ((ir->eDispCorr != edispcEnerPres) && (ir->eDispCorr != edispcAllEnerPres))
            {
                saved_conserved_quantity -= enerd->term[F_DISPCORR];
            }
            /* sum up the foreign energy and dhdl terms for vv.  currently done every step so that dhdl is correct in the .edr */
            if (!bRerunMD)
            {
                sum_dhdl(enerd, state->lambda, ir->fepvals);
            }
        }

        /* ########  END FIRST UPDATE STEP  ############## */
        /* ########  If doing VV, we now have v(dt) ###### */
        if (bDoExpanded)
        {
            /* perform extended ensemble sampling in lambda - we don't
               actually move to the new state before outputting
               statistics, but if performing simulated tempering, we
               do update the velocities and the tau_t. */

            lamnew = ExpandedEnsembleDynamics(fplog, ir, enerd, state, &MassQ, &df_history, step, mcrng, state->v, mdatoms);
        }
        /* ################## START TRAJECTORY OUTPUT ################# */

        /* Now we have the energies and forces corresponding to the
         * coordinates at time t. We must output all of this before
         * the update.
         * for RerunMD t is read from input trajectory
         */
        GMX_MPE_LOG(ev_output_start);

        mdof_flags = 0;
        if (do_per_step(step, ir->nstxout))
        {
            mdof_flags |= MDOF_X;
        }
        if (do_per_step(step, ir->nstvout))
        {
            mdof_flags |= MDOF_V;
        }
        if (do_per_step(step, ir->nstfout))
        {
            mdof_flags |= MDOF_F;
        }
        if (do_per_step(step, ir->nstxtcout))
        {
            mdof_flags |= MDOF_XTC;
        }
        if (bCPT)
        {
            mdof_flags |= MDOF_CPT;
        }
        ;

#if defined(GMX_FAHCORE) || defined(GMX_WRITELASTSTEP)
        if (bLastStep)
        {
            /* Enforce writing positions and velocities at end of run */
            mdof_flags |= (MDOF_X | MDOF_V);
        }
#endif
#ifdef GMX_FAHCORE
        if (MASTER(cr))
        {
            fcReportProgress( ir->nsteps, step );
        }

        /* sync bCPT and fc record-keeping */
        if (bCPT && MASTER(cr))
        {
            fcRequestCheckPoint();
        }
#endif

        if (mdof_flags != 0)
        {
            wallcycle_start(wcycle, ewcTRAJ);
            if (bCPT)
            {
                if (state->flags & (1<<estLD_RNG))
                {
                    get_stochd_state(upd, state);
                }
                if (state->flags  & (1<<estMC_RNG))
                {
                    get_mc_state(mcrng, state);
                }
                if (MASTER(cr))
                {
                    if (bSumEkinhOld)
                    {
                        state_global->ekinstate.bUpToDate = FALSE;
                    }
                    else
                    {
                        update_ekinstate(&state_global->ekinstate, ekind);
                        state_global->ekinstate.bUpToDate = TRUE;
                    }
                    update_energyhistory(&state_global->enerhist, mdebin);
                    if (ir->efep != efepNO || ir->bSimTemp)
                    {
                        state_global->fep_state = state->fep_state; /* MRS: seems kludgy. The code should be
                                                                       structured so this isn't necessary.
                                                                       Note this reassignment is only necessary
                                                                       for single threads.*/
                        copy_df_history(&state_global->dfhist, &df_history);
                    }
                }
            }
            write_traj(fplog, cr, outf, mdof_flags, top_global,
                       step, t, state, state_global, f, f_global, &n_xtc, &x_xtc);
            if (bCPT)
            {
                nchkpt++;
                bCPT = FALSE;
            }
            debug_gmx();
            if (bLastStep && step_rel == ir->nsteps &&
                (Flags & MD_CONFOUT) && MASTER(cr) &&
                !bRerunMD && !bFFscan)
            {
                /* x and v have been collected in write_traj,
                 * because a checkpoint file will always be written
                 * at the last step.
                 */
                fprintf(stderr, "\nWriting final coordinates.\n");
                if (fr->bMolPBC)
                {
                    /* Make molecules whole only for confout writing */
                    do_pbc_mtop(fplog, ir->ePBC, state->box, top_global, state_global->x);
                }
                write_sto_conf_mtop(ftp2fn(efSTO, nfile, fnm),
                                    *top_global->name, top_global,
                                    state_global->x, state_global->v,
                                    ir->ePBC, state->box);
                debug_gmx();
            }
            wallcycle_stop(wcycle, ewcTRAJ);
        }
        GMX_MPE_LOG(ev_output_finish);

        /* kludge -- virial is lost with restart for NPT control. Must restart */
        if (bStartingFromCpt && bVV)
        {
            copy_mat(state->svir_prev, shake_vir);
            copy_mat(state->fvir_prev, force_vir);
        }
        /*  ################## END TRAJECTORY OUTPUT ################ */

        /* Determine the wallclock run time up till now */
        run_time = gmx_gettime() - (double)runtime->real;

        /* Check whether everything is still allright */
        if (((int)gmx_get_stop_condition() > handled_stop_condition)
#ifdef GMX_THREAD_MPI
            && MASTER(cr)
#endif
            )
        {
            /* this is just make gs.sig compatible with the hack
               of sending signals around by MPI_Reduce with together with
               other floats */
            if (gmx_get_stop_condition() == gmx_stop_cond_next_ns)
            {
                gs.sig[eglsSTOPCOND] = 1;
            }
            if (gmx_get_stop_condition() == gmx_stop_cond_next)
            {
                gs.sig[eglsSTOPCOND] = -1;
            }
            /* < 0 means stop at next step, > 0 means stop at next NS step */
            if (fplog)
            {
                fprintf(fplog,
                        "\n\nReceived the %s signal, stopping at the next %sstep\n\n",
                        gmx_get_signal_name(),
                        gs.sig[eglsSTOPCOND] == 1 ? "NS " : "");
                fflush(fplog);
            }
            fprintf(stderr,
                    "\n\nReceived the %s signal, stopping at the next %sstep\n\n",
                    gmx_get_signal_name(),
                    gs.sig[eglsSTOPCOND] == 1 ? "NS " : "");
            fflush(stderr);
            handled_stop_condition = (int)gmx_get_stop_condition();
        }
        else if (MASTER(cr) && (bNS || ir->nstlist <= 0) &&
                 (max_hours > 0 && run_time > max_hours*60.0*60.0*0.99) &&
                 gs.sig[eglsSTOPCOND] == 0 && gs.set[eglsSTOPCOND] == 0)
        {
            /* Signal to terminate the run */
            gs.sig[eglsSTOPCOND] = 1;
            if (fplog)
            {
                fprintf(fplog, "\nStep %s: Run time exceeded %.3f hours, will terminate the run\n", gmx_step_str(step, sbuf), max_hours*0.99);
            }
            fprintf(stderr, "\nStep %s: Run time exceeded %.3f hours, will terminate the run\n", gmx_step_str(step, sbuf), max_hours*0.99);
        }

        if (bResetCountersHalfMaxH && MASTER(cr) &&
            run_time > max_hours*60.0*60.0*0.495)
        {
            gs.sig[eglsRESETCOUNTERS] = 1;
        }

        if (ir->nstlist == -1 && !bRerunMD)
        {
            /* When bGStatEveryStep=FALSE, global_stat is only called
             * when we check the atom displacements, not at NS steps.
             * This means that also the bonded interaction count check is not
             * performed immediately after NS. Therefore a few MD steps could
             * be performed with missing interactions.
             * But wrong energies are never written to file,
             * since energies are only written after global_stat
             * has been called.
             */
            if (step >= nlh.step_nscheck)
            {
                nlh.nabnsb = natoms_beyond_ns_buffer(ir, fr, &top->cgs,
                                                     nlh.scale_tot, state->x);
            }
            else
            {
                /* This is not necessarily true,
                 * but step_nscheck is determined quite conservatively.
                 */
                nlh.nabnsb = 0;
            }
        }

        /* In parallel we only have to check for checkpointing in steps
         * where we do global communication,
         *  otherwise the other nodes don't know.
         */
        if (MASTER(cr) && ((bGStat || !PAR(cr)) &&
                           cpt_period >= 0 &&
                           (cpt_period == 0 ||
                            run_time >= nchkpt*cpt_period*60.0)) &&
            gs.set[eglsCHKPT] == 0)
        {
            gs.sig[eglsCHKPT] = 1;
        }

        /* at the start of step, randomize or scale the velocities (trotter done elsewhere) */
        if (EI_VV(ir->eI))
        {
            if (!bInitStep)
            {
                update_tcouple(fplog, step, ir, state, ekind, wcycle, upd, &MassQ, mdatoms);
            }
            if (ETC_ANDERSEN(ir->etc)) /* keep this outside of update_tcouple because of the extra info required to pass */
            {
                gmx_bool bIfRandomize;
                bIfRandomize = update_randomize_velocities(ir, step, mdatoms, state, upd, &top->idef, constr);
                /* if we have constraints, we have to remove the kinetic energy parallel to the bonds */
                if (constr && bIfRandomize)
                {
                    update_constraints(fplog, step, NULL, ir, ekind, mdatoms,
                                       state, fr->bMolPBC, graph, f,
                                       &top->idef, tmp_vir, NULL,
                                       cr, nrnb, wcycle, upd, constr,
                                       bInitStep, TRUE, bCalcVir, vetanew);
                }
            }
        }

        if (bIterativeCase && do_per_step(step, ir->nstpcouple))
        {
            gmx_iterate_init(&iterate, TRUE);
            /* for iterations, we save these vectors, as we will be redoing the calculations */
            copy_coupling_state(state, bufstate, ekind, ekind_save, &(ir->opts));
        }

        bFirstIterate = TRUE;
        while (bFirstIterate || iterate.bIterationActive)
        {
            /* We now restore these vectors to redo the calculation with improved extended variables */
            if (iterate.bIterationActive)
            {
                copy_coupling_state(bufstate, state, ekind_save, ekind, &(ir->opts));
            }

            /* We make the decision to break or not -after- the calculation of Ekin and Pressure,
               so scroll down for that logic */

            /* #########   START SECOND UPDATE STEP ################# */
            GMX_MPE_LOG(ev_update_start);
            /* Box is changed in update() when we do pressure coupling,
             * but we should still use the old box for energy corrections and when
             * writing it to the energy file, so it matches the trajectory files for
             * the same timestep above. Make a copy in a separate array.
             */
            copy_mat(state->box, lastbox);

            bOK = TRUE;
            dvdl_constr = 0;

            if (!(bRerunMD && !rerun_fr.bV && !bForceUpdate))
            {
                wallcycle_start(wcycle, ewcUPDATE);
                /* UPDATE PRESSURE VARIABLES IN TROTTER FORMULATION WITH CONSTRAINTS */
                if (bTrotter)
                {
                    if (iterate.bIterationActive)
                    {
                        if (bFirstIterate)
                        {
                            scalevir = 1;
                        }
                        else
                        {
                            /* we use a new value of scalevir to converge the iterations faster */
                            scalevir = tracevir/trace(shake_vir);
                        }
                        msmul(shake_vir, scalevir, shake_vir);
                        m_add(force_vir, shake_vir, total_vir);
                        clear_mat(shake_vir);
                    }
                    trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ3);
                    /* We can only do Berendsen coupling after we have summed
                     * the kinetic energy or virial. Since the happens
                     * in global_state after update, we should only do it at
                     * step % nstlist = 1 with bGStatEveryStep=FALSE.
                     */
                }
                else
                {
                    update_tcouple(fplog, step, ir, state, ekind, wcycle, upd, &MassQ, mdatoms);
                    update_pcouple(fplog, step, ir, state, pcoupl_mu, M, wcycle,
                                   upd, bInitStep);
                }

                if (bVV)
                {
                    bUpdateDoLR = (fr->bTwinRange && do_per_step(step, ir->nstcalclr));

                    /* velocity half-step update */
                    update_coords(fplog, step, ir, mdatoms, state, fr->bMolPBC, f,
                                  bUpdateDoLR, fr->f_twin, fcd,
                                  ekind, M, wcycle, upd, FALSE, etrtVELOCITY2,
                                  cr, nrnb, constr, &top->idef);
                }

                /* Above, initialize just copies ekinh into ekin,
                 * it doesn't copy position (for VV),
                 * and entire integrator for MD.
                 */

                if (ir->eI == eiVVAK)
                {
                    copy_rvecn(state->x, cbuf, 0, state->natoms);
                }
                bUpdateDoLR = (fr->bTwinRange && do_per_step(step, ir->nstcalclr));

                update_coords(fplog, step, ir, mdatoms, state, fr->bMolPBC, f,
                              bUpdateDoLR, fr->f_twin, fcd,
                              ekind, M, wcycle, upd, bInitStep, etrtPOSITION, cr, nrnb, constr, &top->idef);
                wallcycle_stop(wcycle, ewcUPDATE);

                update_constraints(fplog, step, &dvdl_constr, ir, ekind, mdatoms, state,
                                   fr->bMolPBC, graph, f,
                                   &top->idef, shake_vir, force_vir,
                                   cr, nrnb, wcycle, upd, constr,
                                   bInitStep, FALSE, bCalcVir, state->veta);

                if (ir->eI == eiVVAK)
                {
                    /* erase F_EKIN and F_TEMP here? */
                    /* just compute the kinetic energy at the half step to perform a trotter step */
                    compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                                    wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                    constr, NULL, FALSE, lastbox,
                                    top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                                    cglo_flags | CGLO_TEMPERATURE
                                    );
                    wallcycle_start(wcycle, ewcUPDATE);
                    trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ4);
                    /* now we know the scaling, we can compute the positions again again */
                    copy_rvecn(cbuf, state->x, 0, state->natoms);

                    bUpdateDoLR = (fr->bTwinRange && do_per_step(step, ir->nstcalclr));

                    update_coords(fplog, step, ir, mdatoms, state, fr->bMolPBC, f,
                                  bUpdateDoLR, fr->f_twin, fcd,
                                  ekind, M, wcycle, upd, bInitStep, etrtPOSITION, cr, nrnb, constr, &top->idef);
                    wallcycle_stop(wcycle, ewcUPDATE);

                    /* do we need an extra constraint here? just need to copy out of state->v to upd->xp? */
                    /* are the small terms in the shake_vir here due
                     * to numerical errors, or are they important
                     * physically? I'm thinking they are just errors, but not completely sure.
                     * For now, will call without actually constraining, constr=NULL*/
                    update_constraints(fplog, step, NULL, ir, ekind, mdatoms,
                                       state, fr->bMolPBC, graph, f,
                                       &top->idef, tmp_vir, force_vir,
                                       cr, nrnb, wcycle, upd, NULL,
                                       bInitStep, FALSE, bCalcVir,
                                       state->veta);
                }
                if (!bOK && !bFFscan)
                {
                    gmx_fatal(FARGS, "Constraint error: Shake, Lincs or Settle could not solve the constrains");
                }

                if (fr->bSepDVDL && fplog && do_log)
                {
                    fprintf(fplog, sepdvdlformat, "Constraint dV/dl", 0.0, dvdl_constr);
                }
                if (bVV)
                {
                    /* this factor or 2 correction is necessary
                       because half of the constraint force is removed
                       in the vv step, so we have to double it.  See
                       the Redmine issue #1255.  It is not yet clear
                       if the factor of 2 is exact, or just a very
                       good approximation, and this will be
                       investigated.  The next step is to see if this
                       can be done adding a dhdl contribution from the
                       rattle step, but this is somewhat more
                       complicated with the current code. Will be
                       investigated, hopefully for 4.6.3. However,
                       this current solution is much better than
                       having it completely wrong.
                    */
                    enerd->term[F_DVDL_CONSTR] += 2*dvdl_constr;
                }
                else
                {
                    enerd->term[F_DVDL_CONSTR] += dvdl_constr;
                }
            }
            else if (graph)
            {
                /* Need to unshift here */
                unshift_self(graph, state->box, state->x);
            }

            GMX_BARRIER(cr->mpi_comm_mygroup);
            GMX_MPE_LOG(ev_update_finish);

            if (vsite != NULL)
            {
                wallcycle_start(wcycle, ewcVSITECONSTR);
                if (graph != NULL)
                {
                    shift_self(graph, state->box, state->x);
                }
                construct_vsites(fplog, vsite, state->x, nrnb, ir->delta_t, state->v,
                                 top->idef.iparams, top->idef.il,
                                 fr->ePBC, fr->bMolPBC, graph, cr, state->box);

                if (graph != NULL)
                {
                    unshift_self(graph, state->box, state->x);
                }
                wallcycle_stop(wcycle, ewcVSITECONSTR);
            }

            /* ############## IF NOT VV, Calculate globals HERE, also iterate constraints  ############ */
            /* With Leap-Frog we can skip compute_globals at
             * non-communication steps, but we need to calculate
             * the kinetic energy one step before communication.
             */
            if (bGStat || (!EI_VV(ir->eI) && do_per_step(step+1, nstglobalcomm)))
            {
                if (ir->nstlist == -1 && bFirstIterate)
                {
                    gs.sig[eglsNABNSB] = nlh.nabnsb;
                }
                compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                                wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                constr,
                                bFirstIterate ? &gs : NULL,
                                (step_rel % gs.nstms == 0) &&
                                (multisim_nsteps < 0 || (step_rel < multisim_nsteps)),
                                lastbox,
                                top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                                cglo_flags
                                | (!EI_VV(ir->eI) || bRerunMD ? CGLO_ENERGY : 0)
                                | (!EI_VV(ir->eI) && bStopCM ? CGLO_STOPCM : 0)
                                | (!EI_VV(ir->eI) ? CGLO_TEMPERATURE : 0)
                                | (!EI_VV(ir->eI) || bRerunMD ? CGLO_PRESSURE : 0)
                                | (iterate.bIterationActive ? CGLO_ITERATE : 0)
                                | (bFirstIterate ? CGLO_FIRSTITERATE : 0)
                                | CGLO_CONSTRAINT
                                );
                if (ir->nstlist == -1 && bFirstIterate)
                {
                    nlh.nabnsb         = gs.set[eglsNABNSB];
                    gs.set[eglsNABNSB] = 0;
                }
            }
            /* bIterate is set to keep it from eliminating the old ekin kinetic energy terms */
            /* #############  END CALC EKIN AND PRESSURE ################# */

            /* Note: this is OK, but there are some numerical precision issues with using the convergence of
               the virial that should probably be addressed eventually. state->veta has better properies,
               but what we actually need entering the new cycle is the new shake_vir value. Ideally, we could
               generate the new shake_vir, but test the veta value for convergence.  This will take some thought. */

            if (iterate.bIterationActive &&
                done_iterating(cr, fplog, step, &iterate, bFirstIterate,
                               trace(shake_vir), &tracevir))
            {
                break;
            }
            bFirstIterate = FALSE;
        }

        if (!bVV || bRerunMD)
        {
            /* sum up the foreign energy and dhdl terms for md and sd. currently done every step so that dhdl is correct in the .edr */
            sum_dhdl(enerd, state->lambda, ir->fepvals);
        }
        update_box(fplog, step, ir, mdatoms, state, graph, f,
                   ir->nstlist == -1 ? &nlh.scale_tot : NULL, pcoupl_mu, nrnb, wcycle, upd, bInitStep, FALSE);

        /* ################# END UPDATE STEP 2 ################# */
        /* #### We now have r(t+dt) and v(t+dt/2)  ############# */

        /* The coordinates (x) were unshifted in update */
        if (bFFscan && (shellfc == NULL || bConverged))
        {
            if (print_forcefield(fplog, enerd->term, mdatoms->homenr,
                                 f, NULL, xcopy,
                                 &(top_global->mols), mdatoms->massT, pres))
            {
                gmx_finalize_par();

                fprintf(stderr, "\n");
                exit(0);
            }
        }
        if (!bGStat)
        {
            /* We will not sum ekinh_old,
             * so signal that we still have to do it.
             */
            bSumEkinhOld = TRUE;
        }

        if (bTCR)
        {
            /* Only do GCT when the relaxation of shells (minimization) has converged,
             * otherwise we might be coupling to bogus energies.
             * In parallel we must always do this, because the other sims might
             * update the FF.
             */

            /* Since this is called with the new coordinates state->x, I assume
             * we want the new box state->box too. / EL 20040121
             */
            do_coupling(fplog, oenv, nfile, fnm, tcr, t, step, enerd->term, fr,
                        ir, MASTER(cr),
                        mdatoms, &(top->idef), mu_aver,
                        top_global->mols.nr, cr,
                        state->box, total_vir, pres,
                        mu_tot, state->x, f, bConverged);
            debug_gmx();
        }

        /* #########  BEGIN PREPARING EDR OUTPUT  ###########  */

        /* use the directly determined last velocity, not actually the averaged half steps */
        if (bTrotter && ir->eI == eiVV)
        {
            enerd->term[F_EKIN] = last_ekin;
        }
        enerd->term[F_ETOT] = enerd->term[F_EPOT] + enerd->term[F_EKIN];

        if (bVV)
        {
            enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + saved_conserved_quantity;
        }
        else
        {
            enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + compute_conserved_from_auxiliary(ir, state, &MassQ);
        }
        /* Check for excessively large energies */
        if (bIonize)
        {
#ifdef GMX_DOUBLE
            real etot_max = 1e200;
#else
            real etot_max = 1e30;
#endif
            if (fabs(enerd->term[F_ETOT]) > etot_max)
            {
                fprintf(stderr, "Energy too large (%g), giving up\n",
                        enerd->term[F_ETOT]);
            }
        }
        /* #########  END PREPARING EDR OUTPUT  ###########  */

        /* Time for performance */
        if (((step % stepout) == 0) || bLastStep)
        {
            runtime_upd_proc(runtime);
        }

        /* Output stuff */
        if (MASTER(cr))
        {
            gmx_bool do_dr, do_or;

            if (fplog && do_log && bDoExpanded)
            {
                /* only needed if doing expanded ensemble */
                PrintFreeEnergyInfoToFile(fplog, ir->fepvals, ir->expandedvals, ir->bSimTemp ? ir->simtempvals : NULL,
                                          &df_history, state->fep_state, ir->nstlog, step);
            }
            if (!(bStartingFromCpt && (EI_VV(ir->eI))))
            {
                if (bCalcEner)
                {
                 if(!mdebin->doAvEner)
                 {
                    upd_mdebin(mdebin, bDoDHDL, TRUE,
                               t, mdatoms->tmass, enerd, state,
                               ir->fepvals, ir->expandedvals, lastbox,
                               shake_vir, force_vir, total_vir, pres,
                               ekind, mu_tot, constr);
                 }
                 else
                 {
                  mdebin->ebin->nsum_sim++;
                 }
                }
                else
                {
                    upd_mdebin_step(mdebin);
                }

                do_dr  = do_per_step(step, ir->nstdisreout);
                do_or  = do_per_step(step, ir->nstorireout);

                if(!mdebin->doAvEner)
                {
                 print_ebin(outf->fp_ene, do_ene, do_dr, do_or, do_log ? fplog : NULL,
                           step, t,
                           eprNORMAL, bCompact, mdebin, fcd, groups, &(ir->opts));
                }
            }
            if (ir->ePull != epullNO)
            {
                pull_print_output(ir->pull, step, t);
            }

            if (do_per_step(step, ir->nstlog))
            {
                if (fflush(fplog) != 0)
                {
                    gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
                }
            }
        }
        if (bDoExpanded)
        {
            /* Have to do this part after outputting the logfile and the edr file */
            state->fep_state = lamnew;
            for (i = 0; i < efptNR; i++)
            {
                state_global->lambda[i] = ir->fepvals->all_lambda[i][lamnew];
            }
        }
        /* Remaining runtime */
        if (MULTIMASTER(cr) && (do_verbose || gmx_got_usr_signal()) && !bPMETuneRunning)
        {
            if (shellfc)
            {
                fprintf(stderr, "\n");
            }
            print_time(stderr, runtime, step, ir, cr);
        }

        /* Replica exchange */
        bExchanged = FALSE;
        if ((repl_ex_nst > 0) && (step > 0) && !bLastStep &&
            do_per_step(step, repl_ex_nst))
        {
            bExchanged = replica_exchange(fplog, cr, repl_ex,
                                          state_global, enerd,
                                          state, step, t);

            if (bExchanged && DOMAINDECOMP(cr))
            {
                dd_partition_system(fplog, step, cr, TRUE, 1,
                                    state_global, top_global, ir,
                                    state, &f, mdatoms, top, fr,
                                    vsite, shellfc, constr,
                                    nrnb, wcycle, FALSE);
            }
        }

        bFirstStep       = FALSE;
        bInitStep        = FALSE;
        bStartingFromCpt = FALSE;

        /* #######  SET VARIABLES FOR NEXT ITERATION IF THEY STILL NEED IT ###### */
        /* With all integrators, except VV, we need to retain the pressure
         * at the current step for coupling at the next step.
         */
        if ((state->flags & (1<<estPRES_PREV)) &&
            (bGStatEveryStep ||
             (ir->nstpcouple > 0 && step % ir->nstpcouple == 0)))
        {
            /* Store the pressure in t_state for pressure coupling
             * at the next MD step.
             */
            copy_mat(pres, state->pres_prev);
        }

        /* #######  END SET VARIABLES FOR NEXT ITERATION ###### */

        if ( (membed != NULL) && (!bLastStep) )
        {
            rescale_membed(step_rel, membed, state_global->x);
        }

        if (bRerunMD)
        {
            if (MASTER(cr))
            {
                /* read next frame from input trajectory */
                bNotLastFrame = read_next_frame(oenv, status, &rerun_fr);
            }

            if (PAR(cr))
            {
                rerun_parallel_comm(cr, &rerun_fr, &bNotLastFrame);
            }
        }
        //if(step_rel > quitframe)
        // bNotLastFrame = FALSE;

        if (!bRerunMD || !rerun_fr.bStep)
        {
            /* increase the MD step number */
            step++;
            step_rel++;
        }

        cycles = wallcycle_stop(wcycle, ewcSTEP);
        if (DOMAINDECOMP(cr) && wcycle)
        {
            dd_cycles_add(cr->dd, cycles, ddCyclStep);
        }

        if (bPMETuneRunning || bPMETuneTry)
        {
            /* PME grid + cut-off optimization with GPUs or PME nodes */

            /* Count the total cycles over the last steps */
            cycles_pmes += cycles;

            /* We can only switch cut-off at NS steps */
            if (step % ir->nstlist == 0)
            {
                /* PME grid + cut-off optimization with GPUs or PME nodes */
                if (bPMETuneTry)
                {
                    if (DDMASTER(cr->dd))
                    {
                        /* PME node load is too high, start tuning */
                        bPMETuneRunning = (dd_pme_f_ratio(cr->dd) >= 1.05);
                    }
                    dd_bcast(cr->dd, sizeof(gmx_bool), &bPMETuneRunning);

                    if (bPMETuneRunning || step_rel > ir->nstlist*50)
                    {
                        bPMETuneTry     = FALSE;
                    }
                }
                if (bPMETuneRunning)
                {
                    /* init_step might not be a multiple of nstlist,
                     * but the first cycle is always skipped anyhow.
                     */
                    bPMETuneRunning =
                        pme_load_balance(pme_loadbal, cr,
                                         (bVerbose && MASTER(cr)) ? stderr : NULL,
                                         fplog,
                                         ir, state, cycles_pmes,
                                         fr->ic, fr->nbv, &fr->pmedata,
                                         step);

                    /* Update constants in forcerec/inputrec to keep them in sync with fr->ic */
                    fr->ewaldcoeff = fr->ic->ewaldcoeff;
                    fr->rlist      = fr->ic->rlist;
                    fr->rlistlong  = fr->ic->rlistlong;
                    fr->rcoulomb   = fr->ic->rcoulomb;
                    fr->rvdw       = fr->ic->rvdw;
                }
                cycles_pmes = 0;
            }
        }

        if (step_rel == wcycle_get_reset_counters(wcycle) ||
            gs.set[eglsRESETCOUNTERS] != 0)
        {
            /* Reset all the counters related to performance over the run */
            reset_all_counters(fplog, cr, step, &step_rel, ir, wcycle, nrnb, runtime,
                               fr->nbv != NULL && fr->nbv->bUseGPU ? fr->nbv->cu_nbv : NULL);
            wcycle_set_reset_counters(wcycle, -1);
            if (!(cr->duty & DUTY_PME))
            {
                /* Tell our PME node to reset its counters */
                gmx_pme_send_resetcounters(cr, step);
            }
            /* Correct max_hours for the elapsed time */
            max_hours                -= run_time/(60.0*60.0);
            bResetCountersHalfMaxH    = FALSE;
            gs.set[eglsRESETCOUNTERS] = 0;
        }

    }
    /* End of main MD loop */
    debug_gmx();

    /* Stop the time */
    runtime_end(runtime);

    if (bRerunMD && MASTER(cr))
    {
        close_trj(status);
    }

    if (!(cr->duty & DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr);
    }

    if (MASTER(cr))
    {
        if (ir->nstcalcenergy > 0 && !bRerunMD)
        {
            print_ebin(outf->fp_ene, FALSE, FALSE, FALSE, fplog, step, t,
                       eprAVER, FALSE, mdebin, fcd, groups, &(ir->opts));
        }
        if(mdebin->doAvEner)
        {
                    upd_mdebin(mdebin, bDoDHDL, TRUE,
                               t, mdatoms->tmass, enerd, state,
                               ir->fepvals, ir->expandedvals, lastbox,
                               shake_vir, force_vir, total_vir, pres,
                               ekind, mu_tot, constr);
                 print_ebin(outf->fp_ene, TRUE, FALSE, FALSE, do_log ? fplog : NULL,
                           step, t,
                           eprNORMAL, bCompact, mdebin, fcd, groups, &(ir->opts));
        }
     gmx_enxnm_t *enm = mdebin->ebin->enm;
     real avener;
     /*for(ii=0;ii<enerd->grpp.nener;ii++)
     {
      printf("lala %f %d\n",enerd->grpp.avener[egLJSR][1]/nf,nf);
     }*/
    }

    done_mdoutf(outf);


    debug_gmx();

    if (ir->nstlist == -1 && nlh.nns > 0 && fplog)
    {
        fprintf(fplog, "Average neighborlist lifetime: %.1f steps, std.dev.: %.1f steps\n", nlh.s1/nlh.nns, sqrt(nlh.s2/nlh.nns - sqr(nlh.s1/nlh.nns)));
        fprintf(fplog, "Average number of atoms that crossed the half buffer length: %.1f\n\n", nlh.ab/nlh.nns);
    }

    if (pme_loadbal != NULL)
    {
        pme_loadbal_done(pme_loadbal, cr, fplog,
                         fr->nbv != NULL && fr->nbv->bUseGPU);
    }

    if (shellfc && fplog)
    {
        fprintf(fplog, "Fraction of iterations that converged:           %.2f %%\n",
                (nconverged*100.0)/step_rel);
        fprintf(fplog, "Average number of force evaluations per MD step: %.2f\n\n",
                tcount/step_rel);
    }

    if (repl_ex_nst > 0 && MASTER(cr))
    {
        print_replica_exchange_statistics(fplog, repl_ex);
    }

    runtime->nsteps_done = step_rel;

    return 0;
}
