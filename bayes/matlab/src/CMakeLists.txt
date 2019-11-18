##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of software.
#
# Copyright (c${NCUT_INCLUDES}/) 2013 University of Pennsylvania. All rights reserved.<br />
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

# ============================================================================
# library target(s)
# ============================================================================

# Add library target for each library using basis_add_library().
#
# This command can not only be used to build libraries from C/C++, but also source
# code written in other programming languages such as Java, Python, Perl,
# MATLAB, and Bash. Note that here we consider modules written in a scripting
# language which are no executables but to be included by other scripts written
# in the particular language as libraries.
#
# Note: Public modules written in a scripting language such as Python, Perl,
#       MATLAB, or Bash which are intended for use by other packages should
#       be placed in the lib/[<lang>/]dppRegression/ directory,
#       where <lang> is the language name in lowercase and is optional.
#       BASIS will automatically add a library build target for these modules.



# ============================================================================
# executable target(s)
# ============================================================================

# Add executable target for each executable program using basis_add_executable().
#
# This command can not only be used to build executables from C/C++, but also
# source code written in other programming languages such as Java, Python, Perl,
# MATLAB, and Bash.

# main analysis of algorithm
# an excutable to run Carbonetto code on all regions
basis_add_executable (runSpikeSlabOnPhenotype.m 
	${VBSPSLAB_INCLUDES}/*.m )   
basis_set_target_properties (runSpikeSlabOnPhenotype   PROPERTIES   COMPILE_FLAGS "${BASIS_MCC_FLAGS} -R -nodisplay")
basis_target_link_libraries (
        runSpikeSlabOnPhenotype
        ${VBSPSLAB_MEX_FILES}
)



# executable to run different steps of the Imaging-genetics models
basis_include_directories(${GPML_INCLUDES})
basis_add_executable (deployEndoPhenVB.m
        ${VBSPSLAB_INCLUDES}/*.m 
	Utils/*.m 
	mapDPP/util/*.m
        mapDPP/rounding/*.m
        mapDPP/lbfgs/*.m    
        mapDPP/match-util/*.m
        mapDPP/map-algorithms/*.m )  
basis_set_target_properties (deployEndoPhenVB   PROPERTIES   COMPILE_FLAGS "${BASIS_MCC_FLAGS} -R -nodisplay")
basis_target_link_libraries (
     deployEndoPhenVB
     mapDPP/lbfgs/lbfgs_mex
     mapDPP/match-util/lap_double
     mapDPP/match-util/lap
     ${GPML_INCLUDE_DIR}/*.m
     ${GPML_INCLUDE_DIR}/mean/*.m
     ${GPML_INCLUDE_DIR}/lik/*.m
     ${GPML_INCLUDE_DIR}/cov/*.m
     ${GPML_INCLUDE_DIR}/inf/*.m
     ${GPML_INCLUDE_DIR}/util/*.m 
     ${LIBLINEAR_MEX_FILES} 
     ${VBSPSLAB_MEX_FILES}
)


set(MCR_DIR "$ENV{MCR_DIR}")
basis_add_executable(scripts/runMatlabExec.sh.in)
basis_add_executable(scripts/sgeLaunchScript.sh.in)
basis_add_executable(scripts/sgeLaunchMATLABScript.sh.in) 

