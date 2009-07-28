include $(top_srcdir)/build-aux/init.mk
include $(top_srcdir)/build-aux/autotest.mk

# Add RobOptim flags.
AM_CPPFLAGS += $(ROBOPTIMCORE_CFLAGS)

# Search for headers in shared tests.
AM_CPPFLAGS += -I$(srcdir)/shared-tests

# Add available warnings flags.
AM_CXXFLAGS += $(WARNING_CXXFLAGS)

# Package the following files.
EXTRA_DIST += package.m4.in

# Clean atconfig (do not distribute it).
CLEANFILES += atconfig

# Source which are shared by all tests.
COMMON_SOURCES = common.hh shared-tests/hs071.hh

# Define tests.

# Simple
check_PROGRAMS = simple
simple_SOURCES = simple.cc $(COMMON_SOURCES)			\
	$(top_srcdir)/src/roboptim-core-cfsqp-plugin.cc		\
	$(top_srcdir)/src/cfsqp.c				\
	$(top_srcdir)/src/qld.c

simple_LDADD = $(ROBOPTIMCORE_LIBS)

# Plug-in
check_PROGRAMS += plugin
plugin_SOURCES = plugin.cc $(COMMON_SOURCES)
plugin_LDADD = $(ROBOPTIMCORE_LIBS)

# Simple 1
check_PROGRAMS += simple-1
simple_1_SOURCES = simple-1.cc $(COMMON_SOURCES)
simple_1_LDADD = $(ROBOPTIMCORE_LIBS)

# Optimization problem 1
check_PROGRAMS += optimization-problem-1
optimization_problem_1_SOURCES = optimization-problem-1.cc $(COMMON_SOURCES)
optimization_problem_1_LDADD = $(ROBOPTIMCORE_LIBS)


.PHONY: generate-reference

# Re-generate reference outputs.
generate-reference:
	@echo "Generating references...";	\
	for i in $(check_PROGRAMS); do		\
	 IN="$(srcdir)/$$i.stdin";		\
	 OUT="$(srcdir)/$$i.stdout";		\
	 if test -f "$$IN"; then 		\
	  "./$$i" < $$IN > $$OUT || true;	\
	 else					\
	  "./$$i" > $$OUT || true;		\
	 fi					\
	done


# ------------ #
# Test suite.  #
# ------------ #

# Distribute reference files.
EXTRA_DIST +=					\
	simple.stdout				\
	plugin.stdout				\
	optimization-problem-1.stdout
