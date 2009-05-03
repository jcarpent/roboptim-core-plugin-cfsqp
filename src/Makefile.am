include $(top_srcdir)/build-aux/init.mk

# Add available warnings flags.
AM_CXXFLAGS += $(WARNING_CXXFLAGS)

lib_LTLIBRARIES = roboptim-core-cfsqp-plugin.la

roboptim_core_cfsqp_plugin_la_SOURCES = 	\
	doc.hh					\
	roboptim-core-cfsqp-plugin.cc

roboptim_core_cfsqp_plugin_la_SOURCES +=	\
	cfsqp.c					\
	cfsqpusr.h				\
	qld.c

roboptim_core_cfsqp_plugin_la_LDFLAGS = -module -version-info 1:1:0
