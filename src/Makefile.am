include $(top_srcdir)/build-aux/init.mk

# Add RobOptim flags.
AM_CPPFLAGS += $(ROBOPTIMCORE_CFLAGS)

# Add available warnings flags.
AM_CXXFLAGS += $(WARNING_CXXFLAGS)

lib_LTLIBRARIES = roboptim-core-cfsqp-plugin.la

roboptim_core_cfsqp_plugin_la_SOURCES = 	\
	debug.hh				\
	doc.hh					\
	roboptim-core-cfsqp-plugin.cc

roboptim_core_cfsqp_plugin_la_SOURCES +=	\
	ofsqp.cc ofsqp.hh

roboptim_core_cfsqp_plugin_la_LIBADD = $(ROBOPTIMCORE_LIBS)
roboptim_core_cfsqp_plugin_la_LDFLAGS = -module -version-info 1:1:0
