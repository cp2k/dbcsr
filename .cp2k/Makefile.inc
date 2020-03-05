ifeq ($(CHECKS),1)
ifneq ($(LEAKS),0)
FCFLAGS    += -fsanitize=leak
endif
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif

ifneq ($(CHECKS),1)
FCFLAGS +=  -DNDEBUG
endif
