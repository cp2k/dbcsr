ifeq ($(CHECKS),1)
ifneq ($(LEAKS),0)
FCFLAGS    += -fsanitize=leak
endif
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif
