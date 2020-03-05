ifneq ($(LEAKS),0)
FCFLAGS    += -fsanitize=leak
endif
