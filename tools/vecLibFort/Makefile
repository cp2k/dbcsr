PREFIX=/usr/local
LIBDIR=$(PREFIX)/lib

CFLAGS=-O

NAME=vecLibFort
SOURCE=$(NAME).c
OBJECT=$(NAME).o
LIBRARY=lib$(NAME)
STATIC=$(LIBRARY).a
DYNAMIC=$(LIBRARY).dylib
PRELOAD=$(LIBRARY)I.dylib
INCLUDES=cloak.h static.h
DEPEND=$(INCLUDES) Makefile

all: static dynamic preload
static: $(STATIC)
dynamic: $(DYNAMIC)
preload: $(PRELOAD)

$(OBJECT): $(DEPEND)

$(STATIC): $(OBJECT)
	ar -cru $@ $^
	ranlib $@

$(DYNAMIC): $(OBJECT)
	clang -shared -o $@ $^ \
		-Wl,-reexport_framework -Wl,Accelerate \
		-install_name $(LIBDIR)/$@

$(PRELOAD): $(SOURCE) $(DEPEND)
	clang -shared $(CFLAGS) -DVECLIBFORT_INTERPOSE -o $@ -O $(SOURCE) \
		-Wl,-reexport_framework -Wl,Accelerate \
		-install_name $(LIBDIR)/$@

install: all
	mkdir -p $(LIBDIR)
	cp -f $(STATIC) $(LIBDIR)
	cp -f $(DYNAMIC) $(LIBDIR)
	cp -f $(PRELOAD) $(LIBDIR)

clean:
	rm -f $(OBJECT) $(STATIC) $(DYNAMIC) $(PRELOAD)

check: tester.f90 $(OBJECT)
	gfortran -o tester -O $^ -framework Accelerate 
	./tester

