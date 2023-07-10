all: build-env-latest-gcc build-env-rocm build-env-ubuntu build-env-ubuntu-cuda

build-env-%: Dockerfile.build-env-%
	docker build -t dbcsr-$@ -f $< .
