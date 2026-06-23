# Fast Screened Coulomb — C Library
# Usage: make test    (run TDD validation suite)
#        make bench   (run expansion benchmark)
#        make clean

CC       = gcc
CFLAGS   = -std=c99 -Wall -O2 -Isrc
LDFLAGS  = -lm

SRC      = src/fast_screened_coulomb.c
HEADER   = src/fast_screened_coulomb.h

.PHONY: all test bench clean

all: test

test: tests/test_fsc
	./tests/test_fsc

bench: tests/benchmark_fsc
	./tests/benchmark_fsc

tests/test_fsc: tests/test_fsc.c $(SRC) $(HEADER)
	$(CC) $(CFLAGS) -o $@ tests/test_fsc.c $(SRC) $(LDFLAGS)

tests/benchmark_fsc: tests/benchmark_fsc.c $(SRC) $(HEADER)
	$(CC) $(CFLAGS) -o $@ tests/benchmark_fsc.c $(SRC) $(LDFLAGS)

clean:
	-rm -f tests/test_fsc tests/benchmark_fsc tests/test_fsc.exe tests/benchmark_fsc.exe
