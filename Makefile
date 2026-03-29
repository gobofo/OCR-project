# Makefile — OCR + crossword solver
#
# Targets:
#   make all    — build ./train and ./solve  (default)
#   make train  — build ./train only
#   make solve  — build ./solve only
#   make clean  — remove object files and binaries

# -------------------------------------------------------------------------
# Compiler and flags
# -------------------------------------------------------------------------

CC      := gcc
CFLAGS  := -std=c99 -Wall -Wextra -Wpedantic \
           -D_POSIX_C_SOURCE=200809L \
           $(shell pkg-config --cflags sdl2) \
           $(shell pkg-config --cflags libpng)
LDFLAGS :=
LIBS    := $(shell pkg-config --libs sdl2) \
           $(shell pkg-config --libs libpng) \
           -lm -lpthread

# Enable optimisation by default; override with: make DEBUG=1
ifeq ($(DEBUG),1)
CFLAGS  += -O0 -g3
else
CFLAGS  += -O2
endif

# -------------------------------------------------------------------------
# Source files
# -------------------------------------------------------------------------

# Common source files shared by both binaries
COMMON_SRCS := \
    src/cnn/cnn.c         \
    src/cnn/model.c       \
    src/preprocess/image.c

# train-only sources
TRAIN_SRCS  := \
    src/cnn/dataset.c     \
    train_main.c

# solve-only sources
SOLVE_SRCS  := \
    src/segment/segment.c \
    src/solver/solver.c   \
    solve_main.c

# -------------------------------------------------------------------------
# Object files
# -------------------------------------------------------------------------

COMMON_OBJS := $(COMMON_SRCS:.c=.o)
TRAIN_OBJS  := $(COMMON_OBJS) $(TRAIN_SRCS:.c=.o)
SOLVE_OBJS  := $(COMMON_OBJS) $(SOLVE_SRCS:.c=.o)

ALL_OBJS    := $(sort $(TRAIN_OBJS) $(SOLVE_OBJS))

# -------------------------------------------------------------------------
# Phony targets
# -------------------------------------------------------------------------

.PHONY: all train solve clean

all: train solve

# -------------------------------------------------------------------------
# Link rules
# -------------------------------------------------------------------------

train: $(TRAIN_OBJS)
	$(CC) $(LDFLAGS) -o train $(TRAIN_OBJS) $(LIBS)

solve: $(SOLVE_OBJS)
	$(CC) $(LDFLAGS) -o solve $(SOLVE_OBJS) $(LIBS)

# -------------------------------------------------------------------------
# Generic compile rule
# -------------------------------------------------------------------------

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# -------------------------------------------------------------------------
# Clean
# -------------------------------------------------------------------------

clean:
	rm -f $(ALL_OBJS) train solve
