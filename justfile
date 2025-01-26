# just is a command runner, Justfile is very similar to Makefile, but simpler.

default:
  @just --list

test:
  uv run --with pytest \
    pytest --doctest-modules \
      --ignore=examples

test-cov:
  uv run --with pytest --with pytest-cov \
    pytest \
      --doctest-modules \
      --cov=swcgeom --cov-report=xml --cov-report=html \
      --ignore=examples
