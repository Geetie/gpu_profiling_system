"""Hardware probing subsystem.

Measures actual GPU parameters using CUDA micro-benchmarks.
All measurements use hardware clock() cycles — immune to frequency locking.

Core principle: clock() counts SM clock cycles. Actual frequency is
derived by comparing cycle counts against ncu wall-clock timing.
"""
