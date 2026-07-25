# Build: `make cpu` (any C++20 compiler) for the CPU targets, `make gpu` (nvcc)
# for the CUDA targets. Verify targets run the exactness gates against the
# verify/golden/ fixtures. If CPU-vs-GPU differs, rebuild with
# NVFLAGS="-std=c++20 -O2 -fmad=false" and record the measured diff.
CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -std=c++20 -O2 -Wall
NVFLAGS  ?= -std=c++20 -O2 -arch=native

BIN := bin

$(BIN):
	mkdir -p $(BIN)

# ---- exactness gates ----
$(BIN)/verify_rng: verify/verify_rng.cpp engine/rng.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/verify_parser: verify/verify_parser.cpp engine/instance.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/verify_compile: verify/verify_compile.cpp engine/tree.hpp engine/vocab.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/verify_sim: verify/verify_sim.cpp engine/sim_host.hpp engine/sim_core.hpp engine/sim_terms.hpp engine/instance.hpp engine/rng.hpp engine/vocab.hpp engine/tree.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/verify_gp: verify/verify_gp.cpp engine/gp.hpp engine/sim_host.hpp engine/sim_core.hpp engine/sim_terms.hpp engine/rng.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/verify_cuda: verify/verify_cuda.cu engine/factory.cuh engine/sim_core.hpp engine/sim_terms.hpp engine/sim_host.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

# ---- stage 1: reference routes and gaps ----
$(BIN)/run_s1: s1_reference/run.cpp s1_reference/reference.hpp engine/json.hpp engine/rng.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/verify_s1: verify/verify_s1.cpp s1_reference/reference.hpp engine/rng.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---- stage 2: terminal construction (route reading) ----
$(BIN)/lookit: s2_terminals/lookit.cpp s2_terminals/decode.hpp engine/json.hpp engine/instance.hpp engine/tree.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/lookit_mech: s2_terminals/lookit_mech.cpp s2_terminals/decode.hpp engine/json.hpp engine/instance.hpp engine/tree.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---- stage 3: rule generation (GP-HH engine) ----
$(BIN)/gp_engine: engine/gp_engine.cu engine/factory.cuh engine/gp.hpp engine/sim_core.hpp engine/sim_terms.hpp engine/sim_host.hpp engine/rng.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

$(BIN)/gen_trace: s3_generation/gen_trace.cu engine/factory.cuh engine/gp.hpp engine/jwrite.hpp engine/baseline_csv.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

$(BIN)/timing: s3_generation/timing.cu engine/factory.cuh engine/gp.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

# ---- stage 4: per-instance vocabulary selection and final choice ----
$(BIN)/ga: s4_selection/ga.cu engine/factory.cuh engine/gp.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp -o $@ $<

$(BIN)/choice: s4_selection/choice.cu engine/factory.cuh engine/json.hpp engine/baseline_csv.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

$(BIN)/paired: s4_selection/paired.cpp engine/json.hpp engine/stats.hpp engine/baseline_csv.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/redeploy: s4_selection/redeploy.cu engine/factory.cuh engine/json.hpp engine/baseline_csv.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

cpu: $(BIN)/verify_rng $(BIN)/verify_parser $(BIN)/verify_compile $(BIN)/verify_sim $(BIN)/verify_gp $(BIN)/run_s1 $(BIN)/verify_s1 $(BIN)/lookit $(BIN)/lookit_mech $(BIN)/paired

gpu: $(BIN)/verify_cuda $(BIN)/gp_engine $(BIN)/gen_trace $(BIN)/timing $(BIN)/ga $(BIN)/choice $(BIN)/redeploy

verify-f0: $(BIN)/verify_rng
	$(BIN)/verify_rng verify/golden/rng.txt

verify-f1: $(BIN)/verify_parser $(BIN)/verify_compile
	GITC_DATA=instances $(BIN)/verify_parser verify/golden/parser.txt
	$(BIN)/verify_compile verify/golden/compile.txt

verify-f2: $(BIN)/verify_sim
	GITC_DATA=instances $(BIN)/verify_sim verify/golden/sim.txt verify/golden/durations.txt

verify-f3: $(BIN)/verify_gp
	GITC_DATA=instances $(BIN)/verify_gp verify/golden/gp.txt

verify-f4: $(BIN)/verify_cuda
	GITC_DATA=instances $(BIN)/verify_cuda verify/golden/sim.txt

verify-f5: $(BIN)/verify_s1
	GITC_DATA=instances $(BIN)/verify_s1 verify/golden/s1.txt

verify: verify-f0 verify-f1 verify-f2 verify-f3 verify-f5

clean:
	rm -rf $(BIN)

.PHONY: cpu gpu verify verify-f0 verify-f1 verify-f2 verify-f3 verify-f4 verify-f5 clean
