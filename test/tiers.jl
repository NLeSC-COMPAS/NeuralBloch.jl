# Test tiers. A tier runs when the machine provides its prerequisites;
# otherwise each of its testsets is skipped and the reason printed. With
# NEURALBLOCH_TEST_ALL=1 a skip counts as a failure, so a machine that should
# run everything cannot silently miss tests.
#
#   T0  Julia only
#   T1  + g++, TensorRT's headers and libraries (native code)
#   T2  + a functional CUDA GPU
#   T3  + BlochSimulators in the test environment (accuracy against EPG)

using NeuralBloch: TensorRT

const TEST_ALL = get(ENV, "NEURALBLOCH_TEST_ALL", "") == "1"
const SYSTEM = TensorRT.detect()

# `nothing` when tier `t` can run here, otherwise the reason it cannot.
function blocker(t::Symbol)
    t === :T0 && return nothing
    native = TensorRT.problems(SYSTEM; gpu = false)
    isempty(native) || return join(last.(native), "; ")
    t === :T1 && return nothing
    SYSTEM.gpu === nothing && return "no functional CUDA GPU: " * SYSTEM.gpu_error
    t === :T2 && return nothing
    Base.find_package("BlochSimulators") === nothing &&
        return "BlochSimulators is not in the test environment"
    t === :T3 && return nothing
    error("unknown test tier $t")
end

# Run `f` as testset `name` of tier `t`, or record why it was skipped.
function tier(f, t::Symbol, name::AbstractString)
    reason = blocker(t)
    if reason === nothing
        @testset "[$t] $name" f()
    else
        @info "Skipping [$t] $name: $reason"
        @testset "[$t] $name" begin
            if TEST_ALL
                @test "skipped: $reason" == "ran (NEURALBLOCH_TEST_ALL=1)"
            else
                @test_skip "skipped: $reason" == "ran"
            end
        end
    end
end
