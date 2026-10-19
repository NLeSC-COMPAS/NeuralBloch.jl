# Shared by every test file: the test tiers, and fixtures.
#
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

# --- fixtures -------------------------------------------------------------------

# The default bundle: the artifact Pkg installed with the package.
const BUNDLE = Model().dir

# The default bundle's signal-and-derivatives model, in its artifact.
const ONNX = Model().onnx.signal_and_derivatives

# A private copy of the bundle in `dir`. Its model_card.toml is
# `edit`(original text), with `version` in place of the original version, and
# `files`(copy) changes the other files. The files of the artifact are
# read-only, and cp keeps their mode, thus the test makes the copy writable.
function edited_bundle(edit, dir; version = "0.1.0-beta.1", files = identity)
    b = joinpath(dir, "bundle")
    cp(BUNDLE, b)
    chmod(b, 0o755; recursive = true)
    card = joinpath(b, "model_card.toml")
    write(card, edit(replace(read(card, String), "version = \"0.1.0-beta.1\"" => "version = \"$version\"")))
    files(b)
    return b
end
