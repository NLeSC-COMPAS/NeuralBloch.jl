# The golden values of test/test_golden.jl: the EPG reference for some voxels,
# for the model that Artifacts.toml binds. Run this script again after you
# bind a new model (README, "Adopt a new model"):
#
#     julia --project=scripts scripts/golden.jl
#
# The script writes test/golden.toml. The file gives the family and the
# version of the model, the voxels, some readouts of the signal and of its
# derivatives from the EPG simulation of BlochSimulators
# (scripts/reference.jl), and the NMAE of the model on these values. The test
# accepts two times this NMAE. An error in a convention gives approximately
# 100 %. The output is the same for each run on the same machine.

using NeuralBloch
using NeuralBloch: cardname
using TOML: TOML

include("reference.jl")

const VOXELS = 8
const SEED = 8
const EACH = 25                      # every 25th readout: 24 of the 600

model = Model()
# Float32 values, as the engine reads them, written as Float64 in the file.
v = map(x -> Float64.(Float32.(x)), evaluation_set(model, VOXELS; seed = SEED))
ref = epg(model, v)
engine = Engine(model)
pred = predict(engine, gpu_voxels(v))
close(engine)

rows = collect(1:EACH:model.ntr)
golden = Dict{String,Any}("family" => model.family, "version" => string(model.version),
                          "voxels" => Dict("T1" => v.T₁, "T2" => v.T₂, "B1" => v.B₁), "readouts" => rows)
for k in (:signal, :∂T₁, :∂T₂)
    r = ComplexF64.(Array(ref[k])[rows, :])
    golden[cardname(k)] = Dict("re" => collect(eachcol(real.(r))), "im" => collect(eachcol(imag.(r))),
                               "nmae" => nmae(Array(pred[k])[rows, :], r))
end

path = joinpath(@__DIR__, "..", "test", "golden.toml")
open(io -> TOML.print(io, golden; sorted = true), path, "w")
println("Wrote ", normpath(path), " for ", model.family, " v", model.version, ": NMAE ",
        join(("$(cardname(k)) $(round(100golden[cardname(k)]["nmae"]; digits = 4)) %" for k in (:signal, :∂T₁, :∂T₂)),
             ", "))
