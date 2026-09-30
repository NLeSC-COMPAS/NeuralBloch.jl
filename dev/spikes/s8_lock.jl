# Spike S8 (2026-09-30): is the engine built only once when several Julia
# processes want it at the same moment?
#
# ① Three processes start together; each takes a FileWatching.Pidfile lock
#   next to the product, builds only if the product is still missing (to a
#   temporary name, then rename), and reports "built" or "hit".
# ② A lock holder is SIGKILLed mid-build: how long until the next process
#   gets in, and what does it print?
# Decides the lock parameters and the message shown while waiting.
#
#     julia --project=dev dev/spikes/s8_lock.jl

include("common.jl")
check_onnx()
isfile(BUILDER) || compile_builder()

const DIR = joinpath(WORK, "s8")
const STALE_AGE = 10          # seconds; the holder refreshes the mtime every STALE_AGE/2

# The child: runs in a fresh julia, prints one line "<pid> <outcome> waited=<s> held=<s>".
child_code(delay) = """
using FileWatching.Pidfile
lock, product = $(repr(joinpath(DIR, "key.pid"))), $(repr(joinpath(DIR, "key.engine")))
t0 = time()
holder = trymkpidlock(lock; stale_age = $STALE_AGE)
if holder === false
    println(stderr, getpid(), ": waiting for the lock")
    holder = mkpidlock(lock; stale_age = $STALE_AGE)
end
waited = time() - t0
try
    if isfile(product)
        outcome = "hit"
    else
        tmp = product * ".tmp-" * string(getpid())
        run(pipeline(`$BUILDER $ONNX \$tmp 1 14400 40000 -1 -1 0`; stdout = devnull))
        sleep($delay)                      # widen the window for ②
        mv(tmp, product)
        outcome = "built"
    end
    println(getpid(), " ", outcome, " waited=", round(waited; digits = 2), " held=", round(time() - t0 - waited; digits = 2))
finally
    close(holder)
end
"""
spawn(delay = 0) = open(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $(child_code(delay))`; stderr = stderr))

println("== ① three processes race for the same product")
rm(DIR; recursive = true, force = true); mkpath(DIR)
procs = [spawn() for _ in 1:3]
lines = [strip(read(p, String)) for p in procs]
foreach(wait, procs)
foreach(l -> println("   ", l), lines)
println("   built exactly once: ", count(contains(" built "), lines) == 1,
        "   leftovers in the directory: ", filter(!=("key.engine"), readdir(DIR)))

println("\n== ② the lock holder is killed mid-build")
rm(DIR; recursive = true, force = true); mkpath(DIR)
victim = spawn(30)                                    # holds the lock for ≥ 30 s
while !isfile(joinpath(DIR, "key.pid")); sleep(0.05); end
sleep(1.0)
t0 = time()
survivor = spawn()
sleep(1.0)
kill(victim, Base.SIGKILL); wait(victim)
tk = time()
line = strip(read(survivor, String)); wait(survivor)
println("   survivor: ", line)
@printf("   survivor finished %.1f s after the kill (stale_age = %d s, build ≈ 5 s)\n", time() - tk, STALE_AGE)
println("   leftovers: ", filter(!=("key.engine"), readdir(DIR)))
