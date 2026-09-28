# Compiles tests/api_shape_refused.cpp once per case (cmake -DCXX=... -DFLAGS=... -DINC=... -DSRC=...
# -DCASES=N -P): case 0, the target's API forms, must compile; every other case, one form the target
# does not have (docs/TARGET_API_SHAPE.md), must fail, and on a static_assert (the private
# LocalTensor::pos on an access error) rather than anything else.
separate_arguments(flag_list UNIX_COMMAND "${FLAGS}")
foreach(k RANGE 0 ${CASES})
    execute_process(COMMAND ${CXX} -std=c++17 ${flag_list} -fsyntax-only -I${INC} -DCASE=${k} ${SRC}
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(k EQUAL 0)
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR "The target's API forms do not compile:\n${err}")
        endif()
    elseif(rc EQUAL 0)
        message(FATAL_ERROR "Case ${k} compiled: a form the target does not have is accepted")
    elseif(NOT err MATCHES "static.assert|private")
        message(FATAL_ERROR "Case ${k} failed, but not on the static_assert that names the form:\n${err}")
    endif()
endforeach()
message(STATUS "Target API shape: the target's forms compile, all ${CASES} refused forms are refused")
