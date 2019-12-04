# OpenMP/offload Backend

## Introduction

The OpenMP based implementation of DBCSR's ACC interface implements the stream programming model using OpenMP primitives, i.e., mainly tasks and task dependencies. The stream programming model extracts parallelism by queuing tasks into streams. Parallelism is achieved by the means of asynchronous tasks with respect to the enqueuing (host-)system. However, typically at least two streams are necessary to effectively parallelize work by the means of double-buffering (odd/even stream). A stream models a pipeline of work and all work is strictly sequential with respect to the enqueued tasks (a task itself may be worked on in parallel however).

## OpenMP Device-Offload

A `target` directive is used to offload work to an accelerator device. The accelerator or specialized device can be also memory coherent, i.e., data movement may be elided. A target region is an OpenMP task and be launched asynchronous (`nowait` clause). Typically multiple directives are combined, e.g., `omp target teams distribute parallel for simd nowait` which offloads the code region underneath of the directive, opens a parallel region on the target device, and then distributes work by potentially exploiting three levels of parallelism (teams, threads, vectors). The `depend` clause applicable to `omp task` (or `omp target`) can be used to describe an acyclic graph of input, output, or inout dependences (DAG). OpenMP will maximize the flow for each DAG of dependencies. A stream primitive is hence not necessary to model parallelism.

## DBCSR ACC Interface

The ACC interface describes a minimal set of primitives (streams, events, etc.) to support the stream programming model. In order to get existing code in DBCSR just working, the purpose of an OpenMP backend is to implement the stream programming model. However, new accelerator code in CP2K or DBCSR could also use OpenMP's target region right away.

## Backend Implementation

Launching OpenMP tasks typically looks like (C/C++ programming language):

```c
#pragma omp parallel
{
  /* other code */
# pragma omp single
  {
    int i = 0;
    for (; i < ntasks; ++i) {
#     pragma omp task
      {
        /* work */
      }
    }
  }
}
```

The parallel region represents a team of threads and every thread of the team performs the work described by the code in the region (work is not distributed among threads unless requested, e.g., per `omp for`). Typically, not every thread should produce N tasks, hence only one of the threads (`omp single`) is instructed to launch tasks. The team lead could do this (`omp master`), however any thread is fine (the first thread that enters `single`).

Tasks can be ordered using a `depend` clause designating variables as inputs (`in`), outputs (`out`), or both (`inout`), e.g., `depend(in:a,b,c)`. In C/C++ and technically, the address of the variable is taken by the clause to identify a dependency. The DAG formed exists in the context of the thread issuing the tasks, i.e., dependencies (edges in the same graph) cannot be formed by different threads. This is normally not an issue, however it induces a more sophisticated pattern if tasks are meant to be enqueued into streams.

```c
#pragma omp parallel
{
  /* other code */
  
# pragma omp master
# pragma omp task depend(out:a)
  {
    /* 1st work item produces "a" */
  }

  /* other code */

# pragma omp master
# pragma omp task depend(in:a) depend(out:b)
  {
    /* 2nd work item consumes "a" and produces "b" */
  }

  /* other code */

# pragma omp master
# pragma omp task depend(in:b) depend(out:c)
  {
    /* 3rd work item consumes "b" and produces "c" */
  }
}
```

In the above code, `master` must be used since otherwise the dependencies (edges) may not contribute to the same DAG. Issuing a task (as shown above) may be placed into a function ("enqueue_workitem()"). However, that function may be launched on a per-thread basis, hence the arguments supplied for such a call must be collected from every thread, packaged and fed into an own task (on a per-thread basis). This is sort of "reduction" collecting work from each thread into `master` and issuing tasks from there. Work on a per-thread basis may look like:

```c
int tid = omp_get_thread_num()
enqueue_workitem(work[tid], stream);
```

Even the stream (representing a DAG) can be different on a per-thread basis (`stream[tid]`). Of course, a dependency (edge) must be (artificially) generated on a per-stream basis in order to assert a stream programming model. Though, a stream enqueues work in a sequential fashion just like in above example (a -> b -> c). However, multiple streams may be managed inside of the same parallel region.
