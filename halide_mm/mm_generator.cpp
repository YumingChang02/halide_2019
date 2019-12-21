#include "Halide.h"
#include <stdio.h>
#include "mm.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideMM : public Generator<HalideMM>
{
public:
    Input<Buffer<>> ma{"MA", 2};
    Input<Buffer<>> mb{"MB", 2};
    Input<int32_t>      k{"k"};
    Output<Func> output{"output", 2};

    void generate(){
    RDom r ( 0, k );
    output ( x, y ) = sum( cast( output.type(), ma( r, y ) ) * cast( output.type(), mb( x, r )  ) );
    }

    void schedule(){
        if(auto_schedule){
            // 1. Buffer : buf.dim(N).set_estimate(MIN, EXTENT)
            ma.dim(0).set_estimate(0,512);
            ma.dim(1).set_estimate(0,512);
            mb.dim(0).set_estimate(0,512);
            mb.dim(1).set_estimate(0,512);
            // 2. parameters : parm.set_estimate(VALUE)
            k.set_estimate(512);

            // 3. Func : func.set_estimate(Var, MIN, EXTENT);
            output.set_estimate(x,0,512);
            output.set_estimate(y,0,512);
        }
        else{
        }
    }
private:
    Var x, y;
};
HALIDE_REGISTER_GENERATOR(HalideMM, halide_mm);
