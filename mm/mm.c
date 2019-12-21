#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define TEST_M 512
#define TEST_K 512
#define TEST_N 512



#include "arm_neon.h"
void gemm4x4_vec(float *a, int sa, float *b, int sb, float *c, int sc)
{
    float32x4_t vb[4];
    for( int i = 0; i < 4; ++i ){
        vb[i] = vld1q_f32( b + sb*i );
    }

    for( int y = 0; y < 4; ++y ){
        float32x4_t va = vld1q_f32( a + y*sa );
        float32x4_t vc = vld1q_f32( c + y*sc );

        // float32x4_t vmlaq_laneq_f32 = ( float32x4_t A, float32x4_t B, float32x4_t C, const int lane )
        // take "lane" th item from C as scaler, multply scaler to vector B, then accumulate result with A, returns float32x4_t
        vc = vmlaq_laneq_f32(vc, vb[0], va, 0);
        vc = vmlaq_laneq_f32(vc, vb[1], va, 1);
        vc = vmlaq_laneq_f32(vc, vb[2], va, 2);
        vc = vmlaq_laneq_f32(vc, vb[3], va, 3);
        vst1q_f32( c + y*sc, vc );
    }

    //for( int y = 0; y < 4; ++y ){
    //    for( int x = 0; x < 4; ++x ){
    //        float temp = *( c + y*sc + x );
    //        for( int k = 0; k < 4; ++k ){
    //            temp += *( a + sa*y + k ) * *( b + sb*k + x );
    //        }
    //        *( c + y*sc + x ) = temp;
    //    }
    //}
}

int main(void)
{
	float* ma = (float*)malloc(sizeof(float)*TEST_K*TEST_M);
	float* mb = (float*)malloc(sizeof(float)*TEST_N*TEST_K);
	float* mc = (float*)malloc(sizeof(float)*TEST_N*TEST_M);
	float* chk = (float*)malloc(sizeof(float)*TEST_N*TEST_M);

	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_K; x++){
			ma[y*TEST_K + x] = (float)(rand()%256/256.0);
		}
	}
	for(int y = 0; y < TEST_K; y++){
		for(int x = 0; x < TEST_N; x++){
			mb[y*TEST_N + x] = (float)(rand()%256/256.0);
		}
	}
	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_N; x++){
			mc[y*TEST_N + x] = (float)0.0;
			chk[y*TEST_N + x] = (float)0.0;
		}
	}

	struct timeval stime, etime;
	gettimeofday(&stime, NULL);

    //parallel here
	for(int m = 0; m < TEST_M; m+=4){
		for(int n = 0; n < TEST_N; n+=4){
			for(int k = 0; k < TEST_K; k+=4){
				gemm4x4_vec(
					ma + m*TEST_K + k, TEST_K,
					mb + k*TEST_N + n, TEST_N,
					mc + m*TEST_N + n, TEST_N
				);
			}
		}
	}
	gettimeofday(&etime, NULL);
	printf("FP32 SIMD: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

	gettimeofday(&stime, NULL);
	for(int m = 0; m < TEST_M; m++){
		for(int n = 0; n < TEST_N; n++){
			float val = 0.0;
			for(int k = 0; k < TEST_K; k++){
				val += ma[m*TEST_K + k]*mb[k*TEST_N+n];
			}
			chk[m*TEST_N + n] = val;
		}
	}
	gettimeofday(&etime, NULL);
	printf("NAIVE: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

	for(int m = 0; m < TEST_M; m++){
		for(int n = 0; n < TEST_N; n++){
			float val = chk[m*TEST_N + n] - mc[m*TEST_N + n];
			if( fabs(val) > 0.1){
				printf("(%d,%d), %f %f\n", m, n, chk[m*TEST_N + n], mc[m*TEST_N + n]);
			}
		}
	}

	printf("DONE!\n");
	return 0;
}
