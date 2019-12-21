#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>

#define TEST_M 512
#define TEST_K 512
#define TEST_N 512

#include "arm_neon.h"
void gemm8x8_vec(uint8_t *a, int32_t sa, uint8_t *b, int32_t sb, uint32_t *c, int32_t sc)
{
    uint8x8_t vb[8];
    for( int i = 0; i < 8; ++i ){
        vb[i] = vld1_u8( b + sb * i );
    }

    for( int j = 0; j < 8; ++j ){
        // init vector c
        uint32x4_t vc[2];
        vc[0] = vld1q_u32( c + sc * j );
        vc[1] = vld1q_u32( c + sc * j + 4 );

        uint8x8_t va = vld1_u8( a + sa * j );
        for( int i = 0; i < 8; ++i ){
            uint8x8_t temp_a = vdup_n_u8( va[i] );
            uint16x8_t temp = vmull_u8( temp_a, vb[i] );
            vc[1] = vaddw_high_u16( vc[1], temp );
            vc[0] = vaddw_u16( vc[0], vget_low_u16( temp ) );
        }
        vst1q_u32( c + sc * j, vc[0] );
        vst1q_u32( c + sc * j + 4, vc[1] );
    }
}

int main(void)
{
	uint8_t* ma = (uint8_t*)malloc(sizeof(uint8_t)*TEST_K*TEST_M);
	uint8_t* mb = (uint8_t*)malloc(sizeof(uint8_t)*TEST_N*TEST_K);
	uint32_t* mc = (uint32_t*)malloc(sizeof(uint32_t)*TEST_N*TEST_M);
	uint32_t* chk = (uint32_t*)malloc(sizeof(uint32_t)*TEST_N*TEST_M);

	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_K; x++){
			ma[y*TEST_K + x] = (uint8_t)(rand()%256);
		}
	}
	for(int y = 0; y < TEST_K; y++){
		for(int x = 0; x < TEST_N; x++){
			mb[y*TEST_N + x] = (uint8_t)(rand()%256);
		}
	}
	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_N; x++){
			mc[y*TEST_N + x] = 0;
			chk[y*TEST_N + x] = 0;
		}
	}

	struct timeval stime, etime;
	gettimeofday(&stime, NULL);

    //parallel here
	for(int m = 0; m < TEST_M; m+=8){
		for(int n = 0; n < TEST_N; n+=8){
			for(int k = 0; k < TEST_K; k+=8){
				gemm8x8_vec(
					ma + m*TEST_K + k, TEST_K,
					mb + k*TEST_N + n, TEST_N,
					mc + m*TEST_N + n, TEST_N
				);
			}
		}
	}
	gettimeofday(&etime, NULL);
	printf("INT SIMD: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

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
			uint32_t val = chk[m*TEST_N + n] - mc[m*TEST_N + n];
			if( val ){
				printf("(%d,%d), %u %u\n", m, n, chk[m*TEST_N + n], mc[m*TEST_N + n]);
			}
		}
	}

	printf("DONE!\n");
	return 0;
}
