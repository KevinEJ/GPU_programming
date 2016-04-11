#include "lab2.h"
#include "SyncedMemory.h"
#include <iostream>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define SIN_12 0.087155
#define COS_12 0.996194
#define DEGREE *3.1415926/180
#define CENTER 153920
#define R      100
#define Y_SAT  200
#define CIRCLE_FRE  901
#define Y_INDEX  (i)*640+j
#define U_INDEX  640*480 + (i/2)*320+(j/2)
#define V_INDEX  640*600 + (i/2)*320+(j/2)
#define gpu_Y_INDEX  (blockIdx.x)*640+threadIdx.x
#define gpu_U_INDEX  640*480 + (blockIdx.x/2)*320+(threadIdx.x/2)
#define gpu_V_INDEX  640*600 + (blockIdx.x/2)*320+(threadIdx.x/2)
#define new_Y_INDEX  (new_i)*640+new_j
#define new_U_INDEX  640*480 + (new_i/2)*320+(new_j/2)
#define new_V_INDEX  640*600 + (new_i/2)*320+(new_j/2)

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

/*
__global__ void init_stuff(curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1337, idx, 0, &state[idx]);
}
__global__ void make_rand(curandState *state, float *randArray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    randArray[idx] = curand_uniform(&state[idx]);
}*/


Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};
#if CPU
void gpu_Generation_firstframe(uint8_t *input_gpu) {
    //extent the current frame
    //unsigned char *tempframe = new unsigned char[640*480];
    //uint8_t *temp_frame = new uint8_t[640*480];
    //uint8_t *temp_2_frame = new uint8_t[640*480];
    
    //center point
    for(int i = 0 ; i < 640*480 ; i++){
        input_gpu[i] = 128 ; 
    }
    input_gpu[240*640+320] = 255 ;
    //other points
    int r = (sqrt(240*240 + 320*320)) / 3 ; 
    for(int i = 0 ; i < 50 ; i ++ ){
        int rand_deg = rand() % 360 ;
        float c = 0.9 + 0.8*(rand()%21)/20; 
        int rand_r      = c * r  ;
        //printf( " rand_r = %d \n" , rand_r  ) ;
        int x = rand_r*sin(rand_deg DEGREE) + 240; 
        int y = rand_r*cos(rand_deg DEGREE) + 320;
        //int idx = x * 640 + y ; 
        input_gpu[x * 640 + y ] = Y_SAT ;
        if( rand() % 4 == 0 ){
            input_gpu[640*480 + (x/2)*320 + (y/2) ] = 230 ;
            input_gpu[640*600 + (x/2)*320 + (y/2) ] = 20 + rand() % 210 ;
        }
        if( rand() % 4 == 1 ){
            input_gpu[640*480 + (x/2)*320 + (y/2) ] = 20 ;
            input_gpu[640*600 + (x/2)*320 + (y/2) ] = 20 + rand() % 210 ;
        }
        if( rand() % 4 == 2 ){
            input_gpu[640*480 + (x/2)*320 + (y/2) ] = 20 + rand() % 210 ;
            input_gpu[640*600 + (x/2)*320 + (y/2) ] = 230 ;
        }
        if( rand() % 4 == 3 ){
            input_gpu[640*480 + (x/2)*320 + (y/2) ] = 20 + rand() % 210 ;
            input_gpu[640*600 + (x/2)*320 + (y/2) ] = 20 ;
        }
        //input_gpu[640*480 + (x/2)*320 + (y/2) ] = 230 ;
        //input_gpu[640*600 + (x/2)*320 + (y/2) ] = 20 ;
    }
}
#else
__global__ void gpu_Generation_firstframe(uint8_t *input_gpu , int t , int* mv) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input_gpu[idx] = 128 ;
    mv[idx*2] = 0 ; 
    mv[idx*2 + 1] = 0 ; 
    // Y
        int distance = sqrtf((blockIdx.x-240)*(blockIdx.x-240)+(threadIdx.x-320)*(threadIdx.x-320));
        if(distance >= R*0.8 && distance <= R*1.8){
  
  int* result = new int(0);
  curandState_t state;
  curand_init(idx*t*t,  0,  0,  &state);
  *result = curand(&state) % 10000001;
  
            //int rand_num = randArray[idx] ;
            int rand_num = *result ;
            //printf("this random = %d \n " , rand_num ) ;
            if( rand_num % CIRCLE_FRE == 0 ){ 
                input_gpu[idx] = Y_SAT ;
                input_gpu[gpu_U_INDEX] = (rand_num/CIRCLE_FRE)*167 % 255 ;  
                input_gpu[gpu_V_INDEX] = (rand_num/CIRCLE_FRE)*193 % 255 ;  
                mv[idx*2] = 1 + ((rand_num / CIRCLE_FRE )% 11 ); 
                mv[idx*2+1] = 1 + ((rand_num / CIRCLE_FRE) % 13 ); 
                
                if(rand_num/CIRCLE_FRE % 4 == 0 ){
                }else if(rand_num/CIRCLE_FRE % 4 == 1 ){
                    mv[idx*2] *= -1 ;
                }else if(rand_num/CIRCLE_FRE % 4 == 2 ){
                    mv[idx*2+1] *= -1 ;
                }else if(rand_num/CIRCLE_FRE % 4 == 3 ){
                    mv[idx*2] *= -1 ;
                    mv[idx*2+1] *= -1 ;
                }

                /*
                if( (rand_num/CIRCLE_FRE)%4 == 0 ){
                    input_gpu[640*480 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 230 ;
                    input_gpu[640*600 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 30  + 10*(rand_num % 20) ;
                }
                else if( (rand_num/CIRCLE_FRE) % 4 == 1 ){
                    input_gpu[640*480 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 20 ;
                    input_gpu[640*600 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 30 + 10*(rand_num % 20) ;
                }
                else if( (rand_num/CIRCLE_FRE) % 4 == 2 ){
                    input_gpu[640*480 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 30 + 10*(rand_num % 20) ;
                    input_gpu[640*600 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 230 ;
                }
                else if( (rand_num/CIRCLE_FRE) % 4 == 3 ){
                    input_gpu[640*480 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 30 + 10*(rand_num % 20) ;
                    input_gpu[640*600 + (blockIdx.x/2)*320 + (threadIdx.x/2) ] = 20 ;
                }*/
            }
        }
    if(idx == CENTER)
        input_gpu[idx] = 255 ;
}
#endif

#if CPU
void gpu_Generation_changeColor(uint8_t *input_gpu){
    input_gpu[240*640+320] = 255 ;
    for(int i = 0 ; i<480 ; i++ ){
        for(int j = 0 ; j < 640 ; j++){
            if( input_gpu[i*640+j] == Y_SAT ){
                //input_gpu[i*640+j] = 200 ;
                /*
                if(input_gpu[U_INDEX] == 230 && input_gpu[V_INDEX]!=230)
                    input_gpu[V_INDEX]+=10 ; 
                else if ( input_gpu[V_INDEX]==230 && input_gpu[U_INDEX]!=20 )
                    input_gpu[U_INDEX]-=10 ; 
                else if ( input_gpu[U_INDEX]==20 && input_gpu[V_INDEX]!=20 )
                    input_gpu[V_INDEX]-=10 ; 
                else if ( input_gpu[V_INDEX]==20 && input_gpu[U_INDEX]!=230 )
                    input_gpu[U_INDEX]+=10 ; 
                  */  
                //input_gpu[640*480 + (i/2)*640+(j/2)] = 240 ;
                //input_gpu[640*600 + (i/2)*640+(j/2)] = 10 ;
            }
        }
    }
}
#else
__global__ void gpu_Generation_changeColor(uint8_t *input_gpu , int* mv){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(input_gpu[idx] == Y_SAT /*|| input_gpu[idx] == Y_SAT+1*/ ){
            //curandState_t state;
            //curand_init( input_gpu[gpu_U_INDEX],  0,  0,  &state);
            //int result = (int)curand(&state) % 10000001;
            int move_u = mv[idx*2] ;
            int move_v = mv[idx*2+1] ;
           /* 
            if(result % 4 == 0 ){
            }else if(result % 4 == 1 ){
                move_u *= -1 ;
            }else if(result % 4 == 2 ){
                move_v *= -1 ;
            }else if(result % 4 == 3 ){
                move_u *= -1 ;
                move_v *= -1 ;
            }*/
             
            if( move_u + input_gpu[gpu_U_INDEX] > 255 )
                mv[idx*2] *= -1 ; 
            else if( move_u + input_gpu[gpu_U_INDEX] < 0 ) 
                mv[idx*2] *= -1 ;
 
            if( move_v + input_gpu[gpu_V_INDEX] > 255 ) 
                mv[idx*2+1] *= -1 ; 
            else if( move_v + input_gpu[gpu_V_INDEX] < 0 ) 
                mv[idx*2+1] *= -1 ; 
            
                input_gpu[gpu_U_INDEX]+=mv[idx*2] ; 
                input_gpu[gpu_V_INDEX]+=mv[idx*2+1] ; 
            /*
            if(input_gpu[gpu_U_INDEX] == 230 && input_gpu[gpu_V_INDEX]!=230)
                input_gpu[gpu_V_INDEX]+=10 ; 
            else if ( input_gpu[gpu_V_INDEX]==230 && input_gpu[gpu_U_INDEX]!=20 )
                input_gpu[gpu_U_INDEX]-=10 ; 
            else if ( input_gpu[gpu_U_INDEX]==20 && input_gpu[gpu_V_INDEX]!=20 )
                input_gpu[gpu_V_INDEX]-=10 ; 
            else if ( input_gpu[gpu_V_INDEX]==20 && input_gpu[gpu_U_INDEX]!=230 )
                input_gpu[gpu_U_INDEX]+=10 ; 
            */   
    }
    if(idx == CENTER)
        input_gpu[idx] = 255 ;
}
#endif
#if CPU 
void gpu_rotation(uint8_t *input_gpu , uint8_t *temp_frame ){
    input_gpu[240*640+320] = 255 ;
    for(int i = 0 ; i < 640 * 480 *1.5 ; i ++)
        temp_frame[i] = 128 ; 
    for(int i = 0 ; i < 480 ; i++){
        for(int j = 0 ; j < 640 ; j++){
            if(input_gpu[i*640+j] == 200){
                int new_i = (i - 240)*COS_12 - (j-320)*SIN_12 + 240 ;
                int new_j = (i - 240)*SIN_12 + (j-320)*COS_12 + 320 ;
                if( new_i>=0 && new_i < 480 && new_j >=0 && new_j < 640 ){
                    temp_frame[new_i*640+new_j] = input_gpu[i*640+j];
                    //temp_frame[640*480 +  new_i/2*320+new_j/2] = input_gpu[640*480 + i/2*320+j/2];
                    //temp_frame[640*600 +  new_i/2*320+new_j/2] = input_gpu[640*600 + i/2*320+j/2];
                    temp_frame[ new_U_INDEX] = input_gpu[ new_U_INDEX ];
                    temp_frame[ new_V_INDEX] = input_gpu[ new_V_INDEX ];
                }
            }    
        }
    }
    for(int i = 0 ; i < 640 * 480 *1.5 ; i ++)
        input_gpu[i] = temp_frame[i];
}
#else
__global__ void gpu_rotation(uint8_t *input_gpu , uint8_t *temp_frame , int direction , int* mv , int* temp_mv){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //temp_frame[idx] = 128 ; 
    //temp_frame[gpu_U_INDEX] = 128 ;
    //temp_frame[gpu_V_INDEX] = 128 ;
    if(input_gpu[idx]==Y_SAT){
        int x = blockIdx.x-240 ,  y = threadIdx.x-320 ;
        int new_i = x*COS_12 - direction*y*SIN_12 + 240 ; 
        int new_j = x*SIN_12*direction + y*COS_12 + 320 ; 
        if( new_i>=0 && new_i < 480 && new_j >=0 && new_j < 640 ){
            temp_frame[ new_i*640+new_j ] = input_gpu[idx];
            temp_frame[ new_U_INDEX ] = input_gpu[ gpu_U_INDEX ];
            temp_frame[ new_V_INDEX ] = input_gpu[ gpu_V_INDEX ];
            temp_mv[(new_i* 640 + new_j)*2 ] = mv[idx*2];
            temp_mv[(new_i* 640 + new_j)*2+1 ] = mv[idx*2+1];
        }
    }
}
__global__ void gpu_tran(uint8_t *input_gpu , uint8_t *temp_frame , int* mv , int* temp_mv){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input_gpu[idx] = temp_frame[idx];
    input_gpu[ gpu_U_INDEX ] = temp_frame[ gpu_U_INDEX ] ;
    input_gpu[ gpu_V_INDEX ] = temp_frame[ gpu_V_INDEX ] ; 
    if( temp_frame[gpu_U_INDEX] != 128 && temp_frame[idx]!=Y_SAT )
        input_gpu[idx] = Y_SAT + 1 ;
    mv[idx*2] = temp_mv[idx*2];
    mv[idx*2+1] = temp_mv[idx*2+1];
    if(idx == CENTER)
        input_gpu[idx] = 255 ;
}
#endif

__global__ void gpu_extention(uint8_t *input_gpu , uint8_t *temp_frame , int *mv , int *temp_mv){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int radius = 10 ; 
    //temp_frame[idx] = 128 ;//input_gpu[idx] ; 
    //temp_frame[gpu_U_INDEX] = 128;//input_gpu[gpu_U_INDEX] ;
    //temp_frame[gpu_V_INDEX] = 128;//input_gpu[gpu_V_INDEX] ;
    if(input_gpu[idx]==Y_SAT){
        for(int i = -10 ; i<=10 ; i++ ){
            for(int j = -10 ; j <= 10 ; j ++){
                if( sqrtf(i*i+j*j) < 10){
                    temp_frame[(blockIdx.x-i)*640 + (threadIdx.x-j)] = Y_SAT+1 ; 
                    temp_frame[640*480 + (blockIdx.x-i)/2*320 + (threadIdx.x-j)/2 ] = input_gpu[gpu_U_INDEX];
                    temp_frame[640*600 + (blockIdx.x-i)/2*320 + (threadIdx.x-j)/2 ] = input_gpu[gpu_V_INDEX];
                    temp_mv[  ((blockIdx.x-i)*640 + (threadIdx.x-j) )*2] = mv[idx*2] ; 
                    temp_mv[  ((blockIdx.x-i)*640 + (threadIdx.x-j) )*2 +1] = mv[idx*2+1] ;
                }
            }
        }
        temp_frame[idx] = input_gpu[idx] ; 
        temp_mv[idx*2] = mv[idx*2] ; 
        temp_mv[idx*2+1] = mv[idx*2+1] ; 
        //temp_frame[gpu_U_INDEX] = input_gpu[gpu_U_INDEX] ;
        //temp_frame[gpu_V_INDEX] = input_gpu[gpu_V_INDEX] ;
    } 
}


void Lab2VideoGenerator::Generate(uint8_t *yuv , uint8_t * temp_frame , int* mv , int* temp_mv) {
    //extent the current frame
    

    #if CPU
    uint8_t *temp_frame = new uint8_t[640*720];
    uint8_t *temp_2_frame = new uint8_t[640*480];
	#else
    cudaMemset(temp_frame , 128 , W*H*1.5);
    cudaMemset(temp_mv , 0 , W*H*2);

    #endif
    //printf(" impl->t = %d \n " , impl->t ); 
	//if (impl->t < 5){
        //cudaMemset(yuv, 128, W*H);
        //cudaMemset(yuv+H*W, 128, W*H/2);
    //}else{
        //printf(t)
    //    gpu_Generation<<<1,1>>>(yuv,temp_frame.get_gpu_rw() ); 
    //    cudaMemset(yuv+H*W, 128, W*H/2);
    //}
    
	if (impl->t < 5){
        //printf( "starting:  t = %d \n" , impl->t );
        #if CPU
        for(int i = 0 ; i < 640*480*1.5 ; i++){
            yuv[i] = 128 ;
        }
        #else
            cudaMemset(yuv, 128, W*H*1.5);
        #endif
        
    }
    else if (impl-> t == 5  ){
        //printf( "firstframe:  t = %d \n" , impl->t );
        #if CPU
            gpu_Generation_firstframe(yuv,temp_frame,temp_2_frame,impl->t , mv);
        #else
            cudaMemset(yuv+W*H, 128, W*H/2);
            cudaMemset(temp_mv , 0 , W*H*2);
            gpu_Generation_firstframe<<<480,640>>>(yuv , impl->t , mv);
            gpu_extention<<<480,640>>>(yuv , temp_frame , mv , temp_mv);
            gpu_tran<<<480,640>>>    (yuv , temp_frame , mv , temp_mv);
        #endif
    }
    else if (impl->t > 180 || (impl-> t > 5 && impl->t < 100 )){ 
        //printf( "change color:  t = %d \n" , impl->t );
        #if CPU
            gpu_Generation_changeColor(yuv);
        #else
            gpu_Generation_changeColor<<<480,640>>>(yuv , mv );
            //cudaMemset(temp_mv , 0 , W*H*2);
            cudaMemset(temp_mv , 0 , W*H*2);
            gpu_extention<<<480,640>>>(yuv , temp_frame , mv , temp_mv);
            gpu_tran<<<480,640>>>    (yuv , temp_frame , mv , temp_mv);
        #endif
    }
    else{ 
        //gpu_Generation_changeColor(yuv);
        //printf( "rotation :  t = %d \n" , impl->t );
        #if CPU
            gpu_Generation_changeColor(yuv);
            gpu_rotation(yuv , temp_frame);
        #else
            gpu_Generation_changeColor<<<480,640>>>(yuv , mv);
            
            cudaMemset(temp_frame , 128 , W*H);
            if (impl->t / 40 % 2 == 1 ){
                printf("do right rotate \n ") ;
                gpu_rotation<<<480,640>>>(yuv , temp_frame , -1 , mv , temp_mv);
            }else{ 
                printf("do left rotate \n ") ;
                gpu_rotation<<<480,640>>>(yuv , temp_frame , 1 , mv , temp_mv);
            }
            gpu_tran<<<480,640>>>    (yuv , temp_frame , mv , temp_mv);
            //cudaMemset(temp_frame.get_gpu_wo() , 128 , W*H);
            gpu_extention<<<480,640>>>(yuv , temp_frame , mv , temp_mv);
            //gpu_extention<<<480,640>>>(temp_frame.get_gpu_rw() , yuv);
            gpu_tran<<<480,640>>>    (yuv , temp_frame , mv , temp_mv);
        #endif
    }
    //rotation the extented frame   
    ++(impl->t);
}



