#include "lab3.h"
#include <cstdio>
#include <cassert>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *gradient,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox )
{	
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
    
    const int up_n    = wt*(yt-1) + xt   ; 
    const int down_n  = wt*(yt+1) + xt   ; 
    const int left_n  = wt*yt     + xt-1 ; 
    const int right_n = wt*yt     + xt+1 ; 
    
    //if( mask[up_n] < 127.0f or mask[down_n] < 127.0f )
        bool b_up    = ( yt!=0  and mask[curt] > 127.0f and mask[up_n] > 127.0f)? false : true ; 
        bool b_down  = ( yt!=(ht-1) and mask[curt] > 127.0f and mask[down_n] > 127.0f)? false : true ; 
        bool b_left  = ( xt!=0  and mask[curt] > 127.0f and mask[left_n] > 127.0f)? false : true ; 
        bool b_right = ( xt!=(wt-1) and mask[curt] > 127.0f and mask[right_n] >127.0f)? false : true ; 
    
        const int yb = oy+yt, xb = ox+xt;
        const int curb_up    = wb*(yb-1) +xb  ;
        const int curb_down  = wb*(yb+1) +xb  ;
        const int curb_left  = wb*yb     +xb-1;
        const int curb_right = wb*yb     +xb+1;
    
    float Rt_up    = (b_up    )? 0 :target[up_n*3]     ; 
    float Rt_down  = (b_down  )? 0 :target[down_n*3]   ; 
    float Rt_left  = (b_left  )? 0 :target[left_n*3]   ; 
    float Rt_right = (b_right )? 0 :target[right_n*3]  ; 
    float Gt_up    = (b_up    )? 0 :target[up_n*3+1]   ; 
    float Gt_down  = (b_down  )? 0 :target[down_n*3+1] ; 
    float Gt_left  = (b_left  )? 0 :target[left_n*3+1] ; 
    float Gt_right = (b_right )? 0 :target[right_n*3+1]; 
    float Bt_up    = (b_up    )? 0 :target[up_n*3+2]   ; 
    float Bt_down  = (b_down  )? 0 :target[down_n*3+2] ; 
    float Bt_left  = (b_left  )? 0 :target[left_n*3+2] ; 
    float Bt_right = (b_right )? 0 :target[right_n*3+2]; 
        
    int n_of_b = 4 - ( (int)b_up + (int)b_down + (int)b_left + (int)b_right ) ;
    if( yt < ht and xt < wt and mask[curt]>127.0f){
        gradient[curt*3]   = ( n_of_b * target[curt*3]   - Rt_up - Rt_down - Rt_left - Rt_right) / n_of_b ;  
        gradient[curt*3+1] = ( n_of_b * target[curt*3+1] - Gt_up - Gt_down - Gt_left - Gt_right) / n_of_b ;  
        gradient[curt*3+2] = ( n_of_b * target[curt*3+2] - Bt_up - Bt_down - Bt_left - Bt_right) / n_of_b ;  
    //mask boundary condition
       /* 
        gradient[curt*3] += background[curb_up*3]*(int)(b_up) ;
        gradient[curt*3] += background[curb_down*3]*(int)(b_down) ;
        gradient[curt*3] += background[curb_left*3]*(int)(b_left) ;
        gradient[curt*3] += background[curb_right*3]*(int)(b_right) ;
        gradient[curt*3+1] += background[curb_up*3+1]*(int)(b_up) ;
        gradient[curt*3+1] += background[curb_down*3+1]*(int)(b_down) ;
        gradient[curt*3+1] += background[curb_left*3+1]*(int)(b_left) ;
        gradient[curt*3+1] += background[curb_right*3+1]*(int)(b_right) ;
        gradient[curt*3+2] += background[curb_up*3+2]*(int)(b_up) ;
        gradient[curt*3+2] += background[curb_down*3+2]*(int)(b_down) ;
        gradient[curt*3+2] += background[curb_left*3+2]*(int)(b_left) ;
        gradient[curt*3+2] += background[curb_right*3+2]*(int)(b_right) ;*/
    }else{
        gradient[curt*3] =   target[curt*3] ;
        gradient[curt*3+1] = target[curt*3+1] ;
        gradient[curt*3+2] = target[curt*3+2] ;
    }
}

__global__ void PoissonImageCloningIteration(
    const float *background,
	const float *fixed,
	const float *mask,
	const float *target,
	float *buf2,
	const int wb, const int hb, const int wt, const int ht,
    const int oy , const int ox)
{	
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
    
    const int up_n    = wt*(yt-1) + xt   ; 
    const int down_n  = wt*(yt+1) + xt   ; 
    const int left_n  = wt*yt     + xt-1 ; 
    const int right_n = wt*yt     + xt+1 ; 
    
    float Rt_up    = (yt==0)?  0 :target[up_n*3] ; 
    float Rt_down  = (yt==(ht-1))? 0 :target[down_n*3] ; 
    float Rt_left  = (xt==0)?  0 :target[left_n*3] ; 
    float Rt_right = (xt==(wt-1))? 0 :target[right_n*3] ; 
    float Gt_up    = (yt==0)?  0 :target[up_n*3+1] ; 
    float Gt_down  = (yt==(ht-1))? 0 :target[down_n*3+1] ; 
    float Gt_left  = (xt==0)?  0 :target[left_n*3+1] ; 
    float Gt_right = (xt==(wt-1))? 0 :target[right_n*3+1] ; 
    float Bt_up    = (yt==0)?  0 :target[up_n*3+2] ; 
    float Bt_down  = (yt==(ht-1))? 0 :target[down_n*3+2] ; 
    float Bt_left  = (xt==0)?  0 :target[left_n*3+2] ; 
    float Bt_right = (xt==(wt-1))? 0 :target[right_n*3+2] ; 
    //if( mask[up_n] < 127.0f or mask[down_n] < 127.0f )
        const int yb = oy+yt, xb = ox+xt;
        const int curb_up    = wb*(yb-1) +xb  ;
        const int curb_down  = wb*(yb+1) +xb  ;
        const int curb_left  = wb*yb     +xb-1;
        const int curb_right = wb*yb     +xb+1;

    if( yt < ht and xt < wt and mask[curt]>127.0f){
    //mask boundary condition
        bool b_up    = ( yt==0  or mask[up_n]<127.0f)? true : false ; 
        bool b_down  = ( yt==(ht-1) or mask[down_n]<127.0f)? true : false ; 
        bool b_left  = ( xt==0  or mask[left_n]<127.0f)? true : false ; 
        bool b_right = ( xt==(wt-1) or mask[right_n]<127.0f)? true : false ; 
        int n_of_b = 4 ; // - ( (int)b_up + (int)b_down + (int)b_left + (int)b_right ) ;
        buf2[curt*3+0] = ( 4*fixed[curt*3+0] + Rt_up*(int)(!b_up)      + background[3*curb_up]   *(int)(b_up)
                                           + Rt_down*(int)(!b_down)  + background[3*curb_down] *(int)(b_down)
                                           + Rt_left*(int)(!b_left)  + background[3*curb_left] *(int)(b_left)
                                           + Rt_right*(int)(!b_right)+ background[3*curb_right]*(int)(b_right) ) 
                        / (float)n_of_b ;  
        buf2[curt*3+1] = ( 4*fixed[curt*3+1] + Gt_up*(int)(!b_up)      + background[3*curb_up+1]   *(int)(b_up)
                                           + Gt_down*(int)(!b_down)  + background[3*curb_down+1] *(int)(b_down)
                                           + Gt_left*(int)(!b_left)  + background[3*curb_left+1] *(int)(b_left)
                                           + Gt_right*(int)(!b_right)+ background[3*curb_right+1]*(int)(b_right) ) 
                        / (float)n_of_b ;  
        buf2[curt*3+2] = ( 4*fixed[curt*3+2] + Bt_up*(int)(!b_up)      + background[3*curb_up+2]   *(int)(b_up)
                                           + Bt_down*(int)(!b_down)  + background[3*curb_down+2] *(int)(b_down)
                                           + Bt_left*(int)(!b_left)  + background[3*curb_left+2] *(int)(b_left)
                                           + Bt_right*(int)(!b_right)+ background[3*curb_right+2]*(int)(b_right) ) 
                        / (float)n_of_b ;  
    }else{
        buf2[curt*3] = target[curt*3] ;
        buf2[curt*3+1] = target[curt*3+1] ;
        buf2[curt*3+2] = target[curt*3+2] ;
    }
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
    printf("wb , hb , wt , ht , oy , ox  = %d , %d , %d , %d , %d , %d \n" , 
            wb , hb , wt , ht , oy , ox );
	
    float *fixed , *buf1, *buf2 ; 
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    CalculateFixed<<<gdim, bdim>>>(  background, target, mask, fixed,
                                        wb, hb, wt, ht, oy, ox );
    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf2, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

    // iterate
    for (int i = 0; i < 1000; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(background, fixed, mask, buf1, buf2,
                                        wb, hb, wt, ht, oy, ox );
        PoissonImageCloningIteration<<<gdim, bdim>>>(background, fixed, mask, buf2, buf1,
                                        wb, hb, wt, ht, oy, ox );
    }


    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
}
