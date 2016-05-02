#include "lab3.h"
#include <cstdio>
#include <cassert>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void DownSampling_2 ( const float *a , float *b , const int wt , const int ht , const int Scale);
__global__ void UpSampling_2 ( const float *a , float *b , const int wt ,const int ht  , const int Scale);
__global__ void Initial_solution( const float *background , float *buf1 , 
                        const int wb , const int hb , const int wt , const int ht , const int oy , const int ox  ) ;

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
        //bool b_up    = ( yt!=0  and mask[curt] > 127.0f and mask[up_n] > 127.0f)? false : true ; 
        //bool b_down  = ( yt!=(ht-1) and mask[curt] > 127.0f and mask[down_n] > 127.0f)? false : true ; 
        //bool b_left  = ( xt!=0  and mask[curt] > 127.0f and mask[left_n] > 127.0f)? false : true ; 
        //bool b_right = ( xt!=(wt-1) and mask[curt] > 127.0f and mask[right_n] >127.0f)? false : true ; 
        bool b_up    = ( yt==0     )? true:false   ; 
        bool b_down  = ( yt==(ht-1))? true:false   ; 
        bool b_left  = ( xt==0     )? true:false   ; 
        bool b_right = ( xt==(wt-1))? true:false   ; 
    
        //const int yb = oy+yt, xb = ox+xt;
        //const int curb_up    = wb*(yb-1) +xb  ;
        //const int curb_down  = wb*(yb+1) +xb  ;
        //const int curb_left  = wb*yb     +xb-1;
        //const int curb_right = wb*yb     +xb+1;
    
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
        
    int n_of_b =   4 -( (int)b_up + (int)b_down + (int)b_left + (int)b_right ) ;
    //if( yt < ht and xt < wt and mask[curt]>127.0f){
    if( yt >= 0 and xt>= 0 and yt < ht and xt < wt ){
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
        //gradient[curt*3] =  128;// target[curt*3] ;
        //gradient[curt*3+1] = 0;//target[curt*3+1] ;
        //gradient[curt*3+2] = 0;//target[curt*3+2] ;
    }
}

__global__ void PoissonImageCloningIteration(
    const float *background,
	const float *fixed,
	const float *mask,
	const float *target,
	float *buf2,
	const int wb, const int hb, const int wt, const int ht,
    const int oy , const int ox , const float w , const int Scale)
{	
    const int yt = (blockIdx.y * blockDim.y + threadIdx.y);
	const int xt = (blockIdx.x * blockDim.x + threadIdx.x);
	const int curt = wt*yt+xt;
    
    const int up_n    = wt*(yt-1) + xt   ; 
    const int down_n  = wt*(yt+1) + xt   ; 
    const int left_n  = wt*yt     + xt-1 ; 
    const int right_n = wt*yt     + xt+1 ; 
    
    float Rt_up    = target[up_n*3] ; 
    float Rt_down  = target[down_n*3] ; 
    float Rt_left  = target[left_n*3] ; 
    float Rt_right = target[right_n*3] ; 
    float Gt_up    = target[up_n*3+1] ; 
    float Gt_down  = target[down_n*3+1] ; 
    float Gt_left  = target[left_n*3+1] ; 
    float Gt_right = target[right_n*3+1] ; 
    float Bt_up    = target[up_n*3+2] ; 
    float Bt_down  = target[down_n*3+2] ; 
    float Bt_left  = target[left_n*3+2] ; 
    float Bt_right = target[right_n*3+2] ; 
    //if( mask[up_n] < 127.0f or mask[down_n] < 127.0f )
        const int yb = oy+(yt*Scale) , xb = ox+(xt*Scale) ;
        const int curb       = wb*(yb) +xb  ;
        const int curb_up    = wb*(yb-1) +xb  ;
        const int curb_down  = wb*(yb+1) +xb  ;
        const int curb_left  = wb*yb     +xb-1;
        const int curb_right = wb*yb     +xb+1;

    if( yt>= 0 and xt >= 0 and  yt < ht and xt < wt and mask[curt]>127.0f){
    //mask boundary condition
        bool b_up    = ( yt==0  or mask[up_n]<127.0f)? true : false ; 
        bool b_down  = ( yt==(ht-1) or mask[down_n]<127.0f)? true : false ; 
        bool b_left  = ( xt==0  or mask[left_n]<127.0f)? true : false ; 
        bool b_right = ( xt==(wt-1) or mask[right_n]<127.0f)? true : false ; 
        int n_of_b = 4 ; // - ( (int)b_up + (int)b_down + (int)b_left + (int)b_right ) ;
        if(b_up == true or b_down == true or b_left == true or b_right == true ){
            buf2[curt*3] = background[curb*3+0] ;//target[curt*3] ;
            buf2[curt*3+1] = background[curb*3+1] ;
            buf2[curt*3+2] = background[curb*3+2] ;
        }
        else{
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
        }
        // SOR??
        float a =  buf2[curt*3]*w + (1-w)*target[curt*3+0] ;
        float b =  buf2[curt*3+1]*w + (1-w)*target[curt*3+1] ;
        float c =  buf2[curt*3+2]*w + (1-w)*target[curt*3+2] ;
        if( a > 255 || a < 0 || b > 255 || b < 0 || c>255 || c<0)
        {}
        else{
        buf2[curt*3]   =  buf2[curt*3]   *w + (1-w)*target[curt*3+0]     ;
        buf2[curt*3+1] =  buf2[curt*3+1] *w + (1-w)*target[curt*3+1]     ;
        buf2[curt*3+2] =  buf2[curt*3+2] *w + (1-w)*target[curt*3+2]     ;
        }

    }else if( yt>= 0 and xt >= 0 and  yt < ht and xt < wt){
        buf2[curt*3] = background[curb*3+0] ;//target[curt*3] ;
        buf2[curt*3+1] = background[curb*3+1] ;
        buf2[curt*3+2] = background[curb*3+2] ;
    }

        
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox)
{
    printf("wb , hb , wt , ht , oy , ox  = %d , %d , %d , %d , %d , %d \n" , 
            wb , hb , wt , ht , oy , ox );
    const int Num_iter = 1000 ;	
    float *fixed , *buf1, *buf2 ; 
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    CalculateFixed<<<gdim, bdim>>>(  background, target, mask, fixed,
                                        wb, hb, wt, ht, oy, ox );
    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    Initial_solution<<<gdim,bdim>>>( background , buf1 , wb , hb , wt , ht , oy , ox  ) ;
    cudaMemcpy(buf2, buf1, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    
    float w = 1 ; 
    //bool *isConv ;
    //cudaMalloc(&isConv, 1*sizeof(bool));
    // Down Scaling
    int Scale = 4 ; 
    
    float *fixed_4 , *buf1_4, *buf2_4 , *mask_4 , *target_4; 
    cudaMalloc(&fixed_4, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&buf1_4, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&buf2_4, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&mask_4, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&target_4, 3*wt/Scale*ht/Scale*sizeof(float));

    dim3 gdim_4(CeilDiv(wt/Scale,32), CeilDiv(ht/Scale,16)), bdim_4(32,16);
    DownSampling_2<<< gdim_4,bdim_4 >>> (target , target_4 , wt/Scale , ht/Scale , Scale );
    DownSampling_2<<< gdim_4,bdim_4 >>> (buf1  , buf1_4  , wt/Scale , ht/Scale , Scale );
    DownSampling_2<<< gdim_4,bdim_4 >>> (buf2  , buf2_4  , wt/Scale , ht/Scale , Scale );
    DownSampling_2<<< gdim_4,bdim_4 >>> (mask  , mask_4  , wt/Scale , ht/Scale , Scale );
    CalculateFixed<<< gdim_4,bdim_4 >>> (background, target_4, mask_4, fixed_4,
                                        wb, hb, wt/Scale, ht/Scale, oy, ox );
    
    for (int i = 0; i < Num_iter*6/10; ++i) {
        PoissonImageCloningIteration<<<gdim_4, bdim_4>>>(background, fixed_4, mask_4, buf1_4, buf2_4,
                                        wb, hb, wt/Scale, ht/Scale, oy, ox ,w , Scale);
        PoissonImageCloningIteration<<<gdim_4, bdim_4>>>(background, fixed_4, mask_4, buf2_4, buf1_4,
                                        wb, hb, wt/Scale, ht/Scale, oy, ox ,w , Scale);
    }
    
    Scale = 2 ;

    float *fixed_2 , *buf1_2, *buf2_2 , *mask_2 , *target_2; 
    cudaMalloc(&fixed_2, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&buf1_2, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&buf2_2, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&mask_2, 3*wt/Scale*ht/Scale*sizeof(float));
    cudaMalloc(&target_2, 3*wt/Scale*ht/Scale*sizeof(float));
    
    dim3 gdim_2(CeilDiv(wt/Scale,32), CeilDiv(ht/Scale,16)), bdim_2(32,16);
    
    UpSampling_2<<< gdim_2,bdim_2 >>> (buf1_4  , buf1_2 , wt/Scale  , ht/Scale , Scale );
    UpSampling_2<<< gdim_2,bdim_2 >>> (buf2_4  , buf2_2 , wt/Scale  , ht/Scale , Scale );
    
    DownSampling_2<<< gdim_2,bdim_2 >>> (target , target_2 , wt/Scale , ht/Scale , Scale );
    //DownSampling_2<<< gdim_2,bdim_2 >>> (buf1  , buf1_2  , wt/Scale , ht/Scale , Scale );
    //DownSampling_2<<< gdim_2,bdim_2 >>> (buf2  , buf2_2  , wt/Scale , ht/Scale , Scale );
    DownSampling_2<<< gdim_2,bdim_2 >>> (mask  , mask_2  , wt/Scale , ht/Scale , Scale );
    CalculateFixed<<< gdim_2,bdim_2 >>> (background, target_2, mask_2, fixed_2,
                                        wb, hb, wt/Scale, ht/Scale, oy, ox );
   

    for (int i = 0; i < Num_iter*3/10; ++i) {
        PoissonImageCloningIteration<<<gdim_2, bdim_2>>>(background, fixed_2, mask_2, buf1_2, buf2_2,
                                        wb, hb, wt/Scale, ht/Scale, oy, ox ,w , Scale);
        PoissonImageCloningIteration<<<gdim_2, bdim_2>>>(background, fixed_2, mask_2, buf2_2, buf1_2,
                                        wb, hb, wt/Scale, ht/Scale, oy, ox ,w , Scale);
    }
    //UpSampling_2<<< gdim,bdim >>> (fixed_2 , fixed , wt , ht , Scale);
    UpSampling_2<<< gdim,bdim >>> (buf1_2  , buf1 , wt  , ht , Scale );
    UpSampling_2<<< gdim,bdim >>> (buf2_2  , buf2 , wt  , ht , Scale );
    //UpSampling_2<<< gdim,bdim >>> (mask_2  , mask);


    // iterate
    //float w = 3 ;
    //float Num_iter = 1000 ;
    w = 1 ;

    for (int i = 0; i < Num_iter*1/10; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(background, fixed, mask, buf1, buf2,
                                        wb, hb, wt, ht, oy, ox ,w ,1 );
        PoissonImageCloningIteration<<<gdim, bdim>>>(background, fixed, mask, buf2, buf1,
                                        wb, hb, wt, ht, oy, ox ,w ,1 );
        w = 1 + ( (w-1) / 1.1 ) ;
    }


    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	if(Num_iter != 0 )
    SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
    else
    SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, fixed, mask, output,
		wb, hb, wt, ht, oy, ox
	);
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
}

__global__ void DownSampling_2 ( const float *a , float *b , const int wt , const int ht , const int Scale){
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt_a = (wt*Scale)*(yt*Scale)+(xt*Scale);
	const int curt_a_r = (wt*Scale)*(yt*Scale+1)+(xt*Scale);
	const int curt_a_d = (wt*Scale)*(yt*Scale)+(xt*Scale+1);
	const int curt_a_rd = (wt*Scale)*(yt*Scale+1)+(xt*Scale+1);
	const int curt_b = wt*(yt)+(xt);
   
    if( yt>= 0 and xt >= 0 and  yt < ht and xt < wt){
    //b[curt_b*3+0] = a[curt_a*3+0] + ; 
        b[curt_b*3] = ( a[curt_a*3] + a[curt_a_r*3] + a[curt_a_d*3] + a[curt_a_rd*3] ) / 4 ;  
        b[curt_b*3+1] = ( a[curt_a*3+1] + a[curt_a_r*3+1] + a[curt_a_d*3+1] + a[curt_a_rd*3+1] ) / 4 ;  
        b[curt_b*3+2] = ( a[curt_a*3+2] + a[curt_a_r*3+2] + a[curt_a_d*3+2] + a[curt_a_rd*3+2] ) / 4 ;  
    //b[curt_b*3+1] = a[curt_a*3+1]; 
    //b[curt_b*3+2] = a[curt_a*3+2]; 
    }
}
__global__ void UpSampling_2 ( const float *a , float *b , const int wt , const int ht, const int Scale){
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt_a = (wt/Scale)*(yt/Scale)+(xt/Scale);
	const int curt_b = (wt)*yt+xt;
    
    const int curt_a_right = (wt/Scale)*(yt/Scale)+(xt/Scale+1);
    const int curt_a_down  = (wt/Scale)*(yt/Scale+1)+(xt/Scale);
    const int curt_a_rd    = (wt/Scale)*(yt/Scale+1)+(xt/Scale+1);
  
    if( yt>= 0 and xt >= 0 and  yt < ht and xt < wt){
	if(Scale == 2 ){
        if(yt%Scale == 1 and yt != (ht-1)){
            if(xt%Scale == 1 and xt !=(wt-1)){
                b[curt_b*3] = ( a[curt_a*3] + a[curt_a_right*3] + a[curt_a_down*3] + a[curt_a_rd*3] ) / 4 ;  
                b[curt_b*3+1] = ( a[curt_a*3+1] + a[curt_a_right*3+1] + a[curt_a_down*3+1] + a[curt_a_rd*3+1] ) / 4 ;  
                b[curt_b*3+2] = ( a[curt_a*3+2] + a[curt_a_right*3+2] + a[curt_a_down*3+2] + a[curt_a_rd*3+2] ) / 4 ;  
            }else{
                b[curt_b*3] = ( a[curt_a*3]     + a[curt_a_down*3]  ) / 2 ;  
                b[curt_b*3+1] = ( a[curt_a*3+1] + a[curt_a_down*3+1] ) / 2 ;  
                b[curt_b*3+2] = ( a[curt_a*3+2] + a[curt_a_down*3+2] ) / 2 ;  
            }
        }
        else{
            if(xt%Scale == 1 and xt!=(wt-1)){
                b[curt_b*3] = ( a[curt_a*3]     + a[curt_a_right*3]   ) / 2 ;  
                b[curt_b*3+1] = ( a[curt_a*3+1] + a[curt_a_right*3+1] ) / 2 ;  
                b[curt_b*3+2] = ( a[curt_a*3+2] + a[curt_a_right*3+2] ) / 2 ;  
            }else{
                b[curt_b*3] =     a[curt_a*3]   ; 
                b[curt_b*3+1] =   a[curt_a*3+1] ;  
                b[curt_b*3+2] =   a[curt_a*3+2] ;  
            }
        
        }
    }
  } 
    //b[curt_b*3+0] = a[curt_a*3+0]; 
    //b[curt_b*3+1] = a[curt_a*3+1]; 
    //b[curt_b*3+2] = a[curt_a*3+2]; 
}
__global__ void Initial_solution( const float *background , float *buf1 , 
                        const int wb , const int hb , const int wt , const int ht , const int oy , const int ox  ) {

    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = (wt)*(yt)+(xt);
    
    const int yb = oy+(yt) , xb = ox+(xt) ;
    const int curb       = wb*(yb) +xb  ;

    const float R = background[curb*3+0] ; 
    const float G = background[curb*3+1] ; 
    const float B = background[curb*3+2] ;
    //const float Y =  0.299*R + 0.587*G + 0.114*B ; 
    const float U = -0.169*R - 0.331*G + 0.500*B ; 
    const float V =  0.500*R - 0.419*G - 0.081*B ; 

    const float Rt = buf1[curt*3+0] ; 
    const float Gt = buf1[curt*3+1] ; 
    const float Bt = buf1[curt*3+2] ;
    const float Yt =  0.299*Rt + 0.587*Gt + 0.114*Bt ; 
    //const float Ut = U ; 
    //const float Vt = V ; 

    const float Rf = Yt + 1.13983*(V) ; 
    const float Gf = Yt - 0.39465*(U) -0.58060*(V); 
    const float Bf = Yt + 2.03211*(U) ; 
    
    if( yt>= 0 and xt >= 0 and  yt < ht and xt < wt){
			buf1[curt*3+0] = Rf ;
			buf1[curt*3+1] = Gf ;
			buf1[curt*3+2] = Bf ;
             
    }

}
