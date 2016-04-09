#include "lab2.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 30;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

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

__global__ void gpu_Generation(uint8_t *input_gpu, int fsize , int t) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //extent the current frame
    //unsigned char *tempframe = new unsigned char[640*480];
    uint8_t *temp_frame = new uint8_t[640*480];
    
    for(int i = 0 ; i < 640*480 ; i++){
        temp_frame[i] = 128 ; 
    }
    for(int i = 0 ; i < 640 ; i++){
        for(int j = 0 ; j < 480 ; j++){
            int new_i = (i - 320)*1.1 + 320 ;
            int new_j = (j - 240)*1.1 + 240 ;
            //if(t==15){
            //    printf("at t=5 i,j = (%d,%d) => newij = (%d , %d ) \n" , i ,j , new_i , new_j);
            //}
            if( new_i>=0 && new_i < 640 && new_j >=0 && new_j < 480)
                temp_frame[new_i*480+new_j] = input_gpu[i*480+j];
        }
    }
    for(int i = 300 ; i < 340 ; i ++)
        for(int j = 220 ; j < 260 ; j++) 
            temp_frame[i*480+j] = 255 ; 

    for(int i = 0 ; i < 640*480 ; i++){
        input_gpu[i] = temp_frame[i] ; 
    }
}
void Lab2VideoGenerator::Generate(uint8_t *yuv) {
    //extent the current frame
    //unsigned char *tempframe = new unsigned char[640*480];
	//MemoryBuffer<uint8_t> a (H*W*1.5);
	//auto b = frameb.CreateSync(H*W*1.5);
    
	if (impl->t < 5){
        cudaMemset(yuv, 0, W*H);
    //    cudaMemset(yuv+H*W, 128, W*H/2);
    }else{
        gpu_Generation<<<1,1>>>(yuv,H*W*1.5,impl->t); 
    //    cudaMemset(yuv+H*W, 128, W*H/2);
    }//for(int i = 0 ; )
    //rotation the extented frame   
    //set the center a block
    //
    //cudaMemset(yuv, (impl->t)*255/NFRAME, W*H/2);
    //cudaMemset(yuv+W*H/2, 255-((impl->t)*255/NFRAME), W*H/2);
	//cudaMemset(yuv+W*H, 128, W*H/2);
    //printf("current t = %d \n" , impl->t);
	//if (impl->t < 10)
    //    cudaMemset(yuv, 0, W*H);
	//else{
    //    cudaMemget
    //    cudaMemset(yuv, yuv[3] , W*H);
	//}
    //cudaMemset(yuv+W*H, 128, W*H/2);
    ++(impl->t);
}



