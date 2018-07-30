
#define RERES 512
#define RERE_S_DOUBLED 512
#define F(AVR,i,j) AVR[(i)+RERES*(j)]
#define G(AVR,i,j) AVR[(i)+RERE_S_DOUBLED*(j)]
#define G_R(AVR,i,j) AVR[0+4*(i)+4*RERE_S_DOUBLED*(j)]
#define G_G(AVR,i,j) AVR[1+4*(i)+4*RERE_S_DOUBLED*(j)]
#define G_B(AVR,i,j) AVR[2+4*(i)+4*RERE_S_DOUBLED*(j)]
kernel void clear_float(global float *A,const float x){
    size_t n = get_global_id(0);
    A[n] = x;
}

float e_2(float x,float x0,float y,float y0,float size){
    return exp(-(pown(x-x0,2)+pown(y-y0,2))/size);
}
float e_4(float x,float x0,float y,float y0,float size){
    return exp(-(pown(x-x0,4)+pown(y-y0,4))/size);
}
void Poisson(int i,int j,global float *X,float a,float b,float c){
    float pre = F(X,i,j);
    if(i > 0 && i < RERES-1 && j > 0 && j < RERES-1) {
        pre = (F(X,i+1,j)+F(X,i-1,j)+F(X,i,j+1)+F(X,i,j-1)+a*b)/c;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    F(X,i,j) = pre;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
void Advection_SMI(int i,int j,global float *U,global float *V,float dt,float d){
    float U_interpolated = F(U,i,j);
    float V_interpolated = F(V,i,j);
    if(i > 0 && i < RERES-1 && j > 0 && j < RERES-1){
        float i_d = (i - F(U,i,j)*dt/d);
        float j_d = (j - F(V,i,j)*dt/d);
        i_d = fmin(fmax(i_d,0.0f),1.0f*(RERES-1));
        j_d = fmin(fmax(j_d,0.0f),1.0f*(RERES-1));
        int i_dinte = (int)i_d;
        int j_dinte = (int)j_d;
        float i_dfrac = i_d-(float)i_dinte;
        float j_dfrac = j_d-(float)j_dinte;
        U_interpolated = F(U,i_dinte,j_dinte)*(1.0f-i_dfrac)*(1.0f-j_dfrac)+F(U,i_dinte,j_dinte+1)*(1.0f-i_dfrac)*(j_dfrac)+F(U,i_dinte+1,j_dinte)*(i_dfrac)*(1.0f-j_dfrac)+F(U,i_dinte+1,j_dinte+1)*(i_dfrac)*(j_dfrac);
        V_interpolated = F(V,i_dinte,j_dinte)*(1.0f-i_dfrac)*(1.0f-j_dfrac)+F(V,i_dinte,j_dinte+1)*(1.0f-i_dfrac)*(j_dfrac)+F(V,i_dinte+1,j_dinte)*(i_dfrac)*(1.0f-j_dfrac)+F(V,i_dinte+1,j_dinte+1)*(i_dfrac)*(j_dfrac);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    F(U,i,j) = U_interpolated;
    F(V,i,j) = V_interpolated;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
void Advection(int i,int j,global float *A,global float *UR,global float *VR,float dt,float d){
    float A_interpolated = G(A,i,j);
    if(i > 0 && i < RERE_S_DOUBLED-1 && j > 0 && j < RERE_S_DOUBLED-1){
        float i_d = (i - G(UR,i,j)*dt/d);
        float j_d = (j - G(VR,i,j)*dt/d);
        i_d = fmin(fmax(i_d,0.0f),1.0f*(RERE_S_DOUBLED-1));
        j_d = fmin(fmax(j_d,0.0f),1.0f*(RERE_S_DOUBLED-1));
        int i_dinte = (int)i_d;
        int j_dinte = (int)j_d;
        float i_dfrac = i_d-(float)i_dinte;
        float j_dfrac = j_d-(float)j_dinte;
        A_interpolated = G(A,i_dinte,j_dinte)*(1.0f-i_dfrac)*(1.0f-j_dfrac)+G(A,i_dinte,j_dinte+1)*(1.0f-i_dfrac)*(j_dfrac)+G(A,i_dinte+1,j_dinte)*(i_dfrac)*(1.0f-j_dfrac)+G(A,i_dinte+1,j_dinte+1)*(i_dfrac)*(j_dfrac);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    G(A,i,j) = A_interpolated;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
void Diffusion(int i,int j,global float *U,global float *V,float d,float dt,float Re){
    float U_before = F(U,i,j);
    float V_before = F(V,i,j);
    for (int ii = 0; ii < 5; ii++){
        Poisson(i,j,U,pown(d,2)*Re/dt,U_before,4.0f+pown(d,2)*Re/dt);
        Poisson(i,j,V,pown(d,2)*Re/dt,V_before,4.0f+pown(d,2)*Re/dt);
    }
}
void External_force(int i,int j,global float *U,global float *V,global float *Fx,global float *Fy,float dt){
    float U_before = F(U,i,j) + dt*F(Fx,i,j);
    float V_before = F(V,i,j) + dt*F(Fy,i,j);
    barrier(CLK_GLOBAL_MEM_FENCE);
    F(U,i,j) = U_before;
    F(V,i,j) = V_before;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
void Pressure(int i,int j,global float *P,global float *U,global float *V,float d){
    float div_UV = (F(U,i+1,j)-F(U,i-1,j)+F(V,i,j+1)-F(V,i,j-1))/(2.0f*d);
    for (int ii = 0; ii < 7; ii++){
        Poisson(i,j,P,-pown(d,2),div_UV,4.0f);
    }
}
void update(int i,int j,global float *P,global float *U,global float *V,float d){
    float U_before = F(U,i,j);
    float V_before = F(V,i,j);
    if(i > 0 && i < RERES-1 && j > 0 && j < RERES-1){
        U_before = F(U,i,j) - 1.0f*(F(P,i+1,j)-F(P,i-1,j))/(2.0f*d);
        V_before = F(V,i,j) - 1.0f*(F(P,i,j+1)-F(P,i,j-1))/(2.0f*d);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    //F(U,i,j) = U_before*0.999f;
    //F(V,i,j) = V_before*0.999f;
    F(U,i,j) = U_before;
    F(V,i,j) = V_before;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
void Boundary(int i,int j,global float *P,global float *U,global float *V){
    if(i == 0 || i == RERES-1 || j == 0 || j == RERES-1){
        F(U,i,j) = -F(U,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
        F(V,i,j) = -F(V,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
        F(P,i,j) = F(P,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
    }
}
kernel void AddForce(global float *Fx, global float *Fy, const float diffX, const float diffY, const float posX, const float posY, const int mou) {
	size_t n = get_global_id(0);
	float siz = 0.0001*0.4;
	float j = (float)(n / RERES);
	float i = (float)(n%RERES);

	float hf = 1.0f*RERES;
	float expF = (float)(10000.0f*exp(-(pow((float)(i - posX) / hf, 2.0f) + pow((float)(j + (-RERES+posY)) / hf, 2.0f)) / siz));
    float IntenseX = mou*diffX*expF;
    float IntenseY = -mou*diffY*expF;
    Fx[n] = IntenseX;
    Fy[n] = IntenseY;
}
kernel void drawCircle(global float *A,const float posX, const float posY, const float mou, const float size, const float strength){
    int n = get_global_id(0);
    int j = n/RERE_S_DOUBLED;
    int i = n%RERE_S_DOUBLED;
    float x = i*1.0f/RERE_S_DOUBLED;
    float y = j*1.0f/RERE_S_DOUBLED;
    float x0 = posX/RERE_S_DOUBLED;
    float y0 = posY/RERE_S_DOUBLED;
    bool is_incircle = (pown(x-x0,2)+pown(y-y0,2) < pown(size,2));
    G(A,i,j) = 0.998f*G(A,i,j) + 1.0f*strength*mou*is_incircle;
}
kernel void kernel_main(global float *P,global float *U,global float *V,global float *Fx,global float *Fy,const float d,const float dt,const float Re){
    int n = get_global_id(0);
    int i = n%RERES;
    int j = n/RERES;
    External_force(i,j,U,V,Fx,Fy,dt);
    Advection_SMI(i,j,U,V,dt,d);
    Diffusion(i,j,U,V,d,dt,Re);
    Pressure(i,j,P,U,V,d);
    update(i,j,P,U,V,d);
    Boundary(i,j,P,U,V);
}
kernel void XY512Func(global float *UR,global float *VR,global float *U,global float *V){
    size_t n = get_global_id(0);
    int x = n%RERE_S_DOUBLED;
    int y = n/RERE_S_DOUBLED;
    int i = x/2;
    int j = y/2;
    int p = (x%2 != 0) ? 1 : -1;
    int q = (y%2 != 0) ? 1 : -1;
    float X = 0.75f;
    float Y = 0.75f;
    if(x > 0 && x < RERE_S_DOUBLED-1 && y > 0 && y < RERE_S_DOUBLED-1){
        G(UR,x,y) = X*Y*F(U,i,j)+(1-X)*Y*F(U,i+p,j)+X*(1-Y)*F(U,i,j+q)+(1-X)*(1-Y)*F(U,i+p,j+q);
        G(VR,x,y) = X*Y*F(V,i,j)+(1-X)*Y*F(V,i+p,j)+X*(1-Y)*F(V,i,j+q)+(1-X)*(1-Y)*F(V,i+p,j+q);
    }
}
kernel void vorticity(global float *U,global float *V,global float *vorticity,const float d){
    int n = get_global_id(0);
    int i = n%RERES;
    int j = n/RERES;
    if(i > 0 && i < RERES-1 && j > 0 && j < RERES-1){
        F(vorticity,i,j) = (F(V,i+1,j)-F(V,i-1,j)-F(U,i,j+1)+F(U,i,j-1))/(2.0f*d);
    }
}
kernel void convertFloatToUchar(global float *data_float,global uchar *data_uchar,const float strength){
    size_t n = get_global_id(0);
    data_uchar[4*n] = (float)fmin(fmax(data_float[n]*strength,0.0f),255.0f);
    data_uchar[4*n+1] = 0.0f;
    data_uchar[4*n+2] = (float)fmin(fmax(-data_float[n]*strength,0.0f),255.0f);
    data_uchar[4*n+3] = 0.0f;
}
kernel void convertFloatToUcharMix(global float *A_R,global float *A_G,global float *A_B,global uchar *data_uchar,const float strength){
    size_t n = get_global_id(0);
    data_uchar[4*n] = (float)fmin(fmax(A_R[n]*strength,0.0f),255.0f);
    data_uchar[4*n+1] = (float)fmin(fmax(A_G[n]*strength,0.0f),255.0f);
    data_uchar[4*n+2] = (float)fmin(fmax(A_B[n]*strength,0.0f),255.0f);
    data_uchar[4*n+3] = 0.0f;
}
kernel void BallInitialize(global float *BallX,global float *BallY,const float x,const float y){
    size_t n = get_global_id(0);
    int i = n%RERE_S_DOUBLED;
    int j = n/RERE_S_DOUBLED;
    BallX[n] = i/3.0f+x;
    BallY[n] = j/3.0f+y;
}
kernel void A_Initialize(global float *A,const float x,const float y,const float size,const float strength){
    size_t n = get_global_id(0);
    int i = n%RERE_S_DOUBLED;
    int j = n/RERE_S_DOUBLED;
    G(A,i,j) = strength*e_2(i*1.0f/RERE_S_DOUBLED,x,j*1.0f/RERE_S_DOUBLED,y,size);
}
kernel void A_Update(global float *A,global float *UR,global float *VR,const float dt,const float d){
    size_t n = get_global_id(0);
    int i = n%RERE_S_DOUBLED;
    int j = n/RERE_S_DOUBLED;
    Advection(i,j,A,UR,VR,dt,d);
}
kernel void RClear(global uchar *R){
    size_t n = get_global_id(0);
    R[4*n]=0;
    R[4*n+1]=0;
    R[4*n+2]=0;
}
kernel void BallFunc(global float *BallX,global float *BallY,global float *U,global float *V,global uchar *R,const float dt,const float c){
    size_t n = get_global_id(0);
    uint kx = (uint)((float)floor((float)BallX[n]));
    uint ky = (uint)((float)floor((float)BallY[n]));
    
    BallX[n] = BallX[n] + c*U[(uint)(kx+512*ky)];
    BallY[n] = BallY[n] + c*V[(uint)(kx+512*ky)];
    
    
    R[(uint)(4*(kx+512*ky))]=(uchar)((float)fmin((float)(R[(uint)(4*(kx+512*ky))]+255*0.6f),255));
}
kernel void Wconfinement(global float *W,global float *WconfinementX,global float *WconfinementY,global float *Fx,global float *Fy,const float d,const float strength){
    int n = get_global_id(0);
    int i = n%RERES;
    int j = n/RERES;
    float gradWx,gradWy,length_gradW,WCX,WCY;
    if(i > 0 && i < RERES-1 && j > 0 && j < RERES-1){
        gradWx = (fabs(F(W,i+1,j))-fabs(F(W,i-1,j)))/(2.0f*d);
        gradWy = (fabs(F(W,i,j+1))-fabs(F(W,i,j-1)))/(2.0f*d);
        length_gradW = sqrt(pown(gradWx,2)+pown(gradWy,2));
        if(length_gradW > 0 && fabs(F(W,i,j)) > 50.5f){
            WCX = F(W,i,j)*gradWy/length_gradW;
            WCY = -F(W,i,j)*gradWx/length_gradW;
            F(WconfinementX,i,j) = WCX;
            F(WconfinementY,i,j) = WCY;
            F(Fx,i,j) = F(Fx,i,j) + strength*WCX;
            F(Fy,i,j) = F(Fy,i,j) + strength*WCY;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(i == 0 || i == RERES-1 || j == 0 || j == RERES-1){
        F(WconfinementX,i,j) = F(WconfinementX,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
        F(WconfinementY,i,j) = F(WconfinementY,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
        F(Fx,i,j) = F(Fx,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
        F(Fx,i,j) = F(Fy,i+1*(i==0)-1*(i==RERES-1),j+1*(j==0)-1*(j==RERES-1));
    }
}
kernel void sharp_filter(global uchar *R){
    size_t n = get_global_id(0);
    int i = n%RERE_S_DOUBLED;
    int j = n/RERE_S_DOUBLED;
    float RR = G_R(R,i,j);
    float GG = G_G(R,i,j);
    float BB = G_B(R,i,j);
    if(i > 0 && i < RERE_S_DOUBLED-1 && j > 0 && j < RERE_S_DOUBLED-1){
        RR = 5*G_R(R,i,j)-G_R(R,i+1,j)-G_R(R,i-1,j)-G_R(R,i,j+1)-G_R(R,i,j-1);
        GG = 5*G_G(R,i,j)-G_G(R,i+1,j)-G_G(R,i-1,j)-G_G(R,i,j+1)-G_G(R,i,j-1);
        BB = 5*G_B(R,i,j)-G_B(R,i+1,j)-G_B(R,i-1,j)-G_B(R,i,j+1)-G_B(R,i,j-1);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    G_R(R,i,j) = fmin(fmax(RR,0.0f),255.0f);
    G_G(R,i,j) = fmin(fmax(GG,0.0f),255.0f);
    G_B(R,i,j) = fmin(fmax(BB,0.0f),255.0f);
}
kernel void Rkuro(global uchar *R){
    size_t n = get_global_id(0);
    R[4*n+1]=R[4*n];
    R[4*n+2]=R[4*n];
}

