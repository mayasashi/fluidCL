#define RERES 256
#define RERE_S_DOUBLED 512
#define F(AVR,i,j) AVR[(i)+RERES*(j)]
#define G(AVR,i,j) AVR[(i)+RERE_S_DOUBLED*(j)]
#define G_R(AVR,i,j) AVR[0+4*(i)+4*RERE_S_DOUBLED*(j)]
#define G_G(AVR,i,j) AVR[1+4*(i)+4*RERE_S_DOUBLED*(j)]
#define G_B(AVR,i,j) AVR[2+4*(i)+4*RERE_S_DOUBLED*(j)]
kernel void clear_float(global float *A, const float x) {
	size_t n = get_global_id(0);
	A[n] = x;
}
float e_2(float x, float x0, float y, float y0, float size) {
	return exp(-(pown(x - x0, 2) + pown(y - y0, 2)) / size);
}
float e_4(float x, float x0, float y, float y0, float size) {
	return exp(-(pown(x - x0, 4) + pown(y - y0, 4)) / size);
}
float Bicubic(float i, float j, global float *A) {
	int iQ = i;
	int jQ = j;
	float iR = i - iQ;
	float jR = j - jQ;
	float f00 = F(A, iQ, jQ);
	float f01 = F(A, iQ, jQ + 1);
	float f10 = F(A, iQ + 1, jQ);
	float f11 = F(A, iQ + 1, jQ + 1);
	float fx00 = (f10 - F(A, iQ - 1, jQ)) / 2;
	float fx01 = (f11 - F(A, iQ - 1, jQ + 1)) / 2;
	float fx10 = (F(A, iQ + 2, jQ) - f00) / 2;
	float fx11 = (F(A, iQ + 2, jQ + 1) - f01) / 2;
	float fy00 = (f01 - F(A, iQ, jQ - 1)) / 2;
	float fy01 = (F(A, iQ, jQ + 2) - f00) / 2;
	float fy10 = (f11 - F(A, iQ + 1, jQ - 1)) / 2;
	float fy11 = (F(A, iQ + 1, jQ + 2) - f10) / 2;
	float fxy00 = (fy10 - (F(A, iQ - 1, jQ + 1) - F(A, iQ - 1, jQ - 1)) / 2) / 2;
	float fxy10 = ((F(A, iQ + 2, jQ + 1) - F(A, iQ + 2, jQ - 1)) / 2 - fy00) / 2;
	float fxy01 = (fy11 - (F(A, iQ - 1, jQ + 2) - F(A, iQ - 1, jQ)) / 2) / 2;
	float fxy11 = ((F(A, iQ + 2, jQ + 2) - F(A, iQ + 2, jQ)) / 2 - fy01) / 2;
	float a00 = f00;
	float a10 = fx00;
	float a20 = -3 * f00 + 3 * f10 - 2 * fx00 - fx10;
	float a30 = 2 * f00 - 2 * f10 + fx00 + fx10;
	float a01 = fy00;
	float a11 = fxy00;
	float a21 = -3 * fy00 + 3 * fy10 - 2 * fxy00 - fxy10;
	float a31 = 2 * fy00 - 2 * fy10 + fxy00 + fxy10;
	float a02 = -3 * f00 + 3 * f01 - 2 * fy00 - fy01;
	float a12 = -3 * fx00 + 3 * fx01 - 2 * fxy00 - fxy01;
	float a22 = 9 * f00 - 9 * f10 - 9 * f01 + 9 * f11 + 6 * fx00 + 3 * fx10 - 6 * fx01 - 3 * fx11 + 6 * fy00 - 6 * fy10 + 3 * fy01 - 3 * fy11 + 4 * fxy00 + 2 * fxy10 + 2 * fxy01 + fxy11;
	float a32 = -6 * f00 + 6 * f10 + 6 * f01 - 6 * f11 - 3 * fx00 - 3 * fx10 + 3 * fx01 + 3 * fx11 - 4 * fy00 + 4 * fy10 - 2 * fy01 + 2 * fy11 - 2 * fxy00 - 2 * fxy10 - fxy01 - fxy11;
	float a03 = 2 * f00 - 2 * f01 + fy00 + fy01;
	float a13 = 2 * fx00 - 2 * fx01 + fxy00 + fxy01;
	float a23 = -6 * f00 + 6 * f10 + 6 * f01 - 6 * f11 - 4 * fx00 - 2 * fx10 + 4 * fx01 + 2 * fx11 - 3 * fy00 + 3 * fy10 - 3 * fy01 + 3 * fy11 - 2 * fxy00 - fxy10 - 2 * fxy01 - fxy11;
	float a33 = 4 * f00 - 4 * f10 - 4 * f01 + 4 * f11 + 2 * fx00 + 2 * fx10 - 2 * fx01 - 2 * fx11 + 2 * fy00 - 2 * fy10 + 2 * fy01 - 2 * fy11 + fxy00 + fxy10 + fxy01 + fxy11;
	return a00 + a10*iR + a20*pown(iR, 2) + a30*pown(iR, 3) + jR*(a01 + a11*iR + a21*pown(iR, 2) + a31*pown(iR, 3)) + pown(jR, 2)*(a02 + a12*iR + a22*pown(iR, 2) + a32*pown(iR, 3)) + pown(jR, 3)*(a03 + a13*iR + a23*pown(iR, 2) + a33*pown(iR, 3));
}
void Advection_SMI(int i, int j, global float *U, global float *V, float dt, float d) {
	float U_interpolated = F(U, i, j);
	float V_interpolated = F(V, i, j);
	if (i > 0 && i < RERES - 1 && j > 0 && j < RERES - 1) {
		float i_d = (i - F(U, i, j)*dt / d);
		float j_d = (j - F(V, i, j)*dt / d);
		i_d = fmin(fmax(i_d, 0.0f), 1.0f*(RERES - 1));
		j_d = fmin(fmax(j_d, 0.0f), 1.0f*(RERES - 1));
		int i_dinte = (int)i_d;
		int j_dinte = (int)j_d;
		float i_dfrac = i_d - (float)i_dinte;
		float j_dfrac = j_d - (float)j_dinte;
		U_interpolated = F(U, i_dinte, j_dinte)*(1.0f - i_dfrac)*(1.0f - j_dfrac) + F(U, i_dinte, j_dinte + 1)*(1.0f - i_dfrac)*(j_dfrac)+F(U, i_dinte + 1, j_dinte)*(i_dfrac)*(1.0f - j_dfrac) + F(U, i_dinte + 1, j_dinte + 1)*(i_dfrac)*(j_dfrac);
		V_interpolated = F(V, i_dinte, j_dinte)*(1.0f - i_dfrac)*(1.0f - j_dfrac) + F(V, i_dinte, j_dinte + 1)*(1.0f - i_dfrac)*(j_dfrac)+F(V, i_dinte + 1, j_dinte)*(i_dfrac)*(1.0f - j_dfrac) + F(V, i_dinte + 1, j_dinte + 1)*(i_dfrac)*(j_dfrac);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	F(U, i, j) = U_interpolated;
	F(V, i, j) = V_interpolated;
	barrier(CLK_GLOBAL_MEM_FENCE);
}
void Advection(int i, int j, global float *A, global float *UR, global float *VR, float dt, float d) {
	float A_interpolated = G(A, i, j);
	if (i > 0 && i < RERE_S_DOUBLED - 1 && j > 0 && j < RERE_S_DOUBLED - 1) {
		float i_d = (i - G(UR, i, j)*dt / d);
		float j_d = (j - G(VR, i, j)*dt / d);
		i_d = fmin(fmax(i_d, 0.0f), 1.0f*(RERE_S_DOUBLED - 1));
		j_d = fmin(fmax(j_d, 0.0f), 1.0f*(RERE_S_DOUBLED - 1));
		int i_dinte = (int)i_d;
		int j_dinte = (int)j_d;
		float i_dfrac = i_d - (float)i_dinte;
		float j_dfrac = j_d - (float)j_dinte;
		A_interpolated = G(A, i_dinte, j_dinte)*(1.0f - i_dfrac)*(1.0f - j_dfrac) + G(A, i_dinte, j_dinte + 1)*(1.0f - i_dfrac)*(j_dfrac)+G(A, i_dinte + 1, j_dinte)*(i_dfrac)*(1.0f - j_dfrac) + G(A, i_dinte + 1, j_dinte + 1)*(i_dfrac)*(j_dfrac);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	G(A, i, j) = A_interpolated;
	barrier(CLK_GLOBAL_MEM_FENCE);
}

void velocityAdvection_Bicubic(int i, int j, global float *U, global float *V, float dt, float d)
{
	float U_interpolated = F(U, i, j);
	float V_interpolated = F(V, i, j);
	if (i > 0 && i < RERES - 1 && j > 0 && j < RERES - 1) {
		float i_d = (i - F(U, i, j)*dt / d);
		float j_d = (j - F(V, i, j)*dt / d);
		i_d = fmin(fmax(i_d, 2.0f), 1.0f*(RERES - 3));
		j_d = fmin(fmax(j_d, 2.0f), 1.0f*(RERES - 3));
		U_interpolated = Bicubic(i_d, j_d, U);
		V_interpolated = Bicubic(i_d, j_d, V);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	F(U, i, j) = U_interpolated;
	F(V, i, j) = V_interpolated;
	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void PaintLatteArt(global uchar *R)
{
	size_t n = get_global_id(0);
	int i = n%RERE_S_DOUBLED;
	int j = n / RERE_S_DOUBLED;
	G_R(R, i, j) = 0;
	G_G(R, i, j) = 0;
	G_B(R, i, j) = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);
	G_R(R, i, j) = 205;
	G_G(R, i, j) = 133;
	G_B(R, i, j) = 63;
	if (pown(i*1.0f - RERE_S_DOUBLED / 2.0f, 2) + pown(j*1.0f - RERE_S_DOUBLED / 2.0f, 2) > pown(RERE_S_DOUBLED / 3.0f, 2)) {
		G_R(R, i, j) = 0;
		G_G(R, i, j) = 0;
		G_B(R, i, j) = 0;
	}
}

kernel void HomographyTransformation(global uchar *R,global uchar *RR,const float a1,const float a2,const float a3,const float a4,const float a5,const float a6,const float a7,const float a8)
{
	size_t n = get_global_id(0);
	int i = n%RERE_S_DOUBLED;
	int j = n / RERE_S_DOUBLED;
	float X = i*1.0f / RERE_S_DOUBLED;
	float Y = j*1.0f / RERE_S_DOUBLED;
	float X_D = (a1*X + a2*Y + a3) / (a7*X + a8*Y + 1);
	float Y_D = (a4*X + a5*Y + a6) / (a7*X + a8*Y + 1);
	G_R(RR, i, j) = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (pown(X - 0.5f, 2) + pown(Y - 0.5f, 2) < pown(1/3.0f, 2)) {
		if (pown(X_D - 0.5f, 2) + pown(Y_D - 0.5f, 2) > pown(1/3.0f, 2)) {
			G_R(RR, i, j) = 255;
		}
	}
}


kernel void Rkuro(global uchar *R) {
	size_t n = get_global_id(0);
	R[4 * n] = 0;
	R[4 * n + 1] = 0;
	R[4 * n + 2] = 0;
	R[4 * n + 3] = 255;
}