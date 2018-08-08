#include <iostream>
#include <fstream>
#include <string>

#include "GLcommon.h"
#include "simpleCL.h"
#include "simpleCLheaderGenerate.h"

#define F_RES 256
#define R_RES 512


extern void SimulateFluid(simpleCLhandler & cl,kernelHandler & hndl);

int print()
{
	printf("-----------------------------------------\n");
	printf("-----------------------------------------\n");
	printf("press any key to continue...\n\n");
	char p[10];
	fgets(p, 10, stdin);
	printf("\n-----------------------------------------\n");
	printf("EXIT\n");
	return 0;
}

void projectionCoef(float f00x,float f00y,float f10x,float f10y,float f01x,float f01y,float f11x,float f11y,float *a, float *b, float *c, float *d, float *e, float *f, float *g, float *h)
{
	*a = -((f01x*f10x - f00x*f01x)*f11y + (-f00x*f10y + (f00y - f01y)*f10x + f00x*f01y)*f11x + f00x*f01x*f10y - f00y*f01x*f10x) / ((f10x - f01x)*f11y + (f01y - f10y)*f11x + f01x*f10y - f01y*f10x);
	*b = ((f01x - f00x)*f10x*f11y + ((f00x - f01x)*f10y - f00x*f01y + f00y*f01x)*f11x + (f00x*f01y - f00y*f01x)*f10x) / ((f10x - f01x)*f11y + (f01y - f10y)*f11x + f01x*f10y - f01y*f10x);
	*c = f00x;
	*d = -(((f01x - f00x)*f10y + f00y*f10x - f00y*f01x)*f11y + (f00y*f01y - f01y*f10y)*f11x + f00x*f01y*f10y - f00y*f01y*f10x) / ((f10x - f01x)*f11y + (f01y - f10y)*f11x + f01x*f10y - f01y*f10x);
	*e = (((f01y - f00y)*f10x - f00x*f01y + f00y*f01x)*f11y + (f00y - f01y)*f10y*f11x + (f00x*f01y - f00y*f01x)*f10y) / ((f10x - f01x)*f11y + (f01y - f10y)*f11x + f01x*f10y - f01y*f10x);
	*f = f00y;
	*g = -((f10x - f00x)*f11y + (f00y - f10y)*f11x + f01x*f10y - f01y*f10x + f00x*f01y - f00y*f01x) / ((f10x - f01x)*f11y + (f01y - f10y)*f11x + f01x*f10y - f01y*f10x);
	*h = ((f01x - f00x)*f11y + (f00y - f01y)*f11x + (f00x - f01x)*f10y + (f01y - f00y)*f10x) / ((f10x - f01x)*f11y + (f01y - f10y)*f11x + f01x*f10y - f01y*f10x);
}
void projectionParamChange(GLFWwindow *gwin,float *f00x, float *f00y, float *f10x, float *f10y, float *f01x, float *f01y, float *f11x, float *f11y)
{
	float val = 0.05f;
	if (glfwGetKey(gwin, GLFW_KEY_LEFT) == GLFW_PRESS)
	{
		if      (glfwGetKey(gwin, GLFW_KEY_Z) == GLFW_PRESS)*f00x -= val;
		else if (glfwGetKey(gwin, GLFW_KEY_X) == GLFW_PRESS)*f10x -= val;
		else if (glfwGetKey(gwin, GLFW_KEY_C) == GLFW_PRESS)*f01x -= val;
		else if (glfwGetKey(gwin, GLFW_KEY_V) == GLFW_PRESS)*f11x -= val;
	}
	else if (glfwGetKey(gwin, GLFW_KEY_RIGHT) == GLFW_PRESS)
	{
		if      (glfwGetKey(gwin, GLFW_KEY_Z) == GLFW_PRESS)*f00x += val;
		else if (glfwGetKey(gwin, GLFW_KEY_X) == GLFW_PRESS)*f10x += val;
		else if (glfwGetKey(gwin, GLFW_KEY_C) == GLFW_PRESS)*f01x += val;
		else if (glfwGetKey(gwin, GLFW_KEY_V) == GLFW_PRESS)*f11x += val;
	}
	else if (glfwGetKey(gwin, GLFW_KEY_UP) == GLFW_PRESS)
	{
		if      (glfwGetKey(gwin, GLFW_KEY_Z) == GLFW_PRESS)*f00y += val;
		else if (glfwGetKey(gwin, GLFW_KEY_X) == GLFW_PRESS)*f10y += val;
		else if (glfwGetKey(gwin, GLFW_KEY_C) == GLFW_PRESS)*f01y += val;
		else if (glfwGetKey(gwin, GLFW_KEY_V) == GLFW_PRESS)*f11y += val;
	}
	else if (glfwGetKey(gwin, GLFW_KEY_DOWN) == GLFW_PRESS)
	{
		if      (glfwGetKey(gwin, GLFW_KEY_Z) == GLFW_PRESS)*f00y -= val;
		else if (glfwGetKey(gwin, GLFW_KEY_X) == GLFW_PRESS)*f10y -= val;
		else if (glfwGetKey(gwin, GLFW_KEY_C) == GLFW_PRESS)*f01y -= val;
		else if (glfwGetKey(gwin, GLFW_KEY_V) == GLFW_PRESS)*f11y -= val;
	}
}

int main(int argc, const char **argv)

{

	simpleCLhandler cl = simpleCL_init();

	kernelHandler krnlhndl;
#ifdef __APPLE__
    krnlhndl.addKernelProgramFile("../../../src/fluidCL/kernel.cl", "fluid", 1);
#else
	krnlhndl.addKernelProgramFile("..\\cudaproj\\src\\kernel.cl", "fluid", 1);
	krnlhndl.addKernelProgramFile("..\\cudaproj\\src\\kernel_latte.cl", "latte", 2);
#endif
	krnlhndl.loadProgramFile();
	krnlhndl.buildProgram(cl);
	krnlhndl.printProgramBuildInfo(cl);

	float dd = 1.0f / F_RES;
	float dt = dd / 10.0f;
	float Re = 8000;
    
    float clrendertexturecoef = 1.0f;
    
	double _mousePosX = 0;
	double _mousePosY = 0;

	float mousePosX = 0;
	float mousePosY = 0;
	float mousePosX_previous = 0;
	float mousePosY_previous = 0;

	float mouse_U = 0;
	float mouse_V = 0;

	int mouseButtonFlg = 0;

	cl_mem P = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem U = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem V = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem Fx = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem Fy = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
    cl_mem W =  clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
    cl_mem CLRenderTexture = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(cl_uchar)*F_RES * F_RES * 4, NULL, NULL);

	cl_mem UR = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * R_RES * R_RES, NULL, NULL);
	cl_mem VR = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * R_RES * R_RES, NULL, NULL);
	cl_mem CLRenderTexture_LATTE = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(cl_uchar)*R_RES * R_RES * 4, NULL, NULL);
	cl_mem CLRenderTexture_LATTE_R = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(cl_uchar)*R_RES * R_RES * 4, NULL, NULL);
    
    cl_mem B = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float)*F_RES*F_RES, NULL, NULL);

    cl_kernel boundarykernel = clCreateKernel(krnlhndl.getProgram(1), "makeBoundary", NULL);
    cl_kernel forcekernel = clCreateKernel(krnlhndl.getProgram(1), "AddForce", NULL);
	cl_kernel fluidkernel = clCreateKernel(krnlhndl.getProgram(1), "kernel_main", NULL);
    cl_kernel vorticitykernel = clCreateKernel(krnlhndl.getProgram(1), "vorticity", NULL);
    cl_kernel convertFloatUCharkernel = clCreateKernel(krnlhndl.getProgram(1), "convertFloatToUchar", NULL);
	cl_kernel upscalekernel = clCreateKernel(krnlhndl.getProgram(1), "XY512Func", NULL);

	cl_kernel kurokernel = clCreateKernel(krnlhndl.getProgram(1), "Rkuro", NULL);
	cl_kernel paintlatteartkernel = clCreateKernel(krnlhndl.getProgram(2), "PaintLatteArt", NULL);
	cl_kernel homographytransformationkernel = clCreateKernel(krnlhndl.getProgram(2), "HomographyTransformation", NULL);

    size_t global_work_size_F[1] = { F_RES * F_RES };
	size_t global_work_size_R[1] = { R_RES * R_RES };
	size_t local_work_size[1] = { 1024 };
    
    clSetKernelArg(boundarykernel, 0, sizeof(cl_mem), &B);
    clEnqueueNDRangeKernel(cl->mainQueue, boundarykernel, 1, NULL, global_work_size_F, NULL, 0, NULL, NULL);

	GLcommon gl;

	gl.createWindowandMakeContext(512, 512);

	float posData[4 * 3] = {
		-1.0f,-1.0f,0.0f,
		 1.0f,-1.0f,0.0f,
		 -1.0f,1.0f,0.0f,
		 1.0f,1.0f,0.0f
	};
	float uvData[4 * 2] = {
		0.0f,0.0f,
		1.0f,0.0f,
		0.0f,1.0f,
		1.0f,1.0f
	};

	gl.VAO_Create(1, "cambus");
	gl.VBO_Create(1, "vbo_pos");
	gl.VBO_Create(2, "vbo_uv");

	gl.VBO_StoreData(1, 3, 4, GL_FLOAT, GL_STATIC_DRAW, GL_FALSE, 0, sizeof(float) * 3 * 4, posData);
	gl.VBO_StoreData(2, 2, 4, GL_FLOAT, GL_STATIC_DRAW, GL_FALSE, 0, sizeof(float) * 2 * 4, uvData);

	gl.Program_Create(1, "program");
#ifdef __APPLE__
    gl.Shader_Create(1, "vertex", GL_VERTEX_SHADER, "../../../src/fluidCL/001.vs");
    gl.Shader_Create(2, "fragment", GL_FRAGMENT_SHADER, "../../../src/fluidCL/001.fs");
    
#else
    gl.Shader_Create(1, "vertex", GL_VERTEX_SHADER, "..\\cudaproj\\src\\001.vs");
    gl.Shader_Create(2, "fragment", GL_FRAGMENT_SHADER, "..\\cudaproj\\src\\001.fs");
#endif
	gl.Program_AttachShader(1, 1);
	gl.Program_AttachShader(1, 2);
	gl.Shader_AddAttribLocation(1, "pos", 0);
	gl.Shader_AddAttribLocation(1, "uv", 1);
	gl.Program_LinkShader(1);

	gl.VAO_VertexAttribArray_Register(1, 1, 1, 0);
	gl.VAO_VertexAttribArray_Register(1, 2, 1, 1);

	
	float *colorTexHstPtr_LATTE = (float *)malloc(sizeof(float) * R_RES * R_RES);
	float *colorTexHstPtr_LATTE_R = (float *)malloc(sizeof(float) * R_RES * R_RES);
	float *colorTexHstPtr = (float *)malloc(sizeof(float) * F_RES * F_RES);
	for (int j = 0; j < F_RES; j++)
	{
		for (int i = 0; i < F_RES; i++)
		{
			colorTexHstPtr[i + F_RES * j] = i*1.0f / F_RES + j*1.0f / F_RES;
		}
	}
	gl.Texture_Create(1, "pattern texture", F_RES, F_RES);
	gl.Texture_Store(1, colorTexHstPtr, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR);
	gl.Texture_Register(1, 1);

	gl.Texture_Create(2, "target_texture", R_RES, R_RES);
	gl.Texture_Store(2, colorTexHstPtr_LATTE, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR);
	gl.Texture_Register(2, 2);

	gl.Texture_Create(3, "target_texture_R", R_RES, R_RES);
	gl.Texture_Store(3, colorTexHstPtr_LATTE_R, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR);
	gl.Texture_Register(3, 3);

	/*gl.Program_USE(1, [&]() {
		//glUniform1i(glGetUniformLocation(gl.getProgram(1), "tex"), 1);
		glUniform1i(glGetUniformLocation(gl.getProgram(1), "tex"), 2);
	});*/

	float f00x = 0, f00y = 0;
	float f10x = 1, f10y = 0;
	float f01x = 0, f01y = 1;
	float f11x = 1, f11y = 1;
	float a, b, c, d, e, f, g, h;
    
	gl.Draw([&]() {

		projectionParamChange(gl.window, &f00x, &f00y, &f10x, &f10y, &f01x, &f01y, &f11x, &f11y);
		projectionCoef(f00x, f00y, f10x, f10y, f01x, f01y, f11x, f11y, &a, &b, &c, &d, &e, &f, &g, &h);
		if (glfwGetKey(gl.window, GLFW_KEY_A) == GLFW_PRESS) {
			f00x = 0; f00y = 0;
			f10x = 1; f10y = 0;
			f01x = 0; f01y = 1;
			f11x = 1; f11y = 1;
		}


		mousePosX_previous = mousePosX;
		mousePosY_previous = mousePosY;
		glfwGetCursorPos(gl.window, &_mousePosX, &_mousePosY);
		mousePosX = _mousePosX * F_RES*1.0f / 512;
		mousePosY = _mousePosY * F_RES*1.0f / 512;
		mousePosX = fmax(fmin(mousePosX, F_RES - 2.0), 1.0);
		mousePosY = fmax(fmin(mousePosY, F_RES - 2.0), 1.0);
		mouse_U = mousePosX - mousePosX_previous;
		mouse_V = mousePosY - mousePosY_previous;
		mouseButtonFlg = (glfwGetMouseButton(gl.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
		printf("%f,%f\n", mousePosX, mousePosY);

		clSetKernelArg(forcekernel, 0, sizeof(cl_mem), &Fx);
		clSetKernelArg(forcekernel, 1, sizeof(cl_mem), &Fy);
		clSetKernelArg(forcekernel, 2, sizeof(cl_float), &mouse_U);
		clSetKernelArg(forcekernel, 3, sizeof(cl_float), &mouse_V);
		clSetKernelArg(forcekernel, 4, sizeof(cl_float), &mousePosX);
		clSetKernelArg(forcekernel, 5, sizeof(cl_float), &mousePosY);
		clSetKernelArg(forcekernel, 6, sizeof(cl_int), &mouseButtonFlg);
		clEnqueueNDRangeKernel(cl->mainQueue, forcekernel, 1, NULL, global_work_size_F, NULL, 0, NULL, NULL);

		clSetKernelArg(fluidkernel, 0, sizeof(cl_mem), &P);
		clSetKernelArg(fluidkernel, 1, sizeof(cl_mem), &U);
		clSetKernelArg(fluidkernel, 2, sizeof(cl_mem), &V);
		clSetKernelArg(fluidkernel, 3, sizeof(cl_mem), &Fx);
		clSetKernelArg(fluidkernel, 4, sizeof(cl_mem), &Fy);
		clSetKernelArg(fluidkernel, 5, sizeof(cl_float), &dd);
		clSetKernelArg(fluidkernel, 6, sizeof(cl_float), &dt);
		clSetKernelArg(fluidkernel, 7, sizeof(cl_float), &Re);
        clSetKernelArg(fluidkernel, 8, sizeof(cl_mem), &B);
		clEnqueueNDRangeKernel(cl->mainQueue, fluidkernel, 1, NULL, global_work_size_F, NULL, 0, NULL, NULL);
        
        clSetKernelArg(vorticitykernel, 0, sizeof(cl_mem), &U);
        clSetKernelArg(vorticitykernel, 1, sizeof(cl_mem), &V);
        clSetKernelArg(vorticitykernel, 2, sizeof(cl_mem), &W);
        clSetKernelArg(vorticitykernel, 3, sizeof(cl_float), &dd);
        clEnqueueNDRangeKernel(cl->mainQueue, vorticitykernel, 1, NULL, global_work_size_F, NULL, 0, NULL, NULL);

		clSetKernelArg(upscalekernel, 0, sizeof(cl_mem), &UR);
		clSetKernelArg(upscalekernel, 1, sizeof(cl_mem), &VR);
		clSetKernelArg(upscalekernel, 2, sizeof(cl_mem), &U);
		clSetKernelArg(upscalekernel, 3, sizeof(cl_mem), &V);
		clEnqueueNDRangeKernel(cl->mainQueue, upscalekernel, 1, NULL, global_work_size_R, NULL, 0, NULL, NULL);



        clSetKernelArg(convertFloatUCharkernel, 0, sizeof(cl_mem), &W);
        clSetKernelArg(convertFloatUCharkernel, 1, sizeof(cl_mem), &CLRenderTexture);
        clSetKernelArg(convertFloatUCharkernel, 2, sizeof(cl_float), &clrendertexturecoef);
        clEnqueueNDRangeKernel(cl->mainQueue, convertFloatUCharkernel, 1, NULL, global_work_size_F, NULL, 0, NULL, NULL);

		clSetKernelArg(paintlatteartkernel, 0, sizeof(cl_mem), &CLRenderTexture_LATTE);
		clEnqueueNDRangeKernel(cl->mainQueue, paintlatteartkernel, 1, NULL, global_work_size_R, NULL, 0, NULL, NULL);

		clSetKernelArg(homographytransformationkernel, 0, sizeof(cl_mem), &CLRenderTexture_LATTE);
		clSetKernelArg(homographytransformationkernel, 1, sizeof(cl_mem), &CLRenderTexture_LATTE_R);
		clSetKernelArg(homographytransformationkernel, 2, sizeof(float), &a);
		clSetKernelArg(homographytransformationkernel, 3, sizeof(float), &b);
		clSetKernelArg(homographytransformationkernel, 4, sizeof(float), &c);
		clSetKernelArg(homographytransformationkernel, 5, sizeof(float), &d);
		clSetKernelArg(homographytransformationkernel, 6, sizeof(float), &e);
		clSetKernelArg(homographytransformationkernel, 7, sizeof(float), &f);
		clSetKernelArg(homographytransformationkernel, 8, sizeof(float), &g);
		clSetKernelArg(homographytransformationkernel, 9, sizeof(float), &h);
		clEnqueueNDRangeKernel(cl->mainQueue, homographytransformationkernel, 1, NULL, global_work_size_R, NULL, 0, NULL, NULL);
        
        
		clEnqueueReadBuffer(cl->mainQueue, CLRenderTexture, CL_FALSE, 0, sizeof(cl_uchar)*F_RES*F_RES * 4, colorTexHstPtr, 0, NULL, NULL);
		clEnqueueReadBuffer(cl->mainQueue, CLRenderTexture_LATTE, CL_FALSE, 0, sizeof(cl_uchar)*R_RES*R_RES * 4, colorTexHstPtr_LATTE, 0, NULL, NULL);
		clEnqueueReadBuffer(cl->mainQueue, CLRenderTexture_LATTE_R, CL_FALSE, 0, sizeof(cl_uchar)*R_RES*R_RES * 4, colorTexHstPtr_LATTE_R, 0, NULL, NULL);
        //clEnqueueReadBuffer(cl->mainQueue, B, CL_FALSE, 0, sizeof(cl_uchar)*F_RES*F_RES * 4, colorTexHstPtr, 0, NULL, NULL);
		
		gl.Texture_Rewrite(1, colorTexHstPtr);
		gl.Texture_Rewrite(2, colorTexHstPtr_LATTE);
		gl.Texture_Rewrite(3, colorTexHstPtr_LATTE_R);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		gl.Program_USE(1);

		if(glfwGetKey(gl.window,GLFW_KEY_Q) == GLFW_PRESS) glUniform1i(glGetUniformLocation(gl.getProgram(1), "tex"), 1);
		else glUniform1i(glGetUniformLocation(gl.getProgram(1), "tex"), 2);

		glUniform1i(glGetUniformLocation(gl.getProgram(1), "subtex"), 3);

		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a1"), a);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a2"), b);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a3"), c);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a4"), d);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a5"), e);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a6"), f);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a7"), g);
		glUniform1f(glGetUniformLocation(gl.getProgram(1), "a8"), h);

		glUniform2f(glGetUniformLocation(gl.getProgram(1), "f00"), f00x, f00y);
		glUniform2f(glGetUniformLocation(gl.getProgram(1), "f10"), f10x, f10y);
		glUniform2f(glGetUniformLocation(gl.getProgram(1), "f01"), f01x, f01y);
		glUniform2f(glGetUniformLocation(gl.getProgram(1), "f11"), f11x, f11y);

		gl.VAO_USE(1);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		gl.VAO_Unbind();
		gl.Program_Unbind();
	});

	FREE_SAFE(colorTexHstPtr_LATTE_R);
	FREE_SAFE(colorTexHstPtr_LATTE);
	FREE_SAFE(colorTexHstPtr);

	clReleaseMemObject(P);
	clReleaseMemObject(U);
	clReleaseMemObject(V);
	clReleaseMemObject(Fx);
	clReleaseMemObject(Fy);
    clReleaseMemObject(W);
    clReleaseMemObject(B);

	clReleaseMemObject(UR);
	clReleaseMemObject(VR);


	clReleaseMemObject(CLRenderTexture_LATTE);
	clReleaseMemObject(CLRenderTexture_LATTE_R);
    clReleaseMemObject(CLRenderTexture);


	simpleCL_close(cl);

	return print();
}
